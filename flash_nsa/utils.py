import torch
import triton
from functools import wraps

device_capability = torch.cuda.get_device_capability()
major, minor = device_capability
IS_HOPPER = True if major >= 9 else False


def cu_seqlens_to_seqlens(cu_seqlens):
    return cu_seqlens[1:] - cu_seqlens[:-1]

def seqlens_to_cu_seqlens(seqlens):
    cu_seqlens = seqlens.cumsum(-1)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, [1, 0])
    return cu_seqlens.to(torch.int32)

def get_cp_cu_seqlens(cu_seqlens, bos, eos):
    tmp = [bos]
    batch = []
    for idx, val in enumerate(cu_seqlens):
        if val >= eos:
            tmp.append(eos)
            batch.append(idx-1)
            break
        if val == bos:
            pass
        elif val > bos:
            tmp.append(val)
            batch.append(idx-1)
    return tmp, batch

def concat_cp_cu_seqlens(cu_seqlens, *args):
    cp_cu_seqlens = [0]
    cp_batch_idx = []
    cp_offset = []
    end = cp_cu_seqlens[-1]
    for part_cu_seqlens, part_batch_idx in args:
        start = part_cu_seqlens[0]
        for idx in range(len(part_batch_idx)):
            cp_cu_seqlens.append(end + part_cu_seqlens[idx + 1] - start)
            cp_batch_idx.append(part_batch_idx[idx])
            cp_offset.append(part_cu_seqlens[idx] - cu_seqlens[part_batch_idx[idx]])
        end = cp_cu_seqlens[-1]
    return cp_cu_seqlens, cp_batch_idx, cp_offset 


class NSAHelper:
    is_hopper = IS_HOPPER

    # base infos, it will not change in training
    kernel_size = 32
    stride = 16
    block_size = 64
    window_size = 512
    top_n = 16
    num_init_blocks = 1
    num_local_blocks = 2

    cmp_k_method = "mean"
    cmp_v_method = "mean"

    recompute_swa_o = False
    recompute_slc_o = False
    recompute_cmp_kv = False
    recompute_cmp_o = False

    is_context_parallel_enable = True
    pp_size = 1

    # varlen infos, maybe change according to input data
    x_cu_seqlens = None
    x_seqlens = None
    x_maxlen = None

    y_cu_seqlens = None
    y_seqlens = None
    y_maxlen = None
    y_len = None

    k_cu_seqlens = None
    k_seqlens = None
    k_maxlen = None
    k_len = None

    process_group = None
    rank = 0
    world_size = 1
    cp_mode = 2
    kv_head_stride = 1
    bench_mode = False

    seqlen = None
    cp_bos1 = None
    cp_bos2 = None
    cp_eos1 = None
    cp_eos2 = None
    cp_offset = None
    cp_batch_idx = None
    cp_cu_seqlens = None
    cp_seqlens = None
    cp_maxlen = None
    atomic_add_dkdv = None
    cp_k_cu_seqlens = None
    cp_k_seqlens = None
    cp_k_maxlen = None
    cp_k_len = None

    # SWA related variables
    use_overlap_swa = False
    swa_q_start_1 = None
    swa_kv_start_1 = None
    swa_cu_seqlens_q1 = None
    swa_cu_seqlens_k1 = None
    swa_cu_seqlens_q2 = None
    swa_cu_seqlens_k2 = None
    swa_max_q1 = None
    swa_max_k1 = None
    swa_max_q2 = None
    swa_max_k2 = None
    swa_kv_start_2 = None
    swa_kv_end_2 = None

    swa_q_start_2 = None
    swa_kv_start_3 = None
    swa_cu_seqlens_q3 = None
    swa_cu_seqlens_k3 = None
    swa_cu_seqlens_q4 = None
    swa_cu_seqlens_k4 = None
    swa_max_q3 = None
    swa_max_k3 = None
    swa_max_q4 = None
    swa_max_k4 = None
    swa_kv_start_4 = None
    swa_kv_end_4 = None

    @classmethod
    def set_hyperparameters(cls, kernel_size=32, stride=16, block_size=64, window_size=512, top_n=16, num_init_blocks=1, num_local_blocks=2):
        cls.kernel_size = kernel_size
        cls.stride = stride
        cls.block_size = block_size
        cls.window_size = window_size
        cls.top_n = top_n
        cls.num_init_blocks = num_init_blocks
        cls.num_local_blocks = num_local_blocks

    @classmethod
    def set_recompute(cls, recompute_swa_o=False, recompute_slc_o=False, recompute_cmp_kv=False, recompute_cmp_o=False):
        # for 8k pretraing, I suggest this setting:
        cls.recompute_swa_o = recompute_swa_o # 🌟🌟🌟🌟🌟
        cls.recompute_slc_o = recompute_slc_o # 🌟🌟
        cls.recompute_cmp_kv = recompute_cmp_kv # 🌟🌟🌟🌟🌟
        cls.recompute_cmp_o = recompute_cmp_o # 🌟🌟🌟🌟🌟

    @classmethod
    def set_pipeline_parallel_world_size(cls, pp_size):
        cls.pp_size = pp_size

    @classmethod   
    def set_context_parallel(cls, process_group=None, kv_head_stride=1, use_overlap_swa=True):
        if process_group is None:
            return

        if not torch.distributed.is_initialized():
            return 

        world_size = torch.distributed.get_world_size(process_group)
        if world_size <= 1:
            return

        cls.process_group = process_group
        cls.kv_head_stride = kv_head_stride
        cls.rank = torch.distributed.get_rank(process_group)
        cls.world_size = world_size
        cls.use_overlap_swa = use_overlap_swa

    @classmethod
    def set_cmp_kv_method(cls, cmp_k_method, cmp_v_method):
        cls.cmp_k_method = cmp_k_method
        cls.cmp_v_method = cmp_v_method

    @classmethod
    def enable_cp(cls):
        cls.is_context_parallel_enable = True

    @classmethod
    def disable_cp(cls):
        cls.is_context_parallel_enable = False

    @classmethod
    def enable_bench_mode(cls):
        cls.bench_mode = True

    @classmethod
    def disable_bench_mode(cls):
        cls.bench_mode = False

    # @classmethod
    # def enable_cuda_graph(cls):
    #     cls.fix_tensor_shape = True

    # @classmethod
    # def disable_cuda_graph(cls):
    #     cls.fix_tensor_shape = False

    @classmethod
    def _need_init(cls, x_cu_seqlens):
        # if x_cu_seqlens == cls.x_cu_seqlens, skip init
        if cls.x_cu_seqlens is None:
            return True
        elif len(cls.x_cu_seqlens) != len(x_cu_seqlens):
            return True
        elif any(cls.x_cu_seqlens != x_cu_seqlens):
            return True
        else:
            return False

    @classmethod
    def _init_xyk_cu_seqlens(cls, x_cu_seqlens: torch.Tensor):
        # normal cu_seqlens
        cls.x_cu_seqlens = x_cu_seqlens.to(torch.int32)
        cls.x_seqlens = cu_seqlens_to_seqlens(x_cu_seqlens)
        cls.x_maxlen = cls.x_seqlens.max().item()
        # cmp_blocks cu_seqlens
        cls.y_seqlens = (cls.x_seqlens - cls.kernel_size) // cls.stride + 1
        cls.y_seqlens.masked_fill_(cls.y_seqlens<0, 0)
        cls.y_maxlen = cls.y_seqlens.max().item()
        cls.y_cu_seqlens = seqlens_to_cu_seqlens(cls.y_seqlens).to(torch.int32)
        cls.y_len = cls.y_cu_seqlens[-1].item()
        # slc_blocks cu_seqlens
        cls.k_seqlens = triton.cdiv(cls.x_seqlens, cls.block_size)
        cls.k_maxlen = cls.k_seqlens.max().item()
        cls.k_cu_seqlens = seqlens_to_cu_seqlens(cls.k_seqlens).to(torch.int32)
        cls.k_len = cls.k_cu_seqlens[-1].item()


    @classmethod
    def init_cu_seqlens(cls, x_cu_seqlens: torch.Tensor, x_cu_seqlens_np=None, cp_mode=1):
        if not cls._need_init(x_cu_seqlens):
            return

        cls._init_xyk_cu_seqlens(x_cu_seqlens)

        if cls.process_group is None:
            return

        cls._init_cp_cu_seqlens(x_cu_seqlens_np, cp_mode)

    @classmethod
    def clear_tensor_data(cls, *tensor):
        for t in tensor:
            if isinstance(t, torch.Tensor):
                t.data = torch.Tensor()
                del t

    @classmethod
    def split_d(cls, d):
        d_power_2 = triton.next_power_of_2(d)
        if d == d_power_2:
            return d, 0
        else:
            d1 = d_power_2 // 2
            d2 = d - d1
            assert d2 == triton.next_power_of_2(d2)
        return d1, d2

    @classmethod  
    def get_bwd_helper(cls):
        if cls.pp_size <= 1:
            return cls
        else:
            return BwdNSAHelper()

    @classmethod
    def _init_cp_cu_seqlens(cls, x_cu_seqlens_np=None, cp_mode=1):
        assert cp_mode == 1 or cp_mode == 2
        cls.cp_mode = cp_mode
        if x_cu_seqlens_np is None:
            x_cu_seqlens_np = cls.x_cu_seqlens.cpu().numpy()
        S = x_cu_seqlens_np[-1]

        if cp_mode == 2:
            assert S % (2 * cls.world_size) == 0
            chunk_size = S // (2 * cls.world_size)
            cls.cp_bos1 = chunk_size * cls.rank
            cls.cp_bos2 = chunk_size * (2 * cls.world_size - 1 - cls.rank)
            cls.cp_eos1 = cls.cp_bos1 + chunk_size
            cls.cp_eos2 = cls.cp_bos2 + chunk_size
            cp_cu_seqlens, cp_batch_idx, cp_offset = concat_cp_cu_seqlens(x_cu_seqlens_np,
                                        get_cp_cu_seqlens(x_cu_seqlens_np, cls.cp_bos1, cls.cp_eos1),
                                        get_cp_cu_seqlens(x_cu_seqlens_np, cls.cp_bos2, cls.cp_eos2),
                                                )
        else:
            assert S % cls.world_size == 0
            chunk_size = S // cls.world_size
            cls.cp_bos1 = chunk_size * cls.rank
            cls.cp_eos1 = cls.cp_bos1 + chunk_size
            cp_cu_seqlens, cp_batch_idx, cp_offset = concat_cp_cu_seqlens(x_cu_seqlens_np,
                                        get_cp_cu_seqlens(x_cu_seqlens_np, cls.cp_bos1, cls.cp_eos1),
                                                )

        cls.cp_cu_seqlens = torch.tensor(cp_cu_seqlens, device=cls.x_cu_seqlens.device, dtype=torch.int32)

        cls.cp_batch_idx = torch.tensor(cp_batch_idx, device=cls.x_cu_seqlens.device, dtype=torch.int32)
        cls.atomic_add_dkdv = len(set(cp_batch_idx)) < len(cp_batch_idx)
        cls.cp_offset = torch.tensor(cp_offset, device=cls.x_cu_seqlens.device, dtype=torch.int32)
        cls.cp_seqlens = cu_seqlens_to_seqlens(cls.cp_cu_seqlens)
        cls.cp_maxlen = cls.cp_seqlens.max().item()

        cls.cp_k_seqlens = triton.cdiv(cls.cp_seqlens + cls.cp_offset, cls.block_size)
        cls.cp_k_cu_seqlens = seqlens_to_cu_seqlens(cls.cp_k_seqlens)
        cls.cp_k_maxlen = cls.cp_k_seqlens.max().item()
        cls.cp_k_len = cls.cp_k_cu_seqlens[-1].item()

        if not cls.use_overlap_swa:
            if cp_mode == 1:
                bacth_start_idx = cp_batch_idx[0]
                start = x_cu_seqlens_np[bacth_start_idx]
                diff = min(cls.cp_bos1 - start, cls.window_size)
                cls.swa_cu_seqlens_q2 = cls.cp_cu_seqlens
                cls.swa_cu_seqlens_k2 = cls.swa_cu_seqlens_q2 + diff
                cls.swa_cu_seqlens_k2[0] = 0
                cls.swa_max_q2 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_q2).max().item()
                cls.swa_max_k2 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_k2).max().item()
                cls.swa_kv_start_2 = cls.cp_bos1 - diff

                cls.swa_q_start_1 = chunk_size
                cls.swa_kv_end_2 = cls.cp_eos1


            elif cp_mode == 2:
                point = cp_cu_seqlens.index(chunk_size)
                bacth_start_idx = cp_batch_idx[0]
                start = x_cu_seqlens_np[bacth_start_idx]
                diff = min(cls.cp_bos1 - start, cls.window_size)

                cls.swa_cu_seqlens_q2 = cls.cp_cu_seqlens[:point+1]
                cls.swa_cu_seqlens_k2 = cls.swa_cu_seqlens_q2 + diff
                cls.swa_cu_seqlens_k2[0] = 0
                cls.swa_max_q2 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_q2).max().item()
                cls.swa_max_k2 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_k2).max().item()
                cls.swa_kv_start_2 = cls.cp_bos1 - diff

                cls.swa_q_start_1 = chunk_size
                cls.swa_kv_end_2 = cls.cp_eos1

                bacth_start_idx = cp_batch_idx[point]
                start = x_cu_seqlens_np[bacth_start_idx]
                diff = min(cls.cp_bos2 - start, cls.window_size)
                cls.swa_cu_seqlens_q4 = cls.cp_cu_seqlens[point:] - chunk_size
                cls.swa_cu_seqlens_k4 = cls.swa_cu_seqlens_q4 + diff
                cls.swa_cu_seqlens_k4[0] = 0
                cls.swa_max_q4 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_q4).max().item()
                cls.swa_max_k4 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_k4).max().item()
                cls.swa_kv_start_4 = cls.cp_bos2 - diff

                cls.swa_q_start_2 = chunk_size
                cls.swa_kv_end_4 = cls.cp_eos2

        else:
            if cp_mode == 1:
                bacth_start_idx = cp_batch_idx[0]
                start = x_cu_seqlens_np[bacth_start_idx]
                diff = min(cls.cp_bos1 - start, cls.window_size)

                if cp_cu_seqlens[1] <= cls.window_size:
                    cls.swa_q_start_1 = cp_cu_seqlens[1]
                    cls.swa_kv_start_1 = cls.swa_q_start_1
                    cls.swa_cu_seqlens_q1 = cls.cp_cu_seqlens[1:]
                    cls.swa_cu_seqlens_q1 = cls.swa_cu_seqlens_q1 - cls.swa_q_start_1
                    cls.swa_cu_seqlens_k1 = cls.swa_cu_seqlens_q1
                    cls.swa_cu_seqlens_q2 = cls.cp_cu_seqlens[:2]
                    cls.swa_cu_seqlens_k2 = cls.swa_cu_seqlens_q2 + diff
                    cls.swa_cu_seqlens_k2[0] = 0
                else:
                    cls.swa_q_start_1 = cls.window_size
                    cls.swa_kv_start_1 = 0
                    cls.swa_cu_seqlens_q1 = cls.cp_cu_seqlens - cls.window_size
                    cls.swa_cu_seqlens_q1[0] = 0
                    cls.swa_cu_seqlens_k1 = cls.cp_cu_seqlens
                    cls.swa_cu_seqlens_q2 = torch.tensor([0, cls.window_size], device=cls.x_cu_seqlens.device, dtype=torch.int32)
                    cls.swa_cu_seqlens_k2 = cls.swa_cu_seqlens_q2 + diff
                    cls.swa_cu_seqlens_k2[0] = 0
                cls.swa_max_q1 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_q1).max().item()
                cls.swa_max_k1 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_k1).max().item()
                cls.swa_max_q2 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_q2).max().item()
                cls.swa_max_k2 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_k2).max().item()
                cls.swa_kv_start_2 = cls.cp_bos1 - diff
                cls.swa_kv_end_2 = cls.swa_q_start_1 + cls.cp_bos1
            elif cp_mode==2:
                point = cp_cu_seqlens.index(chunk_size)
                bacth_start_idx = cp_batch_idx[0]
                start = x_cu_seqlens_np[bacth_start_idx]
                # diff = cls.cp_bos1 - start
                diff = min(cls.cp_bos1 - start, cls.window_size)

                if cp_cu_seqlens[1] <= cls.window_size:
                    cls.swa_q_start_1 = cp_cu_seqlens[1]
                    cls.swa_kv_start_1 = cls.swa_q_start_1
                    cls.swa_cu_seqlens_q1 = cls.cp_cu_seqlens[1:point + 1] - cls.swa_q_start_1
                    cls.swa_cu_seqlens_k1 = cls.swa_cu_seqlens_q1
                    cls.swa_cu_seqlens_q2 = cls.cp_cu_seqlens[:2]
                    cls.swa_cu_seqlens_k2 = cls.swa_cu_seqlens_q2 + diff
                    cls.swa_cu_seqlens_k2[0] = 0
                else:
                    cls.swa_q_start_1 = cls.window_size
                    cls.swa_kv_start_1 = 0
                    cls.swa_cu_seqlens_q1 = cls.cp_cu_seqlens[:point + 1] - cls.window_size
                    cls.swa_cu_seqlens_q1[0] = 0
                    cls.swa_cu_seqlens_k1 = cls.cp_cu_seqlens[:point + 1]
                    cls.swa_cu_seqlens_q2 = torch.tensor([0, cls.window_size], device=cls.x_cu_seqlens.device, dtype=torch.int32)
                    cls.swa_cu_seqlens_k2 = cls.swa_cu_seqlens_q2 + diff
                    cls.swa_cu_seqlens_k2[0] = 0
                cls.swa_max_q1 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_q1).max().item()
                cls.swa_max_k1 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_k1).max().item()
                cls.swa_max_q2 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_q2).max().item()
                cls.swa_max_k2 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_k2).max().item()
                cls.swa_kv_start_2 = cls.cp_bos1 - diff
                cls.swa_kv_end_2 = cls.swa_q_start_1 + cls.cp_bos1

                bacth_start_idx = cp_batch_idx[point]
                start = x_cu_seqlens_np[bacth_start_idx]
                # diff = cls.cp_bos2 - start
                diff = min(cls.cp_bos2 - start, cls.window_size)

                if cp_cu_seqlens[point + 1] - chunk_size <= cls.window_size:
                    cls.swa_q_start_2 = cp_cu_seqlens[point + 1] - chunk_size
                    cls.swa_kv_start_3 = cls.swa_q_start_2
                    cls.swa_cu_seqlens_q3 = cls.cp_cu_seqlens[point + 1: ]
                    cls.swa_cu_seqlens_q3 = cls.swa_cu_seqlens_q3 - cls.swa_q_start_2 - chunk_size
                    cls.swa_cu_seqlens_k3 = cls.swa_cu_seqlens_q3
                    cls.swa_cu_seqlens_q4 = cls.cp_cu_seqlens[point:point + 2] - chunk_size
                    cls.swa_cu_seqlens_k4 = cls.swa_cu_seqlens_q4 + diff
                    cls.swa_cu_seqlens_k4[0] = 0
                else:
                    cls.swa_q_start_2 = cls.window_size
                    cls.swa_kv_start_3 = 0
                    cls.swa_cu_seqlens_q3 = cls.cp_cu_seqlens[point:] - cls.window_size - chunk_size
                    cls.swa_cu_seqlens_q3[0] = 0
                    cls.swa_cu_seqlens_k3 = cls.cp_cu_seqlens[point:] - chunk_size
                    cls.swa_cu_seqlens_q4 = torch.tensor([0, cls.window_size], device=cls.x_cu_seqlens.device, dtype=torch.int32)
                    cls.swa_cu_seqlens_k4 = cls.swa_cu_seqlens_q4 + diff
                    cls.swa_cu_seqlens_k4[0] = 0
                cls.swa_max_q3 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_q3).max().item()
                cls.swa_max_k3 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_k3).max().item()
                cls.swa_max_q4 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_q4).max().item()
                cls.swa_max_k4 = cu_seqlens_to_seqlens(cls.swa_cu_seqlens_k4).max().item()
                cls.swa_kv_start_4 = cls.cp_bos2 - diff
                cls.swa_kv_end_4 = cls.swa_q_start_2 + cls.cp_bos2
            else:
                assert cp_mode in [1, 2], "mode must be 1 or 2,  1: split data into cp chunks,  2: split data into 2xcp chunks"



class BwdNSAHelper:
    __slots__ = [
        'x_cu_seqlens', 'x_maxlen', 
        'y_cu_seqlens', 'y_maxlen',
        'k_cu_seqlens', 'k_maxlen', "k_len", "cp_k_len",
        "cp_mode", 'cp_offset', 'cp_batch_idx', 'cp_cu_seqlens', 'cp_maxlen',
        'atomic_add_dkdv', 'cp_k_cu_seqlens',  'cp_k_maxlen',
        'swa_q_start_1', 'swa_kv_start_1', 'swa_cu_seqlens_q1', 'swa_cu_seqlens_k1',
        'swa_cu_seqlens_q2', 'swa_cu_seqlens_k2', 'swa_max_q1', 'swa_max_k1',
        'swa_max_q2', 'swa_max_k2', 'swa_kv_start_2', 'swa_kv_end_2',
        'swa_q_start_2', 'swa_kv_start_3', 'swa_cu_seqlens_q3', 'swa_cu_seqlens_k3',
        'swa_cu_seqlens_q4', 'swa_cu_seqlens_k4', 'swa_max_q3', 'swa_max_k3',
        'swa_max_q4', 'swa_max_k4', 'swa_kv_start_4', 'swa_kv_end_4',
    ]

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, getattr(NSAHelper, attr))


def set_allocator():
    device = torch.cuda.current_device()
    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

def use_tma(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        set_allocator()

        return func(*args, **kwargs)
    return wrapper


