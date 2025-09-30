import os
import time
import random
import argparse
import warnings
import numpy as np
import pandas as pd

import torch
import triton
import torch.distributed as dist

from flash_nsa.utils import NSAHelper

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
warnings.filterwarnings("ignore")

DTYPE = torch.bfloat16

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', default='nsa', choices=["ag", "swa", "slc", "cmp", "topk", "nsa", "1M"])

    parser.add_argument('--num-heads', default=64, type=int)
    parser.add_argument('--num-kv-heads', default=4, type=int)
    parser.add_argument('--qk-head-dim', default=128, type=int)
    parser.add_argument('--v-head-dim', default=128, type=int)

    parser.add_argument('--seqlen', default=8192*8, type=int)
    parser.add_argument('--mean', default=8192*8, type=int)
    parser.add_argument('--std', default=0, type=int)

    parser.add_argument('--cp-mode', default=1, type=int, choices=[1, 2])
    parser.add_argument('--use-overlap-swa', action="store_true")
    return parser.parse_args()

def init_distributed():
    dist.init_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group(list(range(world_size)))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    seed = torch.randint(0, 999, (1, ), dtype=torch.int32, device=torch.cuda.current_device())
    dist.all_reduce(seed)
    random.seed(seed.item())
    random.seed(40)
    return rank, world_size, group


def print_rank0(*args, **kwargs):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(*args, **kwargs)
        torch.distributed.barrier()
    else:
        print(*args, **kwargs)

def compare(x, y, prefix=""):
    if x is None or y is None:
        return
    if any([x.dtype == torch.float32, y.dtype==torch.float32]):
        x,y = x.float(), y.float()
    diff = (x-y).abs()
    a = diff.max().item()
    b = diff.mean().item()
    c = torch.max(x.abs().max(), y.abs().max()).item()
    print(prefix + f"max_diff: {a}, mean_diff: {b}")
    return [a, b, c]

def get_memory():
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024
    return round(allocated, 2)

def generate_cu_seqlens(end=8192, mean=2048, var=512):
    r = [0]
    while r[-1] < end:
        a = random.randint(mean-var, mean+var)
        r.append(r[-1] + a)
    r[-1] = end
    cu_seqlens = torch.tensor(r, device=torch.cuda.current_device(), dtype=torch.int32)
    return cu_seqlens

def generate_qkvdo(args, cmp_kv=False):
    QH = args.num_heads
    KH = args.num_kv_heads
    D = args.qk_head_dim
    VD = args.v_head_dim
    device = torch.cuda.current_device()
    x_cu_seqlens, y_cu_seqlens = NSAHelper.x_cu_seqlens, NSAHelper.y_cu_seqlens
    q_len = x_cu_seqlens[-1].item() // NSAHelper.world_size
    kv_len = y_cu_seqlens[-1].item() if cmp_kv else q_len
    q = torch.randn(q_len, QH, D, device=device, dtype=DTYPE).requires_grad_(True)
    k = torch.randn(kv_len, KH, D, device=device, dtype=DTYPE).requires_grad_(True)
    v = torch.randn(kv_len, KH, VD, device=device, dtype=DTYPE).requires_grad_(True)
    do = torch.randn(q_len, QH, VD, device=device, dtype=DTYPE)
    if cmp_kv:
        broadcast(k)
        broadcast(v)
    return q, k, v, do

def gather_tensors(*tensors):
    y = []
    for t in tensors:
        y_i = all_gather(t)
        y_i = y_i.detach().requires_grad_(True)
        y.append(y_i)
    return y if len(y) > 1 else y[0]

def bench(fn, step=100, warm_up=20, grad_to_none=None):
    # triton.testing.do_bench have bug if there are some comms in kernels
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(warm_up):
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
    start_event.record()
    for i in range(step):
        fn()
    end_event.record()
    torch.cuda.synchronize()  # 等待CUDA操作完成
    t1 = start_event.elapsed_time(end_event)
    return t1 / step

def all_gather(x, mode=None) -> torch.Tensor:
    world_size = dist.get_world_size()
    group = NSAHelper.process_group
    mode = NSAHelper.cp_mode if mode is None else mode
    y = torch.empty(world_size * x.size(0), *x.shape[1:], device=x.device, dtype=x.dtype)

    if mode == 1:
        dist.all_gather_into_tensor(y, x, group=group)
    else:
        n = y.size(0) // 2
        m = x.size(0) // 2
        dist.all_gather_into_tensor(y[:n], x[:m], group=group)
        # handle2 = dist.all_gather_into_tensor(y[n:], x[m:], group=group, async_op=True)
        dist.all_gather([t for t in y[n:].chunk(world_size, 0)][::-1], x[m:], group=group)
    return y

def broadcast(x):
    dist.broadcast(x, src=0)

def chunk_slice(x, dim=0):
    if x is None:
        return None
    cp_mode = NSAHelper.cp_mode
    if dim != 0:
        x = x.transpose(0, dim)
    if cp_mode == 1:
        y = x[NSAHelper.cp_bos1: NSAHelper.cp_eos1]
    else:
        y1 = x[NSAHelper.cp_bos1: NSAHelper.cp_eos1]
        y2 = x[NSAHelper.cp_bos2: NSAHelper.cp_eos2]
        y = torch.cat([y1, y2], 0)
    if dim != 0:
        y = y.transpose(0, dim).contiguous()
    return y

def test_all_gather(args):
    from flash_nsa.ops.reorder import reorder
    from flash_nsa.ops.torch_ref import torch_reorder
    rank = args.rank
    world_size = args.world_size
    chunk_q, chunk_k, chunk_v, chunk_do = generate_qkvdo(args, cmp_kv=False)

    k1 = torch_reorder(all_gather(chunk_k, mode=1), world_size=world_size, shuffle_mode="unshuffle")
    k2 = all_gather(chunk_k, mode=2)
    k3 = reorder(all_gather(chunk_k, mode=1), world_size=world_size, shuffle_mode="unshuffle", inplace=True)

    diff = []
    diff += compare(k1, k2, f"rank:{rank}, ag + torch reorder vs ag mode 2:")
    diff += compare(k1, k3, f"rank:{rank}, ag + torch reorder vs ag + triton reorder:")
    index = ["ag + torch reorder vs ag mode 2", "ag + torch reorder vs ag + triton reorder"]

    t1 = bench(lambda: torch_reorder(all_gather(chunk_k, mode=1), world_size=world_size, shuffle_mode="unshuffle"))
    t2 = bench(lambda: all_gather(chunk_k, mode=2))
    t3 = bench(lambda: reorder(all_gather(chunk_k, mode=1), world_size=world_size, shuffle_mode="unshuffle", inplace=True))
    t4 = bench(lambda: all_gather(chunk_k, mode=1))
    if rank == 0:
        print(chunk_k.shape, k1.shape)
        print(t1, "\n", t2, "\n", t3, "\n", t4)
    return diff ,index

def test_cmp_attn(args):
    from flash_nsa.ops.compress_attn import cmp_attn_fwd, cmp_attn_bwd
    rank = args.rank
    dkdv_repeat = False

    chunk_q, k, v, chunk_do = generate_qkvdo(args, cmp_kv=True)
    q, do = gather_tensors(chunk_q, chunk_do)
    NSAHelper.disable_cp()    
    out, lse = cmp_attn_fwd(q, k, v)
    dq, ref_dk, ref_dv = cmp_attn_bwd(q, k, v, out, lse, do, dkdv_repeat=dkdv_repeat)
    NSAHelper.enable_cp()   

    ref_out = chunk_slice(out)
    ref_lse = chunk_slice(lse, 1)
    ref_dq = chunk_slice(dq)

    out, lse = cmp_attn_fwd(chunk_q, k, v)
    dq, dk, dv = cmp_attn_bwd(chunk_q, k, v, out, lse, chunk_do, dkdv_repeat=dkdv_repeat) 
    dist.all_reduce(dk)
    dist.all_reduce(dv)

    diff = []
    diff += compare(ref_out, out, f"rank:{rank}, out:")
    diff += compare(ref_lse, lse, f"rank:{rank}, lse:")
    diff += compare(ref_dq, dq, f"rank:{rank}, dq:")
    diff += compare(ref_dk, dk, f"rank:{rank}, dk:")
    diff += compare(ref_dv, dv, f"rank:{rank}, dv:")
    index = ["out", "lse", "dq", "dk", "dv"]

    if rank == 3:
        NSAHelper.disable_cp()  
        out, lse = cmp_attn_fwd(q, k, v)  
        print(triton.testing.do_bench(lambda: cmp_attn_bwd(q, k, v, out, lse, do, dkdv_repeat=dkdv_repeat)))
        NSAHelper.enable_cp()   
        out, lse = cmp_attn_fwd(chunk_q, k, v)
        print(triton.testing.do_bench(lambda: cmp_attn_bwd(chunk_q, k, v, out, lse, do, dkdv_repeat=dkdv_repeat)))

    return diff, index

    return


def test_topk(args):
    from flash_nsa.ops.compress_attn import cmp_attn_fwd
    from flash_nsa.ops.topk import slc_topk_indices
    rank = args.rank
    return_slc_prob = True
    maybe_efficient_version = True
    fp32 = True

    chunk_q, k, v, chunk_do = generate_qkvdo(args, cmp_kv=True)
    q = gather_tensors(chunk_q)

    NSAHelper.disable_cp()    
    _, lse = cmp_attn_fwd(q, k, v)
    topk, p = slc_topk_indices(q, k, lse, return_slc_prob=return_slc_prob, maybe_efficient_version=maybe_efficient_version, fp32=fp32)
    NSAHelper.enable_cp()   

    ref_topk = chunk_slice(topk.float(), 1)
    ref_p = chunk_slice(p, 1)
    lse = chunk_slice(lse, 1)

    topk, p = slc_topk_indices(chunk_q, k, lse, return_slc_prob=return_slc_prob, maybe_efficient_version=maybe_efficient_version, fp32=fp32)
    topk = topk.float()

    diff = []
    diff += compare(ref_topk, topk, f"rank:{rank}, topk :")
    diff += compare(ref_p, p, f"rank:{rank}, p:")
    index = ["topk", "p"]
    return diff, index


def test_slc_attn(args):
    from flash_nsa.ops.compress_attn import cmp_attn_fwd
    from flash_nsa.ops.topk import slc_topk_indices
    from flash_nsa.ops.select_attn import slc_attn_fwd, slc_attn_bwd, get_bind_from_find
    rank = args.rank
    dkdv_repeat = False

    chunk_q, chunk_k, chunk_v, chunk_do = generate_qkvdo(args, cmp_kv=False)
    q, k, v, do = gather_tensors(chunk_q, chunk_k, chunk_v, chunk_do)
    cmp_k, cmp_v = k[:NSAHelper.y_cu_seqlens[-1]], v[:NSAHelper.y_cu_seqlens[-1]]

    NSAHelper.disable_cp()    
    _, cmp_lse = cmp_attn_fwd(q, cmp_k, cmp_v)
    topk, p = slc_topk_indices(q, cmp_k, cmp_lse, return_slc_prob=False, maybe_efficient_version=True)
    out, lse = slc_attn_fwd(q, k, v, topk)
    dq, ref_dk, ref_dv = slc_attn_bwd(q, k, v, topk, out, lse, do, dkdv_repeat=dkdv_repeat)
    NSAHelper.enable_cp()   

    ref_out = chunk_slice(out)
    ref_lse = chunk_slice(lse)
    ref_dq = chunk_slice(dq)
    ref_topk = chunk_slice(topk, 1)

    out, lse = slc_attn_fwd(chunk_q, k, v, ref_topk)
    dq = torch.zeros_like(chunk_q)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    dq, dk, dv = slc_attn_bwd(chunk_q, k, v, ref_topk, out, lse, chunk_do, dq, dk, dv, dkdv_repeat=dkdv_repeat) 
    dk = dk.to(DTYPE)
    dv = dv.to(DTYPE)
    dist.all_reduce(dk)
    dist.all_reduce(dv)

    diff = []
    diff += compare(ref_out, out, f"rank:{rank}, out:")
    diff += compare(ref_lse, lse, f"rank:{rank}, lse:")
    diff += compare(ref_dq, dq, f"rank:{rank}, dq:")
    diff += compare(ref_dk, dk, f"rank:{rank}, dk:")
    diff += compare(ref_dv, dv, f"rank:{rank}, dv:")
    index = ["out", "lse", "dq", "dk", "dv"]

    if rank == 0:
        NSAHelper.disable_cp()  
        out, lse = slc_attn_fwd(q, k, v, topk)
        print(triton.testing.do_bench(lambda: slc_attn_bwd(q, k, v, topk, out, lse, do, dkdv_repeat=dkdv_repeat)))
        NSAHelper.enable_cp()   
        out, lse = slc_attn_fwd(chunk_q, k, v, ref_topk)
        print(triton.testing.do_bench(lambda: slc_attn_bwd(chunk_q, k, v, ref_topk, out, lse, chunk_do, dq, dk, dv, dkdv_repeat=dkdv_repeat) ))
    return diff, index


def test_swa_attn(args):
    from flash_nsa.ops.sliding_window_attention import swa_varlen_func, two_stage_swa_fwd, two_stage_swa_bwd
    from flash_attn_interface import flash_attn_varlen_func

    rank = args.rank
    chunk_q, chunk_k, chunk_v, chunk_do = generate_qkvdo(args, cmp_kv=False)
    q, k, v, do = gather_tensors(chunk_q, chunk_k, chunk_v, chunk_do)

    out = swa_varlen_func(q, k, v)
    out.backward(do)
    assert q.grad is not None

    ref_out = chunk_slice(out)
    ref_dq = chunk_slice(q.grad)
    ref_dk = k.grad
    ref_dv = v.grad

    out = torch.empty_like(ref_out)
    dq = torch.zeros_like(chunk_q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    lse1 = two_stage_swa_fwd(chunk_q, chunk_k, chunk_v, out, stage=1)
    lse2 = two_stage_swa_fwd(chunk_q, k, v, out, stage=2)
    two_stage_swa_bwd(chunk_q, chunk_k, chunk_v, out, lse1, chunk_do, dq, dk, dv, stage=1)
    two_stage_swa_bwd(chunk_q, k, v, out, lse2, chunk_do, dq, dk, dv, stage=2)
    dist.all_reduce(dk)
    dist.all_reduce(dv)

    diff = []
    diff += compare(ref_out, out, f"rank:{rank}, out:")
    diff += compare(ref_dq, dq, f"rank:{rank}, dq:")
    diff += compare(ref_dk, dk, f"rank:{rank}, dk:")
    diff += compare(ref_dv, dv, f"rank:{rank}, dv:")
    index = ["out", "dq", "dk", "dv"]
    if rank == 0:
        print(triton.testing.do_bench(lambda: swa_varlen_func(q, k, v)))
        total_out = swa_varlen_func(q, k, v)
        print(triton.testing.do_bench(lambda: total_out.backward(do, retain_graph=True), grad_to_none=[q, k, v]))
        print(triton.testing.do_bench(lambda: two_stage_swa_fwd(chunk_q, chunk_k, chunk_v, out, stage=1)))
        print(triton.testing.do_bench(lambda: two_stage_swa_fwd(chunk_q, k, v, out, stage=2)))
        print(triton.testing.do_bench(lambda: two_stage_swa_bwd(chunk_q, chunk_k, chunk_v, out, lse1, chunk_do, dq, dk, dv, stage=1)))
        print(triton.testing.do_bench(lambda: two_stage_swa_bwd(chunk_q, k, v, out, lse2, chunk_do, dq, dk, dv, stage=2)))

    return diff, index

    if rank == 0:
        # print(cu_seqlens)
        # print(NSAHelper.cp_cu_seqlens)
        # print(ref_dk[:512, 0])
        # print(dk[:512, 0])
        print(rng_state1, rng_state2)
        # print(triton.testing.do_bench(lambda: compress_attn(total_q, k, v, kernel_size, stride)))
        print(triton.testing.do_bench(lambda: swa_fwd(q, k, v, out, 512, stage=1)))
        print(triton.testing.do_bench(lambda: swa_fwd(q, total_k, total_v, out, 512, stage=2)))
        print(triton.testing.do_bench(lambda: swa_bwd(q, k, v, out, lse1, rng_state1, dout, dq, dk, dv, stage=1)))
        print(triton.testing.do_bench(lambda: swa_bwd(q, total_k, total_v, out, lse2, rng_state2, dout, dq, dk, dv, stage=2)))
        # print(triton.testing.do_bench(lambda: total_q.float()))
        # print(triton.testing.do_bench(lambda: total_q.float().bfloat16()))


def test_nsa(args):

    from flash_nsa.modules.hf_level_nsa import HFNSACore, NSAConfig
    from flash_nsa.modules.megatron_level_nsa import NSACore as MegatronNSACore


    rank = args.rank
    config = NSAConfig()
    config.use_overlap_swa = args.use_overlap_swa
    config.qk_head_dim = args.qk_head_dim
    config.v_head_dim = args.v_head_dim
    config.num_heads = args.num_heads
    config.num_kv_heads = args.num_kv_heads
    config.cp_mode = args.cp_mode
    # config.recompute_cmp_kv = True
    # config.recompute_cmp_o = True
    # config.recompute_slc_o = True
    # config.recompute_swa_o = True

    test_pp = False

    cu_seqlens = NSAHelper.x_cu_seqlens
    NSAHelper.enable_bench_mode()
    # if use "swiglu" or "linear" or "sigmoid-mul", the max abs value is very huge, so the diff is more obvious
    config.cmp_k_method = "mean"
    config.cmp_v_method = "mean"

    chunk_q, chunk_k, chunk_v, chunk_do = generate_qkvdo(args, cmp_kv=False)
    chunk_combine_weight = torch.randn(*chunk_q.shape[:2], 3, device=chunk_q.device, dtype=DTYPE).requires_grad_(True)
    q, k, v, do, combine_weight  = gather_tensors(chunk_q, chunk_k, chunk_v, chunk_do, chunk_combine_weight)

    hf_nsa = HFNSACore(config).to(q)
    megatron_nsa = MegatronNSACore(config).to(q)

    cmp_k_weight = megatron_nsa.cmp_k_weight 
    if cmp_k_weight is not None:
        broadcast(cmp_k_weight.data)
        dist.barrier()
        hf_nsa.compress_k.fc.weight.data.copy_(cmp_k_weight.data)
        cmp_k_weight.main_grad = torch.zeros_like(cmp_k_weight, dtype=torch.float32)

    cmp_v_weight = megatron_nsa.cmp_v_weight 
    if cmp_v_weight is not None:
        broadcast(cmp_v_weight.data)
        dist.barrier()
        hf_nsa.compress_v.fc.weight.data.copy_(cmp_v_weight.data)
        cmp_v_weight.main_grad = torch.zeros_like(cmp_v_weight, dtype=torch.float32)

    NSAHelper.disable_cp()    
    out = hf_nsa(q, k, v, combine_weight, cu_seqlens)
    out.backward(do, retain_graph=True)
    NSAHelper.enable_cp()   

    ref_out = chunk_slice(out)
    ref_dq = chunk_slice(q.grad)
    ref_dk = chunk_slice(k.grad)
    ref_dv = chunk_slice(v.grad)
    ref_dcombine_weight = chunk_slice(combine_weight.grad)

    if test_pp:
        NSAHelper.pp_size = 8
    out = megatron_nsa(chunk_q, chunk_k, chunk_v, chunk_combine_weight, cu_seqlens)
    # test pp
    if test_pp:
        NSAHelper.init_cu_seqlens(torch.tensor([0, 20000], device=torch.cuda.current_device(), dtype=torch.int32), cp_mode=args.cp_mode)
    out.backward(chunk_do)

    diff = []
    diff += compare(ref_out, out, f"rank:{rank}, out:")
    diff += compare(ref_dq, chunk_q.grad, f"rank:{rank}, dq:")
    diff += compare(ref_dk, chunk_k.grad, f"rank:{rank}, dk:")
    diff += compare(ref_dv, chunk_v.grad, f"rank:{rank}, dv:")
    diff += compare(ref_dcombine_weight, chunk_combine_weight.grad, f"rank:{rank}, dcombine_weight:")
    index = ["out", "dq", "dk", "dv", "dcombine_weight"]
    if cmp_k_weight is not None:
        dist.all_reduce(cmp_k_weight.main_grad)
        diff += compare(hf_nsa.compress_k.fc.weight.grad, cmp_k_weight.main_grad, f"rank:{rank}, cmp_k_weight_grad:")
        index += ["cmp_k_weight_grad"]
    if cmp_v_weight is not None:
        dist.all_reduce(cmp_v_weight.main_grad)
        # print(cmp_v_weight.main_grad.abs().max())
        diff += compare(hf_nsa.compress_v.fc.weight.grad, cmp_v_weight.main_grad, f"rank:{rank}, cmp_v_weight_grad:")
        index += ["cmp_v_weight_grad"]

    if test_pp:
        NSAHelper.init_cu_seqlens(cu_seqlens)
        return diff, index

    NSAHelper.disable_cp()    
    out = hf_nsa(q, k, v, combine_weight, cu_seqlens)
    t1 = bench(lambda: hf_nsa(q, k, v, combine_weight, cu_seqlens))
    t2 = bench(lambda: out.backward(do, retain_graph=True), grad_to_none=[q, k, v, combine_weight])
    NSAHelper.enable_cp() 
    if rank == 0:
        print(t1,"\n",t2,"\n")  

    # print(f"rank: {rank}, {NSAHelper.cp_batch_idx}, {NSAHelper.cp_cu_seqlens}, {NSAHelper.cp_offset}")
    # i = 0
    # while True:
    #     megatron_nsa(chunk_q, chunk_k, chunk_v, chunk_combine_weight, cu_seqlens)
    #     i += 1
    #     if rank == 0:
    #         print(f"step: {i}")

    for i in range(10):
        kv_head_stride = 2 ** i
        NSAHelper.kv_head_stride = kv_head_stride
        if kv_head_stride > args.num_kv_heads:
            break
        t3 = bench(lambda: megatron_nsa(chunk_q, chunk_k, chunk_v, chunk_combine_weight, cu_seqlens))
        out = megatron_nsa(chunk_q, chunk_k, chunk_v, chunk_combine_weight, cu_seqlens)
        t4 = bench(lambda: out.backward(chunk_do, retain_graph=True), grad_to_none=[chunk_q, chunk_k, chunk_v, chunk_combine_weight])

        if rank == 0:
            gpus = args.world_size
            print(f"kv_head_stride: {kv_head_stride}")
            print(t3,"\n",t4,"\n")

            print(f"forward, non-cp: {t1:.2f} / {gpus} = {(t1/gpus):.2f} cp_mode: {NSAHelper.cp_mode}, kv_head_stride:{kv_head_stride}, cp: {t3:.2f}, rate: {((t1/gpus/t3) * 100):.2f} %")
            print(f"backward, non-cp: {t2:.2f} / {gpus} = {(t2/gpus):.2f} cp_mode: {NSAHelper.cp_mode}, kv_head_stride:{kv_head_stride}, cp: {t4:.2f}, rate: {((t2/gpus/t4) * 100):.2f} %")
            print(f"forward + backward, non-cp: {(t1+t2):.2f} / {gpus} = {((t1+t2)/gpus):.2f} cp_mode: {NSAHelper.cp_mode}, kv_head_stride:{kv_head_stride}, cp: {(t3+t4):.2f}, rate: {(((t1+t2)/gpus/(t3+t4)) * 100):.2f} %")

    return diff, index



def test_1M(args):
    # It can't confirm the correctness, because the max context of non-cp nsa is 256k.
    from flash_nsa.modules.megatron_level_nsa import NSACore, NSAConfig

    rank = args.rank
    config = NSAConfig()
    config.use_overlap_swa = args.use_overlap_swa
    config.qk_head_dim = args.qk_head_dim
    config.v_head_dim = args.v_head_dim
    config.num_heads = args.num_heads
    config.num_kv_heads = args.num_kv_heads
    config.cp_mode = args.cp_mode
    config.kv_head_stride = 1
    config.recompute_cmp_kv = True
    config.recompute_cmp_o = True
    config.recompute_slc_o = True
    config.recompute_swa_o = True

    cu_seqlens = NSAHelper.x_cu_seqlens
    config.cmp_k_method = "mean"
    config.cmp_v_method = "mean"

    chunk_q, chunk_k, chunk_v, chunk_do = generate_qkvdo(args, cmp_kv=False)
    chunk_combine_weight = torch.randn(*chunk_q.shape[:2], 3, device=chunk_q.device, dtype=DTYPE).requires_grad_(True)

    megatron_nsa = NSACore(config, process_group=args.group).to(chunk_q)

    cmp_k_weight = megatron_nsa.cmp_k_weight 
    if cmp_k_weight is not None:
        broadcast(cmp_k_weight.data)
        dist.barrier()
        cmp_k_weight.main_grad = torch.zeros_like(cmp_k_weight, dtype=torch.float32)

    cmp_v_weight = megatron_nsa.cmp_v_weight 
    if cmp_v_weight is not None:
        broadcast(cmp_v_weight.data)
        dist.barrier()
        cmp_v_weight.main_grad = torch.zeros_like(cmp_v_weight, dtype=torch.float32)
    print(f"rank: {rank}, init_memory: {get_memory()} G")
    time.sleep(2)

    out = megatron_nsa(chunk_q, chunk_k, chunk_v, chunk_combine_weight, cu_seqlens)
    print(f"rank: {rank}, forward_memory: {get_memory()} G")
    time.sleep(2)
    out.backward(chunk_do)
    print(f"rank: {rank}, backward_memory: {get_memory()} G")
    time.sleep(2)
    t1 = bench(lambda: megatron_nsa(chunk_q, chunk_k, chunk_v, chunk_combine_weight, cu_seqlens), warm_up=5, step=20)
    out = megatron_nsa(chunk_q, chunk_k, chunk_v, chunk_combine_weight, cu_seqlens)
    t2 = bench(lambda: out.backward(chunk_do, retain_graph=True), grad_to_none=[chunk_q, chunk_k, chunk_v, chunk_combine_weight], warm_up=5, step=20)
    if rank == 0:
        print(t1,"\n",t2,"\n")

    '''
    rank: 2, init_memory: 4.3 G
    rank: 6, init_memory: 4.3 G
    rank: 1, init_memory: 4.3 G
    rank: 7, init_memory: 4.3 G
    rank: 5, init_memory: 4.3 G
    rank: 3, init_memory: 4.3 G
    rank: 0, init_memory: 4.3 G
    rank: 4, init_memory: 4.3 G
    rank: 4, forward_memory: 6.33 G
    rank: 5, forward_memory: 6.33 G
    rank: 6, forward_memory: 6.33 G
    rank: 7, forward_memory: 6.33 G
    rank: 1, forward_memory: 6.33 G
    rank: 2, forward_memory: 6.33 G
    rank: 3, forward_memory: 6.33 G
    rank: 0, forward_memory: 6.33 G
    rank: 4, backward_memory: 8.59 G
    rank: 7, backward_memory: 8.59 G
    rank: 5, backward_memory: 8.59 G
    rank: 0, backward_memory: 8.59 G
    rank: 6, backward_memory: 8.59 G
    rank: 2, backward_memory: 8.59 G
    rank: 3, backward_memory: 8.59 G
    rank: 1, backward_memory: 8.59 G

    575.5625 
    1509.2296875 
    '''
    return None

def print_diff_infos(diff, index):
    rank = NSAHelper.rank
    world_size = NSAHelper.world_size
    np.save(f"./examples/print_data/max_diff_rank{rank}.npy", np.array(diff[::3], dtype=np.float32))
    np.save(f"./examples/print_data/mean_diff_rank{rank}.npy", np.array(diff[1::3], dtype=np.float32))
    np.save(f"./examples/print_data/max_value_rank{rank}.npy", np.array(diff[2::3], dtype=np.float32))
    dist.barrier()
    if rank == 0:
        print("\n cu_seqlens:")
        print(NSAHelper.x_cu_seqlens)
        max_diff = []
        mean_diff = []
        max_value = []
        for r in range(NSAHelper.world_size):
            max_diff.append(np.load(f"./examples/print_data/max_diff_rank{r}.npy"))
            mean_diff.append(np.load(f"./examples/print_data/mean_diff_rank{r}.npy"))
            max_value.append(np.load(f"./examples/print_data/max_value_rank{r}.npy"))
        max_diff = np.stack(max_diff, 0)
        mean_diff = np.stack(mean_diff, 0)
        max_value = np.stack(max_value, 0)
        df1 = pd.DataFrame(max_diff, columns=index, index=[f"rank{r}" for r in range(world_size)])
        df2 = pd.DataFrame(mean_diff, columns=index, index=[f"rank{r}" for r in range(world_size)])
        df3 = pd.DataFrame(max_value, columns=index, index=[f"rank{r}" for r in range(world_size)])
        print("\n max diff:")
        print(df1)
        print("\n mean diff:")
        print(df2)
        print("\n abs max value:")
        print(df3)

def main():
    rank, world_size, process_group = init_distributed()
    args = get_args()
    args.rank = rank
    args.world_size = world_size
    args.group = process_group
    if args.func == "1M":
        args.seqlen = 1024 * 1024
        args.mean = args.seqlen
        args.std = 0


    NSAHelper.set_context_parallel(process_group=process_group, use_overlap_swa=args.use_overlap_swa)
    cu_seqlens = generate_cu_seqlens(end=args.seqlen, mean=args.mean, var=args.std)
    NSAHelper.init_cu_seqlens(cu_seqlens, cp_mode=args.cp_mode)
    print_rank0("*"*100)
    print_rank0(f"world_size: {world_size}")
    print_rank0('\n'*2)

    func_map = {
                "nsa":test_nsa, 
                "ag":test_all_gather, 
                "topk":test_topk,
                "swa":test_swa_attn,
                "cmp":test_cmp_attn,
                "slc":test_slc_attn,
                "1M":test_1M,
                }

    out = func_map[args.func](args)

    dist.barrier()
    if out is None:
        return
    diff, index = out
    print_diff_infos(diff, index)

    print_rank0('\n'*3)
    print_rank0("*"*100)

if __name__ == '__main__':
    main()
    # torchrun --nproc-per-node=4 test_cp.py 