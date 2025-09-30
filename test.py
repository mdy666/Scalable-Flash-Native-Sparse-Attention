import os
import warnings
import random
import argparse

import torch
import triton

from flash_nsa import NSAHelper

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
warnings.filterwarnings("ignore")

DTYPE = torch.bfloat16
DEVICE = "cuda"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', default='nsa', choices=["construct", "mean", "cmp", "topk", "specific-topk", "slc", "combine", "nsa"])

    parser.add_argument('--num-heads', default=64, type=int)
    parser.add_argument('--num-kv-heads', default=4, type=int)
    parser.add_argument('--qk-head-dim', default=128, type=int)
    parser.add_argument('--v-head-dim', default=128, type=int)

    parser.add_argument('--seqlen', default=8192*8, type=int)
    parser.add_argument('--mean', default=8192*8, type=int)
    parser.add_argument('--std', default=0, type=int)

    return parser.parse_args()

def compare(x, y, prefix=""):
    if x is None or y is None:
        return
    if any([x.dtype == torch.float32, y.dtype==torch.float32]):
        x,y = x.float(), y.float()
    diff = (x-y).abs()
    a = diff.max().item()
    b = diff.mean().item()
    c = torch.max(x.abs().max(), y.abs().max()).item()
    print(prefix + f"max_diff: {a}, mean_diff: {b}, absmax: {c}")


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
    x_cu_seqlens, y_cu_seqlens = NSAHelper.x_cu_seqlens, NSAHelper.y_cu_seqlens
    q_len = x_cu_seqlens[-1].item()
    kv_len = y_cu_seqlens[-1].item() if cmp_kv else q_len
    q = torch.randn(q_len, QH, D, device=DEVICE, dtype=DTYPE).requires_grad_(True)
    k = torch.randn(kv_len, KH, D, device=DEVICE, dtype=DTYPE).requires_grad_(True)
    v = torch.randn(kv_len, KH, VD, device=DEVICE, dtype=DTYPE).requires_grad_(True)
    do = torch.randn(q_len, QH, VD, device=DEVICE, dtype=DTYPE)
    return q, k, v, do

def get_ref_grad(*tensors):
    grads = []
    for t in tensors:
        grads.append(t.grad)
        t.grad = None
    return grads if len(grads) > 1 else grads[0]

def test_construct_blocks(args):
    from flash_nsa.ops.compress_kv import construct_block
    from flash_nsa.ops.torch_ref import torch_construct_block

    x_cu_seqlens = NSAHelper.x_cu_seqlens
    y_cu_seqlens = NSAHelper.y_cu_seqlens
    b = len(x_cu_seqlens) - 1
    q, k, v, do = generate_qkvdo(args, cmp_kv=False)

    def func():
        y2_list = []
        for idx in range(b):
            y2 = torch_construct_block(k[x_cu_seqlens[idx]:x_cu_seqlens[idx+1]], NSAHelper.kernel_size, NSAHelper.stride)
            y2_list.append(y2)
        y2 = torch.cat(y2_list, 0)
        return y2
    ref_y = func()
    dy = torch.randn_like(ref_y).contiguous()
    ref_y.backward(dy, retain_graph=True)
    ref_grad = get_ref_grad(k)

    y = construct_block(k)
    y.backward(dy, retain_graph=True)
    compare(y, ref_y, "out, ")
    compare(ref_grad, k.grad, "grad_input, ")

    print("torch fwd: ", triton.testing.do_bench(lambda: func()))
    print("torch bwd: ", triton.testing.do_bench(lambda: ref_y.backward(dy, retain_graph=True), grad_to_none=[k]))
    print("triton fwd: ", triton.testing.do_bench(lambda: construct_block(k)))
    print("triton bwd: ", triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), grad_to_none=[k]))

def test_mean_pooling(args):
    from flash_nsa.ops.compress_kv import construct_block, mean_pooling
    from flash_nsa.ops.torch_ref import torch_construct_block

    x_cu_seqlens = NSAHelper.x_cu_seqlens
    y_cu_seqlens = NSAHelper.y_cu_seqlens
    b = len(x_cu_seqlens) - 1
    q, k, v, do = generate_qkvdo(args, cmp_kv=False)


    ref_y = construct_block(k).mean(-2)
    dy = torch.randn_like(ref_y)
    ref_y.backward(dy, retain_graph=True)
    ref_grad = get_ref_grad(k)

    y = mean_pooling(k)
    y.backward(dy, retain_graph=True)

    compare(y, ref_y, "out, ")
    compare(ref_grad, k.grad, "grad_input, ")

    print("construct + mean : ", triton.testing.do_bench(lambda: construct_block(k).mean(-2)))
    print("construct + mean: ", triton.testing.do_bench(lambda: ref_y.backward(dy, retain_graph=True), grad_to_none=[k]))
    print("mean pooling : ", triton.testing.do_bench(lambda: mean_pooling(k)))
    print("mean pooling : ", triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), grad_to_none=[k]))

def test_cmp_attn(args):
    from flash_nsa.ops.compress_attn import compress_attn
    from flash_nsa.ops.torch_ref import torch_cmp_attn

    x_cu_seqlens = NSAHelper.x_cu_seqlens
    y_cu_seqlens = NSAHelper.y_cu_seqlens
    b = len(x_cu_seqlens) - 1
    q, k, v, do = generate_qkvdo(args, cmp_kv=True)

    def func():
        y2_list = []
        for idx in range(b):
            y2 = torch_cmp_attn(q[x_cu_seqlens[idx]:x_cu_seqlens[idx+1]], 
                                k[y_cu_seqlens[idx]:y_cu_seqlens[idx+1]], 
                                v[y_cu_seqlens[idx]:y_cu_seqlens[idx+1]], 
                                NSAHelper.kernel_size, 
                                NSAHelper.stride)
            y2_list.append(y2)
        y2 = torch.cat(y2_list, 0)
        return y2
    ref_y = func()
    ref_y.backward(do, retain_graph=True)
    ref_dq, ref_dk, ref_dv = get_ref_grad(q, k, v)

    y, lse = compress_attn(q, k, v)
    y.backward(do, retain_graph=True)

    compare(ref_y, y, "out, ")
    compare(ref_dq, q.grad, "dq, ")
    compare(ref_dk, k.grad, "dk, ")
    compare(ref_dv, v.grad, "dv, ")
    print("torch fwd : ", triton.testing.do_bench(lambda: func()))
    print("torch bwd : ", triton.testing.do_bench(lambda: ref_y.backward(do, retain_graph=True), grad_to_none=[q, k, v]))
    print("triton fwd : ", triton.testing.do_bench(lambda: compress_attn(q, k, v)))
    print("triton bwd : ", triton.testing.do_bench(lambda: y.backward(do, retain_graph=True), grad_to_none=[q, k, v]))


def test_general_topk(args):
    from flash_nsa.ops.compress_attn import compress_attn
    from flash_nsa.ops.topk import slc_topk_indices
    from flash_nsa.ops.torch_ref import torch_topk

    x_cu_seqlens = NSAHelper.x_cu_seqlens
    y_cu_seqlens = NSAHelper.y_cu_seqlens
    b = len(x_cu_seqlens) - 1
    q, k, v, do = generate_qkvdo(args, cmp_kv=True)

    _, lse = compress_attn(q, k, v)
    fp16_topk, _ = slc_topk_indices(q, k, lse, ignore_index=999999, fp32=False)
    fp32_topk, _ = slc_topk_indices(q, k, lse, ignore_index=999999, fp32=True)

    def func():
        y2_list = []
        for idx in range(b):
            y2 = torch_topk(q[x_cu_seqlens[idx]:x_cu_seqlens[idx+1]].float(), 
                                k[y_cu_seqlens[idx]:y_cu_seqlens[idx+1]].float(), 
                                stride=NSAHelper.stride,
                                kernel_size=NSAHelper.kernel_size,
                                block_size=NSAHelper.block_size,
                                topn=NSAHelper.top_n,
                                num_init=NSAHelper.num_init_blocks,
                                num_local=NSAHelper.num_local_blocks,
                                ignore_idx=999999)
            y2_list.append(y2)
        y2 = torch.cat(y2_list, 1)
        return y2
    ref_topk = func()
    fp16_topk = fp16_topk.sort(-1)[0]
    fp32_topk = fp32_topk.sort(-1)[0]
    ref_topk = ref_topk.sort(-1)[0]

    # When the seqlen is huge, the memory is big. save fp16 attn_prob is more fast and use 1/2 kernel peak memory. 
    print("triton fp16 vs triton fp32 precision: ", (fp16_topk != fp32_topk).sum() / fp16_topk.numel())
    print("triton fp16 vs torch fp32 precision: ", (fp16_topk != ref_topk).sum() / fp16_topk.numel())
    print("triton fp32 vs torch fp32 precision: ", (fp32_topk != ref_topk).sum() / fp16_topk.numel())
    print("triton fp16 time: ", triton.testing.do_bench(lambda: slc_topk_indices(q, k, lse, align=True,ignore_index=999999, fp32=False)))
    print("triton fp32 time: ",triton.testing.do_bench(lambda: slc_topk_indices(q, k, lse, align=True,ignore_index=999999, fp32=True)))
    print("torch fp32 time: ",triton.testing.do_bench(lambda: func()))



def test_specific_topk(args):
    from flash_nsa.ops.compress_attn import compress_attn
    from flash_nsa.ops.topk import slc_topk_indices

    x_cu_seqlens = NSAHelper.x_cu_seqlens
    y_cu_seqlens = NSAHelper.y_cu_seqlens
    b = len(x_cu_seqlens) - 1
    q, k, v, do = generate_qkvdo(args, cmp_kv=True)
    _, lse = compress_attn(q, k, v)
    # ignore_index must bigger than max_num_slc_blocks rather than -1. It's useful when compute slc_attn_bwd
    # Setting this arg is only for verfiying the correctness with torch ops. deafult None
    fp16_topk1, _ = slc_topk_indices(q, k, lse, fp32=False)
    fp32_topk1, _ = slc_topk_indices(q, k, lse, fp32=True)
    # maybe_efficient_version=True
    fp16_topk2, _ = slc_topk_indices(q, k, lse, fp32=False, maybe_efficient_version=True)
    fp16_scale_topk2, _ = slc_topk_indices(q, k, lse, fp32=False, maybe_efficient_version=True, scale_slc_p=True)
    fp32_topk2, _ = slc_topk_indices(q, k, lse, fp32=True, maybe_efficient_version=True)

    # fp16_topk = fp16_topk.sort(-1)[0]
    # fp32_topk = fp32_topk.sort(-1)[0]
    # ref_topk = ref_topk.sort(-1)[0]

    # When the seqlen is huge, the memory is big. save fp16 attn_prob is more fast and use 1/2 kernel peak memory. 
    print("general fp16 vs general fp32 correctness: ", (fp16_topk1.sort(-1)[0] != fp32_topk1.sort(-1)[0]).sum() / fp16_topk1.numel())
    print("specific fp16 vs general fp32 correctness: ", (fp16_topk2.sort(-1)[0] != fp32_topk1.sort(-1)[0]).sum() / fp16_topk1.numel())
    print("specific scaled fp16 vs general fp32 correctness: ", (fp16_scale_topk2.sort(-1)[0] != fp32_topk1.sort(-1)[0]).sum() / fp16_topk1.numel())
    print("general fp32 vs general fp32 correctness: ", (fp32_topk2.sort(-1)[0] != fp32_topk1.sort(-1)[0]).sum() / fp16_topk1.numel())

    print("general fp16 time: ", triton.testing.do_bench(lambda: slc_topk_indices(q, k, lse, align=True,ignore_index=999999, fp32=False)))
    print("general fp32 time: ", triton.testing.do_bench(lambda: slc_topk_indices(q, k, lse, align=True,ignore_index=999999, fp32=True)))
    print("specific fp16 time: ", triton.testing.do_bench(lambda: slc_topk_indices(q, k, lse, fp32=False, maybe_efficient_version=True)))
    print("specific scaled fp16 time: ",triton.testing.do_bench(lambda: slc_topk_indices(q, k, lse, fp32=False, maybe_efficient_version=True, scale_slc_p=True)))
    print("specific fp32 time: ",triton.testing.do_bench(lambda: slc_topk_indices(q, k, lse, fp32=True, maybe_efficient_version=True)))


def test_slc_attn(args):
    from flash_nsa.ops.select_attn import select_attn
    from flash_nsa.ops.torch_ref import torch_slc_attn
    from flash_nsa.ops.compress_attn import compress_attn
    from flash_nsa.ops.topk import slc_topk_indices

    x_cu_seqlens = NSAHelper.x_cu_seqlens
    y_cu_seqlens = NSAHelper.y_cu_seqlens
    b = len(x_cu_seqlens) - 1
    q, k, v, do = generate_qkvdo(args, cmp_kv=False)

    with torch.no_grad():
        _, lse = compress_attn(q, k[:y_cu_seqlens[-1]], v[:y_cu_seqlens[-1]])
        topk,_ = slc_topk_indices(q, k[:y_cu_seqlens[-1]], lse, fp32=False, maybe_efficient_version=True, scale_slc_p=True)

    def func():
        y2_list = []
        for idx in range(b):
            y2 = torch_slc_attn(q[x_cu_seqlens[idx]:x_cu_seqlens[idx+1]], 
                                k[x_cu_seqlens[idx]:x_cu_seqlens[idx+1]], 
                                v[x_cu_seqlens[idx]:x_cu_seqlens[idx+1]], 
                                topk=topk[:,x_cu_seqlens[idx]:x_cu_seqlens[idx+1]],
                                block_size=NSAHelper.block_size,)
            y2_list.append(y2)
        y2 = torch.cat(y2_list, 0)
        return y2
    ref_y = func()
    ref_y.backward(do, retain_graph=True)
    ref_dq, ref_dk, ref_dv = get_ref_grad(q, k, v)

    y = select_attn(q, k, v, topk)
    y.backward(do, retain_graph=True)

    compare(ref_y, y, "out, ")
    compare(ref_dq, q.grad, "dq, ")
    compare(ref_dk, k.grad, "dk, ")
    compare(ref_dv, v.grad, "dv, ")
    print("torch fwd : ", triton.testing.do_bench(lambda: func()))
    print("torch bwd : ", triton.testing.do_bench(lambda: ref_y.backward(do, retain_graph=True), grad_to_none=[q, k, v]))
    print("triton fwd : ", triton.testing.do_bench(lambda: select_attn(q, k, v, topk)))
    print("triton bwd : ", triton.testing.do_bench(lambda: y.backward(do, retain_graph=True), grad_to_none=[q, k, v]))

def test_combine(args):
    from flash_nsa.ops.combine import fused_sigmoid_combine
    from flash_nsa.ops.torch_ref import torch_sigmoid_combine


    t, qh, d = args.seqlen, args.num_heads, args.qk_head_dim

    a = torch.randn(t, qh, d, device=DEVICE, dtype=DTYPE).requires_grad_(True)
    b = torch.randn_like(a).requires_grad_(True)
    c = torch.randn_like(a).requires_grad_(True)
    w = torch.randn(t, qh, 3, device=DEVICE, dtype=DTYPE).requires_grad_(True)
    dy = torch.randn_like(a)

    ref_y = torch_sigmoid_combine(a, b, c, w)
    ref_y.backward(dy, retain_graph=True)
    ref_da, ref_db, ref_dc, ref_dw = get_ref_grad(a, b, c, w)
    y = fused_sigmoid_combine(a, b, c, w)
    y.backward(dy, retain_graph=True)


    compare(y, ref_y, "o, ")
    compare(ref_da, a.grad, "dcmp_o, ")
    compare(ref_db, b.grad, "dslc_o, ")
    compare(ref_dc, c.grad, "dswa_o, ")
    compare(ref_dw, w.grad, "dcombine_weight, ")

    print("torch fwd : ",triton.testing.do_bench(lambda: torch_sigmoid_combine(a, b, c, w)))
    print("torch bwd : ",triton.testing.do_bench(lambda: ref_y.backward(dy, retain_graph=True), grad_to_none=[a, b, c, w]))
    print("triton fwd : ",triton.testing.do_bench(lambda: fused_sigmoid_combine(a, b, c, w)))
    print("triton bwd : ",triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), grad_to_none=[a, b, c, w]))

def test_nsa(args):
    from flash_nsa.config import NSAConfig
    from flash_nsa.modules.hf_level_nsa import HFNSACore, NSAHelper
    from flash_nsa.modules.megatron_level_nsa import NSACore as MegatronNSACore

    config = NSAConfig()
    config.qk_head_dim = args.qk_head_dim
    config.v_head_dim = args.v_head_dim
    config.num_heads = args.num_heads
    config.num_kv_heads = args.num_kv_heads

    x_cu_seqlens = NSAHelper.x_cu_seqlens
    q, k, v, do = generate_qkvdo(args, cmp_kv=False)

    weight = torch.randn(x_cu_seqlens[-1], args.num_heads, 3, device=q.device, dtype=DTYPE).requires_grad_(True)
    hf_nsa = HFNSACore(config).to(q)
    megatron_nsa = MegatronNSACore(config).to(q)

    ref_y = hf_nsa(q, k, v, weight, x_cu_seqlens)
    ref_y.backward(do, retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dw = get_ref_grad(q, k, v, weight)

    y = megatron_nsa(q, k, v, weight, x_cu_seqlens)

    y.backward(do, retain_graph=True)
    compare(ref_y, y, "out, ")
    compare(ref_dq, q.grad, "dq, ")
    compare(ref_dk, k.grad, "dk, ")
    compare(ref_dv, v.grad, "dv, ")
    compare(ref_dw, weight.grad, "dw, ")
    print("hf nsa fwd : ", triton.testing.do_bench(lambda: hf_nsa(q, k, v, weight, x_cu_seqlens)))
    print("hf nsa bwd : ", triton.testing.do_bench(lambda: ref_y.backward(do, retain_graph=True), grad_to_none=[q, k, v, weight]))
    print("megatron nsa fwd : ", triton.testing.do_bench(lambda: megatron_nsa(q, k, v, weight, x_cu_seqlens)))
    print("megatron nsa bwd : ", triton.testing.do_bench(lambda: y.backward(do, retain_graph=True), grad_to_none=[q, k, v, weight]))

def main():
    args = get_args()

    cu_seqlens = generate_cu_seqlens(end=args.seqlen, mean=args.mean, var=args.std)
    NSAHelper.init_cu_seqlens(cu_seqlens)
    print(f"cu_seqlens: {cu_seqlens}")

    func_map = {
                "construct": test_construct_blocks,
                "mean": test_mean_pooling,
                "cmp": test_cmp_attn,
                "topk": test_general_topk,
                "specific-topk": test_specific_topk,
                "slc": test_slc_attn,
                "combine":test_combine,
                "nsa": test_nsa,
                }

    func_map[args.func](args)

if __name__ == '__main__':
    main()