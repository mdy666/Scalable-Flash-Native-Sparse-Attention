import math
import torch
from typing import Callable, Union
import numpy as np

from ..ops.compress_kv import construct_block_fwd, construct_block_bwd, mean_pooling_fwd, mean_pooling_bwd
from ..ops.compress_attn import cmp_attn_fwd, cmp_attn_bwd
from ..ops.select_attn import slc_attn_fwd, slc_attn_bwd
from ..ops.topk import slc_topk_indices
from ..ops.act import swiglu_fwd, swiglu_bwd, sigmoid_mul_fwd, sigmoid_mul_bwd
from ..ops.combine import combine_fwd, combine_bwd
from ..ops.sliding_window_attention import swa_fwd, swa_bwd, two_stage_swa_fwd, two_stage_swa_bwd
from ..ops.gemm import gemm

from ..utils import NSAHelper, BwdNSAHelper
from ..config import NSAConfig
from ..comm import Comm


def weight_grad(input, grad_output, weight):
    input = input.view(-1, input.size(-1))
    grad_output = grad_output.view(-1, grad_output.size(-1))
    main_grad = getattr(weight, "main_grad", None)
    if main_grad is not None:
        try:
            import fused_weight_gradient_mlp_cuda
            wgrad = fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(input, grad_output, weight.main_grad)
        except:
            # You can use triton wgrad kernel. In my test, the performance is better than apex for [256 or 128, 8192] @ [8192, 4096]
            # For best performance, please use TransformerEngine kernel
            wgrad = gemm(grad_output, input, out=main_grad, acc=True, transpose_a=True, transpose_b=False, persistent=False)
        if hasattr(weight, 'grad_added_to_main_grad'):
            wgrad = torch.empty(
                weight.main_grad.shape,
                dtype=input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            weight.grad_added_to_main_grad = True
        else:
            wgrad = None
    else:
        wgrad = torch.matmul(grad_output.t(), input)
    return wgrad

def compress_kv_fwd(x, weight, version, helper=NSAHelper):
    block_x, hidden_state = None, None
    if version == "mean":
        out = mean_pooling_fwd(x, helper=helper)
    else:
        block_x = construct_block_fwd(x, helper=helper)
        hidden_state = torch.matmul(torch.flatten(block_x, start_dim=-2, end_dim=-1), weight.t())
        if version == "linear":
            out = hidden_state
        elif version == "sigmoid-mul":
            out = sigmoid_mul_fwd(hidden_state)
        elif version == "swiglu":
            out = swiglu_fwd(hidden_state)
    return block_x, hidden_state, out

def compress_kv_bwd(x, weight, block_x, hidden_state, dout, dx, version, helper=NSAHelper, async_dw=False):
    dw = None
    if version == "mean":
        mean_pooling_bwd(x, dout, dx, helper=helper)
        return dx, dw
    else:
        if version == "linear":
            dhidden_state = dout
        elif version == "sigmoid-mul":
            dhidden_state = sigmoid_mul_bwd(hidden_state, dout)
        elif version == "swiglu":
            dhidden_state = swiglu_bwd(hidden_state, dout)
        dblock_x = torch.matmul(dhidden_state, weight).view(*block_x.shape)
        dx = construct_block_bwd(x, dblock_x, dx, helper=helper)

        def func():
            dw = weight_grad(block_x.flatten(-2, -1), dhidden_state, weight)
            return dw

        if not async_dw:
            dw = func()
            return dx, dw
        else:
            return dx, func


class NSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, combine_weight, cmp_k_weight, cmp_v_weight, sm_scale=None):
        if sm_scale is None:
            sm_scale = q.size(-1) ** -0.5

        block_k, hidden_state_k, cmp_k = compress_kv_fwd(k, cmp_k_weight, NSAHelper.cmp_k_method)
        block_v, hidden_state_v, cmp_v = compress_kv_fwd(v, cmp_v_weight, NSAHelper.cmp_v_method)
        cmp_o, cmp_lse = cmp_attn_fwd(q, cmp_k, cmp_v, sm_scale=sm_scale)
        topk, _ = slc_topk_indices(q, cmp_k, cmp_lse, sm_scale=sm_scale, maybe_efficient_version=True, scale_slc_p=True)
        slc_o, sls_lse = slc_attn_fwd(q, k, v, topk, sm_scale=sm_scale)
        swa_o, swa_lse = swa_fwd(q, k, v, sm_scale=sm_scale)
        combine_o = combine_fwd(cmp_o, slc_o, swa_o, combine_weight)

        input_tensors = [q, k, v, combine_weight, cmp_k_weight, cmp_v_weight]
        cmp_tensors = [block_k, hidden_state_k, cmp_k, block_v, hidden_state_v, cmp_v]
        cmp_attn_tensors = [cmp_o, cmp_lse]
        slc_attn_tensors = [slc_o, sls_lse]
        swa_attn_tensors = [swa_o, swa_lse]

        if NSAHelper.recompute_cmp_kv:
            NSAHelper.clear_tensor_data(*cmp_tensors)
            # cmp_tensors = [None] * 6

        if NSAHelper.recompute_cmp_o:
            NSAHelper.clear_tensor_data(*cmp_attn_tensors)
            # cmp_attn_tensors = [None] * 2

        if NSAHelper.recompute_slc_o:
            NSAHelper.clear_tensor_data(*slc_attn_tensors)
            # slc_attn_tensors = [None] * 2

        if NSAHelper.recompute_swa_o:
            NSAHelper.clear_tensor_data(*swa_attn_tensors)
            # swa_attn_tensors = [None] * 2

        ctx.save_for_backward(*input_tensors, *cmp_tensors, *cmp_attn_tensors, *slc_attn_tensors, *swa_attn_tensors, topk)
        ctx.sm_scale = sm_scale
        ctx.helper = NSAHelper.get_bwd_helper()
        return combine_o

    @staticmethod
    def backward(ctx, do):
        q, k, v, combine_weight, cmp_k_weight, cmp_v_weight, \
        block_k, hidden_state_k, cmp_k, block_v, hidden_state_v, cmp_v, \
        cmp_o, cmp_lse, slc_o, sls_lse, swa_o, swa_lse, topk = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        helper = ctx.helper

        if cmp_k.numel() == 0:
            block_k, hidden_state_k, cmp_k = compress_kv_fwd(k, cmp_k_weight, NSAHelper.cmp_k_method, helper)
            block_v, hidden_state_v, cmp_v = compress_kv_fwd(v, cmp_v_weight, NSAHelper.cmp_v_method, helper)

        if cmp_o.numel() == 0:
            cmp_o, cmp_lse = cmp_attn_fwd(q, cmp_k, cmp_v, sm_scale=sm_scale, helper=helper)

        if slc_o.numel() == 0:
            slc_o, sls_lse = slc_attn_fwd(q, k, v, topk, sm_scale=sm_scale, helper=helper)

        if swa_o.numel() == 0:
            swa_o, swa_lse = swa_fwd(q, k, v, sm_scale=sm_scale, helper=helper)

        dcmp_o, dslc_o, dswa_o, dcombine_weight = combine_bwd(cmp_o, slc_o, swa_o, combine_weight, do)
        _dq, _dk, _dv = swa_bwd(q, k, v, swa_o, swa_lse, dswa_o, sm_scale=sm_scale, helper=helper)
        # if you want higher precision gradients, you can use _dq.float() here
        dq, dk, dv = _dq, _dk.float(), _dv.float()
        NSAHelper.clear_tensor_data(_dk, _dv)

        slc_attn_bwd(q, k, v, topk, slc_o, sls_lse, dslc_o, dq, dk, dv, sm_scale=sm_scale, helper=helper)
        _, dcmp_k, dcmp_v = cmp_attn_bwd(q, cmp_k, cmp_v, cmp_o, cmp_lse, dcmp_o, dq, dkdv_repeat=False, sm_scale=sm_scale, helper=helper)
        _, dcmp_k_weight = compress_kv_bwd(k, cmp_k_weight, block_k, hidden_state_k, dcmp_k, dk, version=NSAHelper.cmp_k_method, helper=helper, async_dw=False)
        _, dcmp_v_weight = compress_kv_bwd(v, cmp_v_weight, block_v, hidden_state_v, dcmp_v, dv, version=NSAHelper.cmp_v_method, helper=helper, async_dw=False)
        return dq, dk, dv, dcombine_weight, dcmp_k_weight, dcmp_v_weight, None


class CPNSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, chunk_q, chunk_k, chunk_v, combine_weight, cmp_k_weight, cmp_v_weight, sm_scale=None):
        t, qh, d = chunk_q.shape
        kh = chunk_k.size(1)
        vd = chunk_v.size(-1)
        G = qh // kh
        if sm_scale is None:
            sm_scale = d ** -0.5
        kv_head_stride = NSAHelper.kv_head_stride
        assert kh % kv_head_stride == 0
        if kv_head_stride > kh: kv_head_stride = kh

        world_size = NSAHelper.world_size
        kv = torch.empty(t * world_size * kv_head_stride * (d+vd), device=chunk_k.device, dtype=chunk_k.dtype)
        k = kv[:t * world_size * kv_head_stride * d].view(t * world_size, kv_head_stride, d)
        v = kv[t * world_size * kv_head_stride * d:].view(t * world_size, kv_head_stride, vd)
        buffer = {Comm.KEY: k, Comm.VALUE: v}
        comm = Comm(process_group=NSAHelper.process_group, buffer=buffer, cp_mode=NSAHelper.cp_mode)

        tensors = [chunk_q, chunk_k, chunk_v, combine_weight, cmp_k_weight, cmp_v_weight]
        combine_o = torch.empty(t, qh, vd, dtype=chunk_q.dtype, device=chunk_q.device)

        for start_kv_head_idx in range(0, kh, kv_head_stride):
            q = chunk_q[:, start_kv_head_idx * G: (start_kv_head_idx + kv_head_stride) * G]
            chunk_k_i = chunk_k[:, start_kv_head_idx: start_kv_head_idx + kv_head_stride]
            chunk_v_i = chunk_v[:, start_kv_head_idx: start_kv_head_idx + kv_head_stride]
            combine_o_i = combine_o[:, start_kv_head_idx * G: (start_kv_head_idx + kv_head_stride) * G]
            combine_weight_i = combine_weight[:, start_kv_head_idx * G: (start_kv_head_idx + kv_head_stride) * G]
            if start_kv_head_idx == 0:
                comm.all_gather(chunk_k_i, key=Comm.KEY)
                comm.all_gather(chunk_v_i, key=Comm.VALUE)

            swa_o = torch.empty_like(combine_o_i)
            assert swa_o.is_contiguous()
            swa_lse1 = two_stage_swa_fwd(q, chunk_k_i, chunk_v_i, sm_scale=sm_scale, out=swa_o, stage=1)

            comm.wait(key=Comm.KEY)
            block_k, hidden_state_k, cmp_k = compress_kv_fwd(k, cmp_k_weight, NSAHelper.cmp_k_method)
            comm.wait(key=Comm.VALUE)
            block_v, hidden_state_v, cmp_v = compress_kv_fwd(v, cmp_v_weight, NSAHelper.cmp_v_method)
            cmp_o, cmp_lse = cmp_attn_fwd(q, cmp_k, cmp_v, sm_scale=sm_scale)
            topk, _ = slc_topk_indices(q, cmp_k, cmp_lse, sm_scale=sm_scale, maybe_efficient_version=True, scale_slc_p=True)
            slc_o, sls_lse = slc_attn_fwd(q, k, v, topk, sm_scale=sm_scale)
            swa_lse2 = two_stage_swa_fwd(q, k, v, sm_scale=sm_scale, out=swa_o, stage=2)

            if start_kv_head_idx + kv_head_stride < kh:
                comm.all_gather(chunk_k[:, start_kv_head_idx + kv_head_stride: start_kv_head_idx + kv_head_stride * 2], key=Comm.KEY)
                comm.all_gather(chunk_v[:, start_kv_head_idx + kv_head_stride: start_kv_head_idx + kv_head_stride * 2], key=Comm.VALUE)

            combine_fwd(cmp_o, slc_o, swa_o, combine_weight_i, out=combine_o_i)

            cmp_tensors = [block_k, hidden_state_k, cmp_k, block_v, hidden_state_v, cmp_v]
            cmp_attn_tensors = [cmp_o, cmp_lse]
            slc_attn_tensors = [slc_o, sls_lse]
            swa_attn_tensors = [swa_o] + swa_lse1 + swa_lse2
            if NSAHelper.recompute_cmp_kv:
                NSAHelper.clear_tensor_data(*cmp_tensors)
                # cmp_tensors = [None] * 6

            if NSAHelper.recompute_cmp_o:
                NSAHelper.clear_tensor_data(*cmp_attn_tensors)
                # cmp_attn_tensors = [None] * 2

            if NSAHelper.recompute_slc_o:
                NSAHelper.clear_tensor_data(*slc_attn_tensors)
                # slc_attn_tensors = [None] * 2

            if NSAHelper.recompute_swa_o:
                NSAHelper.clear_tensor_data(*swa_attn_tensors)
                # swa_attn_tensors = [None] * 5

            tensors = tensors + cmp_tensors + cmp_attn_tensors + slc_attn_tensors + swa_attn_tensors + [topk]

        ctx.save_for_backward(*tensors)
        ctx.sm_scale = sm_scale
        ctx.helper = NSAHelper.get_bwd_helper()
        ctx.kv_head_stride = NSAHelper.kv_head_stride
        return combine_o

    @staticmethod
    def backward(ctx, do, *args):
        tensors = ctx.saved_tensors
        chunk_q, chunk_k, chunk_v, combine_weight, cmp_k_weight, cmp_v_weight = tensors[:6]
        act_tensors = tensors[6:]
        sm_scale = ctx.sm_scale
        helper:BwdNSAHelper = ctx.helper
        kv_head_stride = ctx.kv_head_stride

        t, qh, d = chunk_q.shape
        kh = chunk_k.size(1)
        vd = chunk_v.size(-1)
        G = qh // kh

        world_size = NSAHelper.world_size
        kv = torch.empty(t * world_size * kv_head_stride * (d+vd), device=chunk_k.device, dtype=chunk_k.dtype)
        k = kv[:t * world_size * kv_head_stride * d].view(t * world_size, kv_head_stride, d)
        v = kv[t * world_size * kv_head_stride * d:].view(t * world_size, kv_head_stride, vd)
        dkdv = torch.zeros(t * world_size * kv_head_stride * (d+vd), device=chunk_k.device, dtype=torch.float32)
        dk = dkdv[:t * world_size * kv_head_stride * d].view(t * world_size, kv_head_stride, d)
        dv = dkdv[t * world_size * kv_head_stride * d:].view(t * world_size, kv_head_stride, vd)
        buffer = {Comm.KEY: k, Comm.VALUE: v, Comm.GRAD_KEY: dk, Comm.GRAD_VALUE: dv}
        comm = Comm(process_group=NSAHelper.process_group, buffer=buffer, cp_mode=helper.cp_mode)

        chunk_dq = torch.zeros_like(chunk_q, dtype=chunk_q.dtype)
        chunk_dk = torch.empty_like(chunk_k, dtype=chunk_k.dtype)
        chunk_dv = torch.empty_like(chunk_v, dtype=chunk_v.dtype)
        # reduce sactter in fp32 , don't bring overhead compare with bf16
        chunk_dk_i = torch.empty(t, kv_head_stride, d, device=chunk_k.device, dtype=torch.float32)
        chunk_dv_i = torch.empty(t, kv_head_stride, vd, device=chunk_k.device, dtype=torch.float32)

        dcombine_weight = torch.empty_like(combine_weight, dtype=combine_weight.dtype)

        if cmp_k_weight is not None and getattr(cmp_k_weight, 'main_grad', None) is None:
            dcmp_k_weight = torch.zeros_like(cmp_k_weight)
        else:
            dcmp_k_weight = None

        if cmp_v_weight is not None and getattr(cmp_v_weight, 'main_grad', None) is None:
            dcmp_v_weight = torch.zeros_like(cmp_v_weight)
        else:
            dcmp_v_weight = None

        for start_kv_head_idx in range(0, kh, kv_head_stride):
            '''
            There are many methods for overlap in backward.
            When the memory of cmp_kv is less than chunk_kv,
            we can save cmp_kv(not contain block_kv) and cmp_o, 
            then use cmp_attn_bwd to overlap all-gather kv. 
            Please do it by yourself if you need.
            '''
            q = chunk_q[:, start_kv_head_idx * G: (start_kv_head_idx + kv_head_stride) * G]
            chunk_k_i = chunk_k[:, start_kv_head_idx: start_kv_head_idx + kv_head_stride]
            chunk_v_i = chunk_v[:, start_kv_head_idx: start_kv_head_idx + kv_head_stride]
            dq = chunk_dq[:, start_kv_head_idx * G: (start_kv_head_idx + kv_head_stride) * G]
            dcombine_o_i = do[:, start_kv_head_idx * G: (start_kv_head_idx + kv_head_stride) * G]
            combine_weight_i = combine_weight[:, start_kv_head_idx * G: (start_kv_head_idx + kv_head_stride) * G]
            dcombine_weight_i = dcombine_weight[:, start_kv_head_idx * G: (start_kv_head_idx + kv_head_stride) * G]
            block_k, hidden_state_k, cmp_k, block_v, hidden_state_v, cmp_v, cmp_o, cmp_lse, slc_o, slc_lse, \
            swa_o, swa_lse11, swa_lse12, swa_lse21, swa_lse22, topk = act_tensors[:16]
            swa_lse1 = [swa_lse11, swa_lse12]
            swa_lse2 = [swa_lse21, swa_lse22]
            act_tensors = act_tensors[16:]

            if start_kv_head_idx == 0:
                comm.all_gather(chunk_k_i, key=Comm.KEY)
                comm.all_gather(chunk_v_i, key=Comm.VALUE)

            recompute_swa_o = False
            if swa_o.numel() == 0:
                recompute_swa_o = True
                swa_o = torch.empty_like(dcombine_o_i)
                swa_lse1 = two_stage_swa_fwd(q, chunk_k_i, chunk_v_i, sm_scale=sm_scale, out=swa_o, stage=1, helper=helper)

            if cmp_k.numel() == 0:
                comm.wait(key=Comm.KEY)
                block_k, hidden_state_k, cmp_k = compress_kv_fwd(k, cmp_k_weight, NSAHelper.cmp_k_method, helper)
                comm.wait(key=Comm.VALUE)
                block_v, hidden_state_v, cmp_v = compress_kv_fwd(v, cmp_v_weight, NSAHelper.cmp_v_method, helper)

            if recompute_swa_o:
                comm.wait_all()
                swa_lse2 = two_stage_swa_fwd(q, k, v, sm_scale=sm_scale, out=swa_o, stage=2, helper=helper)

            if cmp_o.numel() == 0:
                comm.wait_all()
                cmp_o, cmp_lse = cmp_attn_fwd(q, cmp_k, cmp_v, sm_scale=sm_scale, helper=helper)

            if slc_o.numel() == 0:
                comm.wait_all()
                slc_o, slc_lse = slc_attn_fwd(q, k, v, topk, sm_scale=sm_scale, helper=helper)

            dcmp_o, dslc_o, dswa_o, _ = combine_bwd(cmp_o, slc_o, swa_o, combine_weight_i, dcombine_o_i, dw=dcombine_weight_i)

            two_stage_swa_bwd(q, chunk_k_i, chunk_v_i, swa_o, swa_lse1, dswa_o, dq=dq, dk=dk, dv=dv, sm_scale=sm_scale, stage=1, helper=helper)
            comm.wait_all()
            two_stage_swa_bwd(q, k, v, swa_o, swa_lse2, dswa_o, dq=dq, dk=dk, dv=dv, sm_scale=sm_scale, stage=2, helper=helper)

            _, _, _, dq_func1 = slc_attn_bwd(q, k, v, topk, slc_o, slc_lse, dslc_o, dq, dk, dv, async_dq=True, sm_scale=sm_scale, helper=helper)

            _, dcmp_k, dcmp_v, dq_func2 = cmp_attn_bwd(q, cmp_k, cmp_v, cmp_o, cmp_lse, dcmp_o, dq, async_dq=True, dkdv_repeat=False, sm_scale=sm_scale, helper=helper)

            _, dcmp_k_weight_i = compress_kv_bwd(k, cmp_k_weight, block_k, hidden_state_k, dcmp_k, dk, async_dw=True, version=NSAHelper.cmp_k_method, helper=helper)
            comm.reduce_scatter(chunk_dk_i, key=Comm.GRAD_KEY)
            _, dcmp_v_weight_i = compress_kv_bwd(v, cmp_v_weight, block_v, hidden_state_v, dcmp_v, dv, async_dw=True, version=NSAHelper.cmp_v_method, helper=helper)
            comm.reduce_scatter(chunk_dv_i, key=Comm.GRAD_VALUE)

            dq_func1()
            dq_func2()

            if start_kv_head_idx + kv_head_stride < kh:
                comm.all_gather(chunk_k[:, start_kv_head_idx + kv_head_stride: start_kv_head_idx + kv_head_stride * 2], key=Comm.KEY)
                comm.all_gather(chunk_v[:, start_kv_head_idx + kv_head_stride: start_kv_head_idx + kv_head_stride * 2], key=Comm.VALUE)

            if isinstance(dcmp_k_weight_i, Callable):
                dcmp_k_weight_i = dcmp_k_weight_i()
            if isinstance(dcmp_v_weight_i, Callable):
                dcmp_v_weight_i = dcmp_v_weight_i()

            if dcmp_k_weight is not None:
                dcmp_k_weight += dcmp_k_weight_i
            if dcmp_v_weight is not None:
                dcmp_v_weight += dcmp_v_weight_i

            comm.wait(key=Comm.GRAD_KEY)
            comm.wait(key=Comm.GRAD_VALUE)
            if kv_head_stride < kh:
                chunk_dk[:, start_kv_head_idx: start_kv_head_idx + kv_head_stride].data.copy_(chunk_dk_i)
                chunk_dv[:, start_kv_head_idx: start_kv_head_idx + kv_head_stride].data.copy_(chunk_dv_i)

            # if not NSAHelper.bench_mode:
            #     NSAHelper.clear_tensor_data(swa_o, dswa_o, *swa_lse1, *swa_lse2)
            #     NSAHelper.clear_tensor_data(slc_o, dslc_o, slc_lse)
            #     NSAHelper.clear_tensor_data(cmp_o, cmp_k, cmp_v, dcmp_o, cmp_lse)

            dkdv.zero_()

        if dcmp_k_weight is None:
            dcmp_k_weight = dcmp_k_weight_i
        if dcmp_v_weight is None:
            dcmp_v_weight = dcmp_v_weight_i

        return chunk_dq, chunk_dk, chunk_dv, dcombine_weight, dcmp_k_weight, dcmp_v_weight, None

def flash_nsa_varlen_func(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    combine_weight: torch.Tensor, 
    cmp_k_weight: torch.Tensor, 
    cmp_v_weight: torch.Tensor, 
    sm_scale: float = None
) -> torch.Tensor:

    out = NSAFunction.apply(
        q, 
        k, 
        v, 
        combine_weight, 
        cmp_k_weight, 
        cmp_v_weight, 
        sm_scale,
    )

    return out

def cp_flash_nsa_varlen_func(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    combine_weight: torch.Tensor, 
    cmp_k_weight: torch.Tensor, 
    cmp_v_weight: torch.Tensor, 
    sm_scale: float = None
):

    out = CPNSAFunction.apply(
        q, 
        k, 
        v, 
        combine_weight, 
        cmp_k_weight, 
        cmp_v_weight, 
        sm_scale,
    )

    return out

class NSACore(torch.nn.Module):
    """
    native sparse attention module for huggingface level users.
    You can change the compress method and combine method according to your design.
    """
    def __init__(
        self, 
        config: NSAConfig, 
        process_group=None,
        pp_size=1
    ):
        super().__init__()
        self.config = config
        self.qk_head_dim = config.qk_head_dim if config.qk_head_dim is not None else config.head_dim
        self.v_head_dim = config.v_head_dim if config.v_head_dim is not None else config.head_dim

        self.kernel_size = getattr(config, "kernel_size", 32)
        self.stride = getattr(config, "stride", 16)
        self.block_size = getattr(config, "block_size", 64)
        self.top_n = getattr(config, "top_n", 16)
        self.num_local_blocks = getattr(config, "num_local_blocks", 1)
        self.num_init_blocks = getattr(config, "num_init_blocks", 2)
        self.window_size = getattr(config, "window_size", 512)

        self.cmp_k_method = getattr(config, "cmp_k_method", "mean")
        self.cmp_v_method = getattr(config, "cmp_v_method", "mean")

        self.cp_mode = getattr(config, "cp_mode", 2)
        self.kv_head_stride = getattr(config, "kv_head_stride", 1)
        self.use_overlap_swa = getattr(config, "use_overlap_swa", False)

        self.recompute_cmp_kv = getattr(config, "recompute_cmp_kv", False)
        self.recompute_cmp_o = getattr(config, "recompute_cmp_o", False)
        self.recompute_slc_o = getattr(config, "recompute_slc_o", False)
        self.recompute_swa_o = getattr(config, "recompute_swa_o", False)

        self.sm_scale = self.qk_head_dim ** -0.5

        assert math.log2(self.stride).is_integer(), "stride must be power of 2, like 8, 16, 32, 64, etc."
        assert self.kernel_size % self.stride == 0
        assert self.block_size % self.stride == 0

        self.process_group = process_group
        NSAHelper.set_hyperparameters(
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            block_size=self.block_size, 
            window_size=self.window_size, 
            top_n=self.top_n, 
            num_init_blocks=self.num_init_blocks, 
            num_local_blocks=self.num_local_blocks
        )

        NSAHelper.set_recompute(
            recompute_cmp_kv=self.recompute_cmp_kv, 
            recompute_cmp_o=self.recompute_cmp_o, 
            recompute_slc_o=self.recompute_slc_o, 
            recompute_swa_o=self.recompute_swa_o
        )

        NSAHelper.set_context_parallel(
            process_group=self.process_group,
            kv_head_stride=self.kv_head_stride, 
            use_overlap_swa=self.use_overlap_swa
        )

        NSAHelper.set_cmp_kv_method(cmp_k_method=self.cmp_k_method, cmp_v_method=self.cmp_v_method)
        NSAHelper.set_pipeline_parallel_world_size(pp_size=pp_size)


        self.cmp_k_weight = None
        self.cmp_v_weight = None

        if self.cmp_k_method != "mean":
            out_features = self.qk_head_dim if self.cmp_k_method == "linear" else self.qk_head_dim * 2
            self.cmp_k_weight = torch.nn.Parameter(torch.randn(out_features, self.kernel_size * self.qk_head_dim, device=torch.cuda.current_device()))
            setattr(self.cmp_k_weight, "sequence_parallel", getattr(config.sequence_parallel, True))

        if self.cmp_v_method != "mean":
            out_features = self.v_head_dim if self.cmp_v_method == "linear" else self.v_head_dim * 2
            self.cmp_v_weight = torch.nn.Parameter(torch.randn(out_features, self.kernel_size * self.v_head_dim, device=torch.cuda.current_device()))
            setattr(self.cmp_v_weight, "sequence_parallel", getattr(config.sequence_parallel, True))

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        combine_weight: torch.Tensor, 
        cu_seqlens: torch.IntTensor, 
        cu_seqlens_np: np.ndarray = None, 
        cp_mode: Union[int, None] = None, 
    ) -> torch.Tensor:
        """
        Forward pass for the NSA Attention module.

        Args:
            q (torch.Tensor): [t, num_q_head, qk_head_dim]
            k (torch.Tensor): [t, num_kv_head, qk_head_dim]
            v (torch.Tensor): [t, num_kv_head, v_head_dim]
            combine_weight (torch.Tensor): [t, num_q_head, 3]
            cu_seqlens (Int32 tensor): [batch+1].
            cu_seqlens_np (np.ndarray): [batch+1], default None.
            cp_mode (int): 1 or 2, defualt None.

        Return:
            o (torch.Tensor): [t, num_q_head, v_head_dim]
        """
        assert cu_seqlens is not None
        cp_mode = cp_mode if cp_mode is not None else self.cp_mode
        NSAHelper.init_cu_seqlens(cu_seqlens, x_cu_seqlens_np=cu_seqlens_np, cp_mode=cp_mode)
        if NSAHelper.world_size <= 1:
            combine_o = flash_nsa_varlen_func(q, k, v, combine_weight, self.cmp_k_weight, self.cmp_v_weight, self.sm_scale)
        else:
            combine_o = cp_flash_nsa_varlen_func(q, k, v, combine_weight, self.cmp_k_weight, self.cmp_v_weight, self.sm_scale)
        return combine_o

if not NSAHelper.is_triton34():
    
    from .hf_level_nsa import HFNSACore
    class NSACore(HFNSACore):
        def __init__(self, config: NSAConfig):
            print("Megatron level NSA must have triton-3.4 or later version. Now it transforms to HuggingFace level NSA")
            super().__init__(config)