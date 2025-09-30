import torch

HAVE_FA3 = False
HAVE_FA2 = False

try:
    from flash_attn_interface import (
        flash_attn_varlen_func as flash_attn_varlen_func_v3,
        _flash_attn_forward as _flash_attn_forward_v3, 
        _flash_attn_backward as _flash_attn_backward_v3
    )
    HAVE_FA3 = True
except:
    pass

try:
    from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_v2
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_backward
    import flash_attn_2_cuda
    HAVE_FA2 = True
except:
    pass

from ..utils import NSAHelper

def swa_varlen_func(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    sm_scale: float = None
) -> torch.Tensor:
    if HAVE_FA3:
        return flash_attn_varlen_func_v3(q, k, v, NSAHelper.x_cu_seqlens, NSAHelper.x_cu_seqlens, NSAHelper.x_maxlen, NSAHelper.x_maxlen,
                                         window_size=(NSAHelper.window_size, -1), softmax_scale=sm_scale, causal=True)
    elif HAVE_FA2:
        return flash_attn_varlen_func_v2(q, k, v, NSAHelper.x_cu_seqlens, NSAHelper.x_cu_seqlens, NSAHelper.x_maxlen, NSAHelper.x_maxlen,
                                         window_size=(NSAHelper.window_size, -1), softmax_scale=sm_scale, causal=True)
    else:
        raise ImportError("Neither flash_attn_interface nor flash_attn is available.")


def swa_fwd(q, k, v, sm_scale=None, out=None, helper=NSAHelper):
    if sm_scale is None:
        sm_scale = q.size(-1) ** 2
    if HAVE_FA3:
        out, softmax_lse, *rest = _flash_attn_forward_v3(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            None,  # qv
            out,  # out
            helper.x_cu_seqlens,
            helper.x_cu_seqlens,
            None,  # cu_seqlens_k_new
            None, # seqused_q
            None, # seqused_q
            helper.x_maxlen,
            helper.x_maxlen,
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            None, None, None, # qkv descale
            sm_scale,
            causal=True,
            window_size=(NSAHelper.window_size, -1),
            attention_chunk=0,
            softcap=0.0,
            num_splits=1,
            pack_gqa=None,
            sm_margin=0,
        )
    elif HAVE_FA2:
        out, softmax_lse, S_dmask, rng_state = flash_attn_2_cuda.varlen_fwd(
            q,
            k,
            v,
            out,
            helper.x_cu_seqlens,
            helper.x_cu_seqlens,
            None, #seqused_k,
            None, #leftpad_k,
            None, #block_table,
            None, #alibi_slopes,
            helper.x_maxlen,
            helper.x_maxlen,
            0.0, # dropout
            sm_scale,
            False, # zero_tensors
            True, # causal
            NSAHelper.window_size,
            -1, # right window size
            0.0, # softcap
            False, # return_softmax,
            None,
        )
    else:
        raise ImportError("Neither flash_attn_interface nor flash_attn is available.")

    return out, softmax_lse

def swa_bwd(q, k, v, out, lse, dout, sm_scale=None, helper=NSAHelper):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    if sm_scale is None:
        sm_scale = q.size(-1) ** 2

    if HAVE_FA3:
        _flash_attn_backward_v3(
            dout,
            q,
            k,
            v,
            out,
            lse, # softmax_lse,
            helper.x_cu_seqlens, helper.x_cu_seqlens, # cu_seqlens_q, cu_seqlens_k,
            None, None, # sequed_q, sequed_k,
            helper.x_maxlen, helper.x_maxlen, # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            sm_scale, #ctx.softmax_scale,
            True, #ctx.causal,
            (NSAHelper.window_size, -1), #ctx.window_size,
            0.0, #ctx.softcap,
            False, # deterministic,
            0, #c tx.sm_margin,
        )
    elif HAVE_FA2:
        _wrapped_flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            helper.x_cu_seqlens, #cu_seqlens_q,
            helper.x_cu_seqlens, #cu_seqlens_k,
            helper.x_maxlen, #ctx.max_seqlen_q,
            helper.x_maxlen, #ctx.max_seqlen_k,
            0.0, # ctx.dropout_p,
            sm_scale, # ctx.softmax_scale,
            True, #ctx.causal,
            NSAHelper.window_size, #ctx.window_size[0],
            -1, # ctx.window_size[1],
            0.0, #ctx.softcap,
            None, # ctx.alibi_slopes,
            False, # ctx.deterministic,
            rng_state=None,
        )
    else:
        raise ImportError("Neither flash_attn_interface nor flash_attn is available.")

    return dq, dk, dv



def two_stage_swa_fwd(q, k, v, out, sm_scale=None, stage=1, helper=NSAHelper):
    if not NSAHelper.use_overlap_swa and stage == 1:
        return [None, None]

    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5

    seqused_k = None
    leftpad_k = None
    block_table = None
    alibi_slopes = None
    dropout_p = 0
    zero_tensors = False
    causal = True
    window_size_left = NSAHelper.window_size
    window_size_right = -1
    softcap = 0.
    return_softmax = False

    if helper.cp_mode == 1:
        q = q[helper.swa_q_start_1:] if stage == 1 else q[:helper.swa_q_start_1]
        k = k[helper.swa_kv_start_1:] if stage == 1 else k[helper.swa_kv_start_2: helper.swa_kv_end_2]
        v = v[helper.swa_kv_start_1:] if stage == 1 else v[helper.swa_kv_start_2: helper.swa_kv_end_2]
        out = out[helper.swa_q_start_1:] if stage == 1 else out[:helper.swa_q_start_1]
        cu_seqlens_q = helper.swa_cu_seqlens_q1 if stage == 1 else helper.swa_cu_seqlens_q2
        cu_seqlens_k = helper.swa_cu_seqlens_k1 if stage == 1 else helper.swa_cu_seqlens_k2
        max_seqlen_q = helper.swa_max_q1 if stage == 1 else helper.swa_max_q2
        max_seqlen_k = helper.swa_max_k1 if stage == 1 else helper.swa_max_k2
        if HAVE_FA3:
            out, softmax_lse, *rest = _flash_attn_forward_v3(
                q,
                k,
                v,
                None, None,  # k_new, v_new
                None,  # qv
                out,  # out
                cu_seqlens_q,
                cu_seqlens_k,
                None,  # cu_seqlens_k_new
                None, # seqused_q
                None, # seqused_q
                max_seqlen_q,
                max_seqlen_k,
                None, None, None,   # page_table, kv_batch_idx, leftpad_k,
                None, None, None,  # rotary_cos/sin, seqlens_rotary
                None, None, None, # qkv descale
                sm_scale,
                causal=causal,
                window_size=(NSAHelper.window_size, -1),
                attention_chunk=0,
                softcap=0.0,
                num_splits=1,
                pack_gqa=None,
                sm_margin=0,
            )
        elif HAVE_FA2:
            out, softmax_lse, S_dmask, rng_state = flash_attn_2_cuda.varlen_fwd(
                q,
                k, 
                v,
                out,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                leftpad_k,
                block_table,
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                sm_scale,
                zero_tensors,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                return_softmax,
                None,
            )
        softmax_lse = [softmax_lse, None]
    else:
        q1, q2 = q.chunk(2)
        out1, out2 = out.chunk(2)
        if stage == 1:
            k1, k2 = k.chunk(2)
            v1, v2 = v.chunk(2)
        else:
            k1, k2 = k, k
            v1, v2 = v, v
        q = q1[helper.swa_q_start_1:] if stage == 1 else q1[:helper.swa_q_start_1]
        k = k1[helper.swa_kv_start_1:] if stage == 1 else k1[helper.swa_kv_start_2: helper.swa_kv_end_2]
        v = v1[helper.swa_kv_start_1:] if stage == 1 else v1[helper.swa_kv_start_2: helper.swa_kv_end_2]
        out = out1[helper.swa_q_start_1:] if stage == 1 else out1[:helper.swa_q_start_1]
        cu_seqlens_q = helper.swa_cu_seqlens_q1 if stage == 1 else helper.swa_cu_seqlens_q2
        cu_seqlens_k = helper.swa_cu_seqlens_k1 if stage == 1 else helper.swa_cu_seqlens_k2
        max_seqlen_q = helper.swa_max_q1 if stage == 1 else helper.swa_max_q2
        max_seqlen_k = helper.swa_max_k1 if stage == 1 else helper.swa_max_k2

        if HAVE_FA3:
            out, softmax_lse1, *rest = _flash_attn_forward_v3(
                q,
                k,
                v,
                None, None,  # k_new, v_new
                None,  # qv
                out,  # out
                cu_seqlens_q,
                cu_seqlens_k,
                None,  # cu_seqlens_k_new
                None, # seqused_q
                None, # seqused_q
                max_seqlen_q,
                max_seqlen_k,
                None, None, None,   # page_table, kv_batch_idx, leftpad_k,
                None, None, None,  # rotary_cos/sin, seqlens_rotary
                None, None, None, # qkv descale
                sm_scale,
                causal=causal,
                window_size=(NSAHelper.window_size, -1),
                attention_chunk=0,
                softcap=0.0,
                num_splits=1,
                pack_gqa=None,
                sm_margin=0,
            )
        elif HAVE_FA2:
            out, softmax_lse1, S_dmask, rng_state1 = flash_attn_2_cuda.varlen_fwd(
                q,
                k, 
                v,
                out,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                leftpad_k,
                block_table,
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                sm_scale,
                zero_tensors,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                return_softmax,
                None,
            )

        q = q2[helper.swa_q_start_2:] if stage == 1 else q2[:helper.swa_q_start_2]
        k = k2[helper.swa_kv_start_3:] if stage == 1 else k2[helper.swa_kv_start_4: helper.swa_kv_end_4]
        v = v2[helper.swa_kv_start_3:] if stage == 1 else v2[helper.swa_kv_start_4: helper.swa_kv_end_4]
        out = out2[helper.swa_q_start_2:] if stage == 1 else out2[:helper.swa_q_start_2]
        cu_seqlens_q = helper.swa_cu_seqlens_q3 if stage == 1 else helper.swa_cu_seqlens_q4
        cu_seqlens_k = helper.swa_cu_seqlens_k3 if stage == 1 else helper.swa_cu_seqlens_k4
        max_seqlen_q = helper.swa_max_q3 if stage == 1 else helper.swa_max_q4
        max_seqlen_k = helper.swa_max_k3 if stage == 1 else helper.swa_max_k4


        if HAVE_FA3:
            out, softmax_lse2, *rest = _flash_attn_forward_v3(
                q,
                k,
                v,
                None, None,  # k_new, v_new
                None,  # qv
                out,  # out
                cu_seqlens_q,
                cu_seqlens_k,
                None,  # cu_seqlens_k_new
                None, # seqused_q
                None, # seqused_q
                max_seqlen_q,
                max_seqlen_k,
                None, None, None,   # page_table, kv_batch_idx, leftpad_k,
                None, None, None,  # rotary_cos/sin, seqlens_rotary
                None, None, None, # qkv descale
                sm_scale,
                causal=causal,
                window_size=(NSAHelper.window_size, -1),
                attention_chunk=0,
                softcap=0.0,
                num_splits=1,
                pack_gqa=None,
                sm_margin=0,
            )
        elif HAVE_FA2:
            out, softmax_lse2, S_dmask, rng_state1 = flash_attn_2_cuda.varlen_fwd(
                q,
                k, 
                v,
                out,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                leftpad_k,
                block_table,
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                sm_scale,
                zero_tensors,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                return_softmax,
                None,
            )
        softmax_lse = [softmax_lse1, softmax_lse2]
    return softmax_lse

def two_stage_swa_bwd(q, k, v, out, lse, dout, dq, dk, dv, sm_scale=None, stage=1, helper=NSAHelper):
    if not NSAHelper.use_overlap_swa and stage == 1:
        return 

    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5
    if helper is None:
        helper = NSAHelper


    alibi_slopes = None
    dropout_p = 0
    causal = True
    window_size_left = NSAHelper.window_size
    window_size_right = -1
    softcap = 0.
    deterministic=False

    if helper.cp_mode == 1:
        lse = lse[0] if not isinstance(lse, torch.Tensor) else lse
        q = q[helper.swa_q_start_1:] if stage == 1 else q[:helper.swa_q_start_1]
        k = k[helper.swa_kv_start_1:] if stage == 1 else k[helper.swa_kv_start_2: helper.swa_kv_end_2]
        v = v[helper.swa_kv_start_1:] if stage == 1 else v[helper.swa_kv_start_2: helper.swa_kv_end_2]
        out = out[helper.swa_q_start_1:] if stage == 1 else out[:helper.swa_q_start_1]
        _dq = dq[helper.swa_q_start_1:] if stage == 1 else dq[:helper.swa_q_start_1]
        part_dk = dk[helper.cp_bos1 + helper.swa_kv_start_1: helper.cp_eos1] if stage == 1 else dk[helper.swa_kv_start_2: helper.swa_kv_end_2]
        part_dv = dv[helper.cp_bos1 + helper.swa_kv_start_1: helper.cp_eos1] if stage == 1 else dv[helper.swa_kv_start_2: helper.swa_kv_end_2]

        # dk = dk[helper.swa_kv_start_1:] if stage == 1 else dk[helper.swa_kv_start_2: helper.swa_kv_end_2]
        # dv = dv[helper.swa_kv_start_1:] if stage == 1 else dv[helper.swa_kv_start_2: helper.swa_kv_end_2]
        dout = dout[helper.swa_q_start_1:] if stage == 1 else dout[:helper.swa_q_start_1]
        cu_seqlens_q = helper.swa_cu_seqlens_q1 if stage == 1 else helper.swa_cu_seqlens_q2
        cu_seqlens_k = helper.swa_cu_seqlens_k1 if stage == 1 else helper.swa_cu_seqlens_k2
        max_seqlen_q = helper.swa_max_q1 if stage == 1 else helper.swa_max_q2
        max_seqlen_k = helper.swa_max_k1 if stage == 1 else helper.swa_max_k2

        _dk, _dv = torch.empty_like(part_dk, dtype=k.dtype), torch.empty_like(part_dv, dtype=v.dtype)
        if HAVE_FA3:
            _flash_attn_backward_v3(
                dout,
                q,
                k,
                v,
                out,
                lse, # softmax_lse,
                cu_seqlens_q, cu_seqlens_k, # cu_seqlens_q, cu_seqlens_k,
                None, None, # sequed_q, sequed_k,
                max_seqlen_q, max_seqlen_k, # max_seqlen_q, max_seqlen_k,
                _dq,
                _dk,
                _dv,
                sm_scale, #ctx.softmax_scale,
                True, #ctx.causal,
                (NSAHelper.window_size, -1), #ctx.window_size,
                0.0, #ctx.softcap,
                False, # deterministic,
                0, #c tx.sm_margin,
            )
        elif HAVE_FA2:
            _wrapped_flash_attn_varlen_backward(
                dout,
                q,
                k,
                v,
                out,
                lse,
                _dq,
                _dk,
                _dv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                sm_scale,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                alibi_slopes,
                deterministic,
                None,
            )
        part_dk += _dk
        part_dv += _dv
    else:
        q1, q2 = q.chunk(2)
        out1, out2 = out.chunk(2)
        dq1, dq2 = dq.chunk(2)
        dout1, dout2 = dout.chunk(2)
        if stage == 1:
            k1, k2 = k.chunk(2)
            v1, v2 = v.chunk(2)
        else:
            k1, k2 = k, k
            v1, v2 = v, v

        q = q1[helper.swa_q_start_1:] if stage == 1 else q1[:helper.swa_q_start_1]
        k = k1[helper.swa_kv_start_1:] if stage == 1 else k1[helper.swa_kv_start_2: helper.swa_kv_end_2]
        v = v1[helper.swa_kv_start_1:] if stage == 1 else v1[helper.swa_kv_start_2: helper.swa_kv_end_2]
        out = out1[helper.swa_q_start_1:] if stage == 1 else out1[:helper.swa_q_start_1]
        _dq = dq1[helper.swa_q_start_1:] if stage == 1 else dq1[:helper.swa_q_start_1]
        part_dk = dk[helper.cp_bos1 + helper.swa_kv_start_1: helper.cp_eos1] if stage == 1 else dk[helper.swa_kv_start_2: helper.swa_kv_end_2]
        part_dv = dv[helper.cp_bos1 + helper.swa_kv_start_1: helper.cp_eos1] if stage == 1 else dv[helper.swa_kv_start_2: helper.swa_kv_end_2]
        dout = dout1[helper.swa_q_start_1:] if stage == 1 else dout1[:helper.swa_q_start_1]
        cu_seqlens_q = helper.swa_cu_seqlens_q1 if stage == 1 else helper.swa_cu_seqlens_q2
        cu_seqlens_k = helper.swa_cu_seqlens_k1 if stage == 1 else helper.swa_cu_seqlens_k2
        max_seqlen_q = helper.swa_max_q1 if stage == 1 else helper.swa_max_q2
        max_seqlen_k = helper.swa_max_k1 if stage == 1 else helper.swa_max_k2

        _dk, _dv = torch.empty_like(part_dk, dtype=k.dtype), torch.empty_like(part_dv, dtype=v.dtype)
        if HAVE_FA3:
            _flash_attn_backward_v3(
                dout,
                q,
                k,
                v,
                out,
                lse[0], # softmax_lse,
                cu_seqlens_q, cu_seqlens_k, # cu_seqlens_q, cu_seqlens_k,
                None, None, # sequed_q, sequed_k,
                max_seqlen_q, max_seqlen_k, # max_seqlen_q, max_seqlen_k,
                _dq,
                _dk,
                _dv,
                sm_scale, #ctx.softmax_scale,
                True, #ctx.causal,
                (NSAHelper.window_size, -1), #ctx.window_size,
                0.0, #ctx.softcap,
                False, # deterministic,
                0, #c tx.sm_margin,
            )
        elif HAVE_FA2:
            _wrapped_flash_attn_varlen_backward(
                dout,
                q,
                k,
                v,
                out,
                lse[0],
                _dq,
                _dk,
                _dv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                sm_scale,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                alibi_slopes,
                deterministic,
                None,
            )

        part_dk += _dk
        part_dv += _dv

        q = q2[helper.swa_q_start_2:] if stage == 1 else q2[:helper.swa_q_start_2]
        k = k2[helper.swa_kv_start_3:] if stage == 1 else k2[helper.swa_kv_start_4: helper.swa_kv_end_4]
        v = v2[helper.swa_kv_start_3:] if stage == 1 else v2[helper.swa_kv_start_4: helper.swa_kv_end_4]
        out = out2[helper.swa_q_start_2:] if stage == 1 else out2[:helper.swa_q_start_2]
        _dq = dq2[helper.swa_q_start_2:] if stage == 1 else dq2[:helper.swa_q_start_2]
        part_dk = dk[helper.cp_bos2 + helper.swa_kv_start_3: helper.cp_eos2] if stage == 1 else dk[helper.swa_kv_start_4: helper.swa_kv_end_4]
        part_dv = dv[helper.cp_bos2 + helper.swa_kv_start_3: helper.cp_eos2] if stage == 1 else dv[helper.swa_kv_start_4: helper.swa_kv_end_4]
        dout = dout2[helper.swa_q_start_2:] if stage == 1 else dout2[:helper.swa_q_start_2]
        cu_seqlens_q = helper.swa_cu_seqlens_q3 if stage == 1 else helper.swa_cu_seqlens_q4
        cu_seqlens_k = helper.swa_cu_seqlens_k3 if stage == 1 else helper.swa_cu_seqlens_k4
        max_seqlen_q = helper.swa_max_q3 if stage == 1 else helper.swa_max_q4
        max_seqlen_k = helper.swa_max_k3 if stage == 1 else helper.swa_max_k4

        _dk, _dv = torch.empty_like(part_dk, dtype=k.dtype), torch.empty_like(part_dv, dtype=v.dtype)
        if HAVE_FA3:
            _flash_attn_backward_v3(
                dout,
                q,
                k,
                v,
                out,
                lse[1], # softmax_lse,
                cu_seqlens_q, cu_seqlens_k, # cu_seqlens_q, cu_seqlens_k,
                None, None, # sequed_q, sequed_k,
                max_seqlen_q, max_seqlen_k, # max_seqlen_q, max_seqlen_k,
                _dq,
                _dk,
                _dv,
                sm_scale, #ctx.softmax_scale,
                True, #ctx.causal,
                (NSAHelper.window_size, -1), #ctx.window_size,
                0.0, #ctx.softcap,
                False, # deterministic,
                0, #c tx.sm_margin,
            )
        elif HAVE_FA2:
            _wrapped_flash_attn_varlen_backward(
                dout,
                q,
                k,
                v,
                out,
                lse[1],
                _dq,
                _dk,
                _dv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                sm_scale,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                alibi_slopes,
                deterministic,
                None,
            )

        part_dk += _dk
        part_dv += _dv

