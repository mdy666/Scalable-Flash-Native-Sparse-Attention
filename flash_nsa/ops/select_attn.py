# Copyright (c) 2025 Duyue Ma

import math

import torch
import triton
import triton.language as tl
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except:
    TensorDescriptor = None

from ..utils import NSAHelper, use_tma
from .topk import get_bind_from_find
from . import ampere_ops

# @triton.autotune([triton.Config({}, num_warps=nw, num_stages=ns)
#                  for nw in [1, 2, 4, 8]
#                  for ns in [1,2,3,4]
#                  ], key=["D1", "D2", "VD" 'BLOCK_H', 'BLOCK_M'])
@triton.jit
def _fwd_kernel(
    desc_q,
    desc_q2, 
    desc_k,
    desc_k2,
    desc_v,
    desc_o, 
    Lse, 
    Ind,
    X_CU_SEQLENS, 
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    lse_stride_n, lse_stride_h,
    ind_stride_h, ind_stride_n, ind_stride_k,
    sm_scale, 
    top_n: tl.constexpr,
    KH: tl.constexpr,
    G: tl.constexpr,
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_H: tl.constexpr=16, 
    BLOCK_M: tl.constexpr=64,
):
    cp_idx = tl.program_id(0)
    off_bh = tl.program_id(1)
    cp_off_b = off_bh // KH
    off_kh = off_bh % KH
    start_qh = off_kh * G
    if not CP:
        off_b = cp_off_b
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if cp_idx >= x_len:
            return
        cp_offset = 0
        cp_bos = x_bos
        cp_len = x_len
        q_idx = cp_idx
    else:
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if cp_idx >= cp_len:
            return
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        q_idx = cp_idx + cp_offset


    Ind += off_kh * ind_stride_h + (cp_bos + cp_idx) * ind_stride_n
    Lse += (cp_bos + cp_idx) * lse_stride_n + start_qh * lse_stride_h

    q = desc_q.load([cp_bos + cp_idx, off_kh, 0, 0]).reshape(BLOCK_H, D1)
    if D2 > 0:
        q2 = desc_q2.load([cp_bos + cp_idx, off_kh, 0, D1]).reshape(BLOCK_H, D2)

    sm_scale *= 1.44269504
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, VD], dtype=tl.float32)

    stop_n = tl.constexpr(tl.minimum(top_n, tl.cdiv(q_idx+1, BLOCK_M)))
    for i in tl.range(0, stop_n, flatten=True):
        start_m = tl.load(Ind + i) * BLOCK_M
        k = desc_k.load([x_bos + start_m, off_kh, 0]).reshape(BLOCK_M, D1)
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            k2 = desc_k2.load([x_bos + start_m, off_kh, D1]).reshape(BLOCK_M, D2)
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        v = desc_v.load([x_bos + start_m, off_kh, 0]).reshape(BLOCK_M, VD)
        attn_score = tl.where(q_idx >= (start_m + tl.arange(0, BLOCK_M))[None, :], attn_score * sm_scale, float('-inf'))
        new_m_i = tl.maximum(m_i, tl.max(attn_score, axis=1))
        alpha = tl.exp2(m_i - new_m_i)
        exp_attn_score = tl.exp2(attn_score - new_m_i[:, None])
        l_i = tl.fma(l_i, alpha, tl.sum(exp_attn_score, axis=-1))
        acc = acc * alpha[:, None] + tl.dot(exp_attn_score.to(v.dtype), v)
        m_i = new_m_i

    acc /= l_i[:, None]
    lse = m_i + tl.log2(l_i)

    desc_o.store([cp_bos + cp_idx, off_kh, 0, 0], acc.reshape(1, 1, BLOCK_H, VD).to(desc_o.dtype))
    tl.store(Lse + tl.arange(0, BLOCK_H), lse, mask=tl.arange(0, BLOCK_H) < G)

# @triton.autotune([triton.Config({'BLOCK_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [16, 32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['T'])
@triton.jit
def _bwd_preprocess(
    O, 
    DO, 
    Delta,
    o_stride_n, o_stride_h, o_stride_d,
    do_stride_n, do_stride_h, do_stride_d,
    delta_stride_n, delta_stride_h,
    T, 
    VD: tl.constexpr,
    BLOCK_N: tl.constexpr=16
):

    off_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    off_h = tl.program_id(1)

    O += off_h * o_stride_h
    DO += off_h * do_stride_h
    Delta += off_h * delta_stride_h

    cols = tl.arange(0, VD)
    o = tl.load(O + off_n[:, None] * o_stride_n + cols[None, :], mask=off_n[:, None] < T, other=0.).to(tl.float32)
    do = tl.load(DO + off_n[:, None] * do_stride_n + cols[None, :], mask=off_n[:, None] < T, other=0.).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_n * delta_stride_n , delta, mask=off_n < T)


# @triton.autotune([triton.Config({'BLOCK_N':bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['D1', 'D2', "BLOCK_M"])
@triton.jit
def _dkdv_kernel(
    DQ, 
    DK, 
    DV, 
    DO, 
    Q, 
    K, 
    V, 
    Lse, 
    Delta, 
    X_CU_SEQLENS, 
    K_CU_SEQLENS,
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET, 
    CP_K_CU_SEQLENS,
    Ind, 
    Count,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    v_stride_m, v_stride_h, v_stride_d,
    dq_stride_n, dq_stride_h, dq_stride_d,
    dk_stride_m, dk_stride_h, dk_stride_d,
    dv_stride_m, dv_stride_h, dv_stride_d,
    do_stride_n, do_stride_h, do_stride_d,
    lse_stride_n, lse_stride_h,
    ind_stride_h, ind_stride_m, ind_stride_n,
    cnt_stride_h, cnt_stride_m,
    sm_scale, 
    ATOMIC: tl.constexpr, 
    REPEAT: tl.constexpr, 
    COMPUTE_DQ: tl.constexpr,
    G: tl.constexpr, 
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(2)
    start_m = pid * BLOCK_M
    cp_off_b = tl.program_id(0)
    off_qh = tl.program_id(1)
    off_kh = off_qh // G

    if not CP:
        off_b = cp_off_b
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if start_m >= x_len:
            return
        k_bos = tl.load(K_CU_SEQLENS + off_b)
        cp_bos = x_bos
        cp_len = x_len
        cp_k_bos = k_bos
        cp_offset = 0
    else:
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        if start_m >= cp_offset + cp_len:
            return
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        cp_k_bos = tl.load(CP_K_CU_SEQLENS + cp_off_b)

    if REPEAT:
        off_dkdvh = off_qh
    else:
        off_dkdvh = off_kh

    Q += cp_bos * q_stride_n + off_qh * q_stride_h
    K += x_bos * k_stride_m + off_kh * k_stride_h
    V += x_bos * v_stride_m + off_kh * v_stride_h
    DK += x_bos * dk_stride_m + off_dkdvh * dk_stride_h
    DV += x_bos * dv_stride_m + off_dkdvh * dv_stride_h
    DO += cp_bos * do_stride_n + off_qh * do_stride_h
    Lse += cp_bos * lse_stride_n + off_qh * lse_stride_h 
    Delta += cp_bos * lse_stride_n + off_qh * lse_stride_h 
    Ind += off_kh * ind_stride_h.to(tl.int64) + (cp_k_bos + pid) * ind_stride_m.to(tl.int64)
    Count += (cp_k_bos + pid) * cnt_stride_m + off_kh * cnt_stride_h
    if COMPUTE_DQ:
        DQ += cp_bos * dq_stride_n + off_qh * dq_stride_h

    off_m = start_m + tl.arange(0, BLOCK_M)
    desc_k = tl.make_tensor_descriptor(K, (x_len, D1), (k_stride_m, k_stride_d), (BLOCK_M, D1))
    desc_v = tl.make_tensor_descriptor(V, (x_len, VD), (v_stride_m, v_stride_d), (BLOCK_M, VD))
    desc_dk = tl.make_tensor_descriptor(DK, (x_len, D1), (dk_stride_m, dk_stride_d), (BLOCK_M, D1))
    desc_dv = tl.make_tensor_descriptor(DV, (x_len, D1), (dv_stride_m, dv_stride_d), (BLOCK_M, D1))
    if D2 > 0:
        desc_k2 = tl.make_tensor_descriptor(K + D1, (x_len, D2), (k_stride_m, k_stride_d), (BLOCK_M, D2))
        desc_dk2 = tl.make_tensor_descriptor(DK + D1, (x_len, D2), (dk_stride_m, dk_stride_d), (BLOCK_M, D2))
    k = desc_k.load([start_m, 0])
    v = desc_v.load([start_m, 0])
    acc_dk = tl.zeros((BLOCK_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_M, VD), dtype=tl.float32)

    if D2 > 0:
        k2 = desc_k2.load([start_m, 0])
        acc_dk2 = tl.zeros((BLOCK_M, D2), dtype=tl.float32)

    sm_scale_ln2 = sm_scale * 1.44269504
    count = tl.load(Count)
    mid = tl.minimum(tl.cdiv(BLOCK_M, BLOCK_N) * BLOCK_N, count)
    for start in range(0, mid, BLOCK_N):
        off_ind = start + tl.arange(0, BLOCK_N)
        cp_idx = tl.load(Ind + off_ind, off_ind < count, other=0)
        q_idx = tl.where(off_ind < count, cp_idx + cp_offset, -1)
        q = tl.load(Q + cp_idx[:, None] * q_stride_n + tl.arange(0, D1)[None, :], mask=q_idx[:, None] >= 0, other=0.)
        do = tl.load(DO + cp_idx[:, None] * do_stride_n + tl.arange(0, VD)[None, :], mask=q_idx[:, None] >= 0, other=0.)
        lse = tl.load(Lse + cp_idx * lse_stride_n, mask=q_idx >= 0, other=0.)
        delta = tl.load(Delta + cp_idx * lse_stride_n, mask=q_idx >= 0, other=0.)

        attn_score = tl.dot(q, tl.permute(k, 1, 0)) 
        if D2 > 0:
            q2 = tl.load(Q + cp_idx[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1, mask=q_idx[:, None] >= 0, other=0.)
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        attn_score = tl.where(q_idx[:, None] >= off_m[None, :], attn_score, float('-inf'))
        p = tl.exp2(attn_score * sm_scale_ln2 - lse[:, None])

        acc_dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, acc_dv)

        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])
        ds = ds.to(k.dtype)

        acc_dk = tl.dot(tl.permute(ds, 1, 0), q, acc_dk)
        if D2 > 0:
            acc_dk2 = tl.dot(tl.trans(ds, 1, 0), q2, acc_dk2)

        if COMPUTE_DQ:
            dq = tl.dot(ds, k) * sm_scale
            tl.atomic_add(DQ + cp_idx[:, None] * dq_stride_n + tl.arange(0, D1)[None, :], dq, mask=q_idx[:, None] >= 0)
            if D2 > 0:
                dq2 = tl.dot(ds, k2) * sm_scale
                tl.atomic_add(DQ + cp_idx[:, None] * dq_stride_n + tl.arange(0, D2)[None, :] + D1, dq2, mask=q_idx[:, None] >= 0)

    for start in range(mid, count, BLOCK_N):
        off_ind = start + tl.arange(0, BLOCK_N)
        cp_idx = tl.load(Ind + off_ind, off_ind < count, other=0)
        q_idx = tl.where(off_ind < count, cp_idx + cp_offset, -1)
        q = tl.load(Q + cp_idx[:, None] * q_stride_n + tl.arange(0, D1)[None, :], mask=q_idx[:, None] >= 0, other=0.)
        do = tl.load(DO + cp_idx[:, None] * do_stride_n + tl.arange(0, VD)[None, :], mask=q_idx[:, None] >= 0, other=0.)
        lse = tl.load(Lse + cp_idx * lse_stride_n, mask=q_idx >= 0, other=0.)
        delta = tl.load(Delta + cp_idx * lse_stride_n, mask=q_idx >= 0, other=0.)

        attn_score = tl.dot(q, tl.permute(k, 1, 0)) 
        if D2 > 0:
            q2 = tl.load(Q + cp_idx[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1, mask=q_idx[:, None] >= 0, other=0.)
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        p = tl.exp2(attn_score * sm_scale_ln2 - lse[:, None])

        acc_dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, acc_dv)

        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])
        ds = ds.to(k.dtype)

        acc_dk = tl.dot(tl.permute(ds, 1, 0), q, acc_dk)
        if D2 > 0:
            acc_dk2 = tl.dot(tl.trans(ds, 1, 0), q2, acc_dk2)

        if COMPUTE_DQ:
            dq = tl.dot(ds, k) * sm_scale
            tl.atomic_add(DQ + cp_idx[:, None] * dq_stride_n + tl.arange(0, D1)[None, :], dq, mask=q_idx[:, None] >= 0)
            if D2 > 0:
                dq2 = tl.dot(ds, k2) * sm_scale
                tl.atomic_add(DQ + cp_idx[:, None] * dq_stride_n + tl.arange(0, D2)[None, :] + D1, dq2, mask=q_idx[:, None] >= 0)

    if not ATOMIC:
        desc_dk.store([start_m, 0], (acc_dk * sm_scale).to(desc_dk.dtype))
        desc_dv.store([start_m, 0], acc_dv.to(desc_dv.dtype))
        if D2 > 0:
            desc_dk2.store([start_m, 0], (acc_dk2 * sm_scale).to(desc_dk2.dtype))
    else:
        desc_dk.atomic_add([start_m, 0], (acc_dk * sm_scale).to(desc_dk.dtype))
        desc_dv.atomic_add([start_m, 0], acc_dv.to(desc_dv.dtype))
        if D2 > 0:
            desc_dk2.atomic_add([start_m, 0], (acc_dk2 * sm_scale).to(desc_dk2.dtype))



# @triton.autotune([triton.Config({}, num_warps=nw, num_stages=ns)
#                  for nw in [1, 2, 4, 8]
#                  for ns in [1, 2, 3, 4]
#                  ], key=['D1',"D2", "BLOCK_H", "BLOCK_M", "ATOMIC"])
@triton.jit
def _dq_kernel( 
    desc_q,
    desc_q2, 
    desc_k,
    desc_k2,
    desc_v,
    desc_do, 
    desc_dq,
    desc_dq2,
    Lse, 
    Delta,
    Ind,
    X_CU_SEQLENS,
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    lse_stride_n, lse_stride_h,
    ind_stride_h, ind_stride_n, ind_stride_k,
    sm_scale, 
    top_n, 
    ATOMIC: tl.constexpr, 
    KH: tl.constexpr, 
    G:tl.constexpr,
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_H: tl.constexpr, 
    BLOCK_M: tl.constexpr,
):

    cp_idx = tl.program_id(0)
    off_bh = tl.program_id(1)
    cp_off_b = off_bh // KH
    off_kh = off_bh % KH
    start_qh = off_kh * G

    if not CP:
        off_b = cp_off_b
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if cp_idx >= x_len:
            return
        cp_offset = 0
        cp_bos = x_bos
        cp_len = x_len
    else:
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if cp_idx >= cp_len:
            return
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos

    q_idx = cp_idx + cp_offset

    Ind += off_kh * ind_stride_h + (cp_bos + cp_idx) * ind_stride_n
    Lse += (cp_bos + cp_idx) * lse_stride_n + start_qh * lse_stride_h
    Delta += (cp_bos + cp_idx) * lse_stride_n + start_qh * lse_stride_h

    q = desc_q.load([cp_bos + cp_idx, off_kh, 0, 0]).reshape(BLOCK_H, D1)
    do = desc_do.load([cp_bos + cp_idx, off_kh, 0,0]).reshape(BLOCK_H, VD)
    lse = tl.load(Lse + tl.arange(0, BLOCK_H) * lse_stride_h, mask=tl.arange(0, BLOCK_H)<G, other=0.)
    delta = tl.load(Delta + tl.arange(0, BLOCK_H) * lse_stride_h, mask=tl.arange(0, BLOCK_H)<G, other=0.)
    acc_dq = tl.zeros([BLOCK_H, D1], dtype=tl.float32)

    if D2 > 0:
        q2 = desc_q2.load([cp_bos + cp_idx, off_kh, 0, D1]).reshape(BLOCK_H, D2)
        acc_dq2 = tl.zeros([BLOCK_H, D2], dtype=tl.float32)

    sm_scale_ln2 = sm_scale * 1.44269504
    stop_n = tl.minimum(top_n, tl.cdiv(q_idx+1, BLOCK_M))
    for i in range(0, stop_n):
        start_m = tl.load(Ind + i).to(tl.int32) * BLOCK_M
        k = desc_k.load([x_bos + start_m, off_kh, 0]).reshape(BLOCK_M, D1)

        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            k2 = desc_k2.load([x_bos + start_m, off_kh, D1]).reshape(BLOCK_M, D2)
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

        attn_score = tl.where(q_idx >= (start_m + tl.arange(0, BLOCK_M))[None, :], attn_score * sm_scale_ln2, float('-inf'))
        p = tl.exp2(attn_score - lse[:, None])

        v = desc_v.load([x_bos + start_m, off_kh, 0]).reshape(BLOCK_M, VD)
        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None]) 
        acc_dq = tl.dot(ds.to(k.dtype), k, acc_dq)
        if D2 > 0:
            acc_dq2 = tl.dot(ds.to(k.dtype), k2, acc_dq2)

    if not ATOMIC:
        desc_dq.store([cp_bos + cp_idx, off_kh, 0,0], (acc_dq.reshape(1, 1, BLOCK_H, D1) * sm_scale).to(desc_dq.dtype))
        if D2 > 0:
            desc_dq2.store([cp_bos + cp_idx, off_kh, 0, D1], (acc_dq2.reshape(1, 1, BLOCK_H, D2) * sm_scale).to(desc_dq2.dtype))
    else:
        desc_dq.atomic_add([cp_bos + cp_idx, off_kh, 0,0], (acc_dq.reshape(1, 1, BLOCK_H, D1) * sm_scale).to(desc_dq.dtype))
        if D2 > 0:
            desc_dq2.atomic_add([cp_bos + cp_idx, off_kh, 0, D1], (acc_dq2.reshape(1, 1, BLOCK_H, D2) * sm_scale).to(desc_dq2.dtype))

@use_tma
def slc_attn_fwd(q, k, v, topk, sm_scale=None, o=None, helper=NSAHelper):
    if NSAHelper.is_use_ampere_ops():
        return ampere_ops.slc_attn_fwd(q, k, v, topk, sm_scale, o, helper)

    T, QH, D = q.shape
    T2, KH, D2 = k.shape
    T3, KH2, VD = v.shape
    G = QH // KH
    assert T2 == T3 and KH == KH2 and D == D2
    assert QH % KH == 0
    assert math.log2(VD).is_integer()
    D1, D2 = NSAHelper.split_d(D)
    if sm_scale is None:
        sm_scale = D**-0.5

    block_size, top_n = NSAHelper.block_size, NSAHelper.top_n
    x_cu_seqlens, x_maxlen = helper.x_cu_seqlens, helper.x_maxlen
    cp_cu_seqlens, cp_maxlen = helper.cp_cu_seqlens, helper.cp_maxlen
    cp_batch_idx, cp_offset = helper.cp_batch_idx, helper.cp_offset
    CP = cp_cu_seqlens is not None and NSAHelper.is_context_parallel_enable
    S = x_maxlen if not CP else cp_maxlen
    B = len(x_cu_seqlens)-1 if not CP else len(cp_batch_idx)

    if o is None:
        o = torch.empty(T, QH, VD, device=q.device, dtype=q.dtype)
    lse = torch.empty(T, QH, dtype=torch.float32, device=q.device,)

    BLOCK_H = max(triton.next_power_of_2(G), 16)
    BLOCK_M = block_size
    desc_q = TensorDescriptor.from_tensor(q.view(T, KH, G, D), (1, 1, BLOCK_H, D1))
    desc_q2 = TensorDescriptor.from_tensor(q.view(T, KH, G, D), (1, 1, BLOCK_H, D2)) if D2 > 0 else None
    desc_k = TensorDescriptor.from_tensor(k, (BLOCK_M, 1, D1))
    desc_k2 = TensorDescriptor.from_tensor(k, (BLOCK_M, 1, D2)) if D2 > 0 else None
    desc_v = TensorDescriptor.from_tensor(v, (BLOCK_M, 1, VD))
    desc_o = TensorDescriptor.from_tensor(o.view(T, KH, G, VD), (1, 1, BLOCK_H, VD))
    if D <= 128:
        kwargs = {"num_warps": 1, "num_stages": 2}
    else:
        kwargs = {"num_warps": 1, "num_stages": 1}
    grid = lambda meta: (S, B * KH)
    _fwd_kernel[grid](
        desc_q, 
        desc_q2, 
        desc_k, 
        desc_k2, 
        desc_v, 
        desc_o, 
        lse, 
        topk,
        x_cu_seqlens,
        CP,
        cp_cu_seqlens,
        cp_batch_idx,
        cp_offset,
        *lse.stride(),
        *topk.stride(),
        sm_scale,
        top_n,
        KH,
        G,
        D1, 
        D2,
        VD,
        BLOCK_H=BLOCK_H,
        BLOCK_M=BLOCK_M,
        **kwargs
    )
    return o, lse

@use_tma
def slc_attn_bwd(
    q, 
    k, 
    v, 
    topk, 
    o, 
    lse, 
    do, 
    dq=None, 
    dk=None, 
    dv=None, 
    sm_scale=None, 
    fuse_dqdkdv=False, 
    dkdv_dtype=torch.float, 
    dkdv_repeat=True, 
    async_dq=False, 
    helper=NSAHelper
):
    '''
    dkdv_repeat=False is non-deterministic
    if dkdv is provided, use atomic_add on dkdv.
    '''
    if NSAHelper.is_use_ampere_ops():
        return ampere_ops.slc_attn_bwd(q, k, v, topk, o, lse, do, dq, dk, dv, sm_scale, fuse_dqdkdv, dkdv_dtype, dkdv_repeat, async_dq, helper)
    T, QH, D = q.shape
    T2, KH, D2 = k.shape
    T3, KH2, VD = v.shape
    G = QH // KH
    assert T2 == T3 and KH == KH2 and D == D2
    assert QH % KH == 0
    assert math.log2(VD).is_integer()
    D1, D2 = NSAHelper.split_d(D)
    if sm_scale is None:
        sm_scale = D ** -0.5

    top_n = topk.size(-1)
    block_size = NSAHelper.block_size
    BLOCK_M = block_size
    BLOCK_H = max(triton.next_power_of_2(G), 16)

    x_cu_seqlens, k_cu_seqlens, x_maxlen = helper.x_cu_seqlens, helper.k_cu_seqlens, helper.x_maxlen
    cp_cu_seqlens, cp_maxlen = helper.cp_cu_seqlens, helper.cp_maxlen
    cp_batch_idx, cp_offset = helper.cp_batch_idx, helper.cp_offset
    cp_k_cu_seqlens, cp_k_maxlen = helper.cp_k_cu_seqlens, helper.cp_k_maxlen
    CP = cp_cu_seqlens is not None and NSAHelper.is_context_parallel_enable
    S = x_maxlen if not CP else cp_maxlen
    B = len(x_cu_seqlens)-1 if not CP else len(cp_batch_idx)

    delta = torch.empty_like(lse)
    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_N"]), QH)
    kwargs = {"BLOCK_N": 64, "num_warps": 8, "num_stages": 4}
    _bwd_preprocess[grid](
        o, 
        do, 
        delta,
        *o.stride(), 
        *do.stride(),
        *delta.stride(),
        T, 
        VD,
        **kwargs
    )

    bwd_ind, count = get_bind_from_find(topk, helper=helper)

    if dk is None:
        dkdv_dtype = dkdv_dtype if G > 1 else k.dtype
        DKDVH = QH if dkdv_repeat else KH
        dk = torch.zeros(T, DKDVH, D, device=q.device, dtype=k.dtype if G == 1 else torch.float)
        dv = torch.zeros(T, DKDVH, VD, device=q.device, dtype=v.dtype if G == 1 else torch.float)
        cast_to_kv = True
        dkdv_atomic = not dkdv_repeat or helper.atomic_add_dkdv
    else:
        dkdv_repeat = KH != dk.size(1)
        cast_to_kv = False
        dkdv_atomic = True

    if dq is None:
        dq = torch.zeros_like(q)
        dq_atomic = False
    else:
        dq_atomic  = True

    if D<= 128:
        kwargs = {"BLOCK_N":64, "num_warps": 4, "num_stages": 2}
    else:
        kwargs = {"BLOCK_N":64, "num_warps": 4, "num_stages": 1}
    grid = (B, QH, triton.cdiv(x_maxlen, BLOCK_M))
    _dkdv_kernel[grid](
        dq, 
        dk, 
        dv, 
        do, 
        q, 
        k, 
        v, 
        lse, 
        delta, 
        x_cu_seqlens,
        k_cu_seqlens,
        CP,
        cp_cu_seqlens,
        cp_batch_idx,
        cp_offset,
        cp_k_cu_seqlens,
        bwd_ind, 
        count,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *dq.stride(),
        *dk.stride(),
        *dv.stride(),
        *do.stride(), 
        *lse.stride(),
        *bwd_ind.stride(),
        *count.stride(),
        sm_scale, 
        dkdv_atomic, 
        dkdv_repeat, 
        fuse_dqdkdv,
        G, 
        D1, 
        D2, 
        VD,
        BLOCK_M,
        **kwargs
        )

    if dkdv_repeat:
        dk = dk.view(T2, KH, G, D).sum(2)
        dv = dv.view(T2, KH, G, VD).sum(2)
    if cast_to_kv:
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)

    if fuse_dqdkdv:
        return dq, dk, dv


    desc_q = TensorDescriptor.from_tensor(q.view(T, KH, G, D), (1, 1, BLOCK_H, D1))
    desc_q2 = TensorDescriptor.from_tensor(q.view(T, KH, G, D), (1, 1, BLOCK_H, D2)) if D2 > 0 else None
    desc_k = TensorDescriptor.from_tensor(k, (BLOCK_M, 1, D1))
    desc_k2 = TensorDescriptor.from_tensor(k, (BLOCK_M, 1, D2)) if D2 > 0 else None
    desc_v = TensorDescriptor.from_tensor(v, (BLOCK_M, 1, VD))
    desc_do = TensorDescriptor.from_tensor(do.view(T, KH, G, VD), (1, 1, BLOCK_H, VD))
    desc_dq = TensorDescriptor.from_tensor(dq.view(T, KH, G, D), (1, 1, BLOCK_H, D1))
    desc_dq2 = TensorDescriptor.from_tensor(dq.view(T, KH, G, D), (1, 1, BLOCK_H, D2)) if D2 > 0 else None

    if D <= 128:
        kwargs = {"num_warps": 2, "num_stages": 3}
    else:
        kwargs = {"num_warps": 2, "num_stages": 2}
    grid = lambda meta: (S, B * KH)
    def func():
        _dq_kernel[grid](
            desc_q, 
            desc_q2, 
            desc_k, 
            desc_k2, 
            desc_v, 
            desc_do, 
            desc_dq, 
            desc_dq2,
            lse, 
            delta, 
            topk,
            x_cu_seqlens,
            CP,
            cp_cu_seqlens,
            cp_batch_idx,
            cp_offset,
            *lse.stride(),
            *topk.stride(),
            sm_scale, 
            top_n, 
            dq_atomic,
            KH, 
            G,  
            D1, 
            D2, 
            VD,
            BLOCK_H, 
            BLOCK_M,
            **kwargs
        )

    if not async_dq:
        func()
        return dq, dk, dv
    else:
        return dq, dk, dv, func


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk, sm_scale):
        if sm_scale is None:
            sm_scale = q.size(-1) ** -0.5
        o, lse = slc_attn_fwd(q, k, v, topk, sm_scale)
        ctx.save_for_backward(q, k, v, o, lse, topk)
        ctx.sm_smcale = sm_scale
        ctx.helper = NSAHelper.get_bwd_helper()
        return o

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, o, lse, topk = ctx.saved_tensors
        dq, dk, dv = slc_attn_bwd(q, k, v, topk, o, lse, do, dkdv_repeat=False, helper=ctx.helper, sm_scale=ctx.sm_smcale)
        return dq, dk, dv, None, None


def select_attn(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    topk: torch.Tensor, 
    sm_scale: float=None
) -> torch.Tensor:
    """
    select attention

    Args:
        q (torch.Tensor): [t, num_q_head, qk_head_dim]
        k (torch.Tensor): [t, num_kv_head, qk_head_dim]
        v (torch.Tensor): [t, num_kv_head, v_head_dim]
        topk: (torch.Tensor): [num_kv_head, t, top_n]
        sm_scale (float): softmax_scale
    Return:
        o (torch.Tensor): [t, num_q_head, v_head_dim]
    """
    return _attention.apply(q, k, v, topk, sm_scale)







