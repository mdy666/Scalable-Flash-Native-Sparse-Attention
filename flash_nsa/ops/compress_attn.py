# Copyright (c) 2025 Duyue Ma

import math

import torch
import triton
import triton.language as tl

from ..utils import NSAHelper, use_tma

# @triton.autotune([triton.Config({'BLOCK_N': bsn, 'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['D1', "D2", "VD"])
@triton.jit
def _fwd_kernel(
    Q, 
    K, 
    V, 
    O, 
    LSE, 
    X_CU_SEQLENS, 
    Y_CU_SEQLENS,
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    v_stride_m, v_stride_h, v_stride_d,
    o_stride_n, o_stride_h, o_stride_d,
    lse_stride_h, lse_stride_n,
    sm_scale, 
    kernel_size, 
    stride,
    G, 
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_N: tl.constexpr=64, 
    BLOCK_M: tl.constexpr=128
):

    cp_start_n = tl.program_id(0) * BLOCK_N
    cp_off_b = tl.program_id(1)
    off_qh = tl.program_id(2)
    off_kh = off_qh // G
    if not CP:
        off_b = cp_off_b
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if cp_start_n >= x_len:
            return
        cp_bos = x_bos
        cp_len = x_len
        cp_offset = 0
        start_n = cp_start_n
    else:
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if cp_start_n >= cp_len:
            return
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        start_n = cp_start_n + cp_offset

    y_bos, y_eos = tl.load(Y_CU_SEQLENS + off_b), tl.load(Y_CU_SEQLENS + off_b + 1)
    y_len = y_eos - y_bos

    Q += cp_bos * q_stride_n + off_qh * q_stride_h
    K += y_bos * k_stride_m + off_kh * k_stride_h
    V += y_bos * v_stride_m + off_kh * v_stride_h
    O += cp_bos * o_stride_n + off_qh * o_stride_h
    LSE += off_qh * lse_stride_h + cp_bos * lse_stride_n

    cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
    q_idx = cp_idx + cp_offset

    desc_q = tl.make_tensor_descriptor(Q, (cp_len, D1), (q_stride_n, q_stride_d), (BLOCK_N, D1))
    desc_o = tl.make_tensor_descriptor(O, (cp_len, D1), (o_stride_n, o_stride_d), (BLOCK_N, VD))
    desc_k = tl.make_tensor_descriptor(K, (y_len, D1), (k_stride_m, k_stride_d), (BLOCK_M, D1))
    desc_v = tl.make_tensor_descriptor(V, (y_len, VD), (v_stride_m, v_stride_d), (BLOCK_M, VD))
    if D2 > 0:
        desc_q2 = tl.make_tensor_descriptor(Q + D1, (cp_len, D2), (q_stride_n, q_stride_d), (BLOCK_N, D2))
        desc_k2 = tl.make_tensor_descriptor(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (BLOCK_M, D2))
    q = desc_q.load([cp_start_n, 0])
    if D2 > 0:
        q2 = desc_q2.load([cp_start_n, 0])

    sm_scale *= 1.44269504
    m_i = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_N, VD], dtype=tl.float32)

    mid = tl.minimum((start_n - kernel_size) // stride + 1, y_len).to(tl.int32)
    mid = (mid // BLOCK_M) * BLOCK_M
    end = tl.minimum((start_n + BLOCK_N - kernel_size) // stride + 1, y_len).to(tl.int32)
    for start_block_kv_idx in range(0, mid, BLOCK_M):
        k = desc_k.load([start_block_kv_idx, 0])
        v = desc_v.load([start_block_kv_idx, 0])
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            k2 = desc_k2.load([start_block_kv_idx, 0])
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

        attn_score *= sm_scale

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - new_m_i)

        exp_attn_score = tl.exp2(attn_score - new_m_i[:, None])

        l_i = tl.fma(l_i, alpha, tl.sum(exp_attn_score, axis=-1))
        acc = acc * alpha[:, None]

        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)
        m_i = new_m_i

    for start_block_kv_idx in range(mid, end, BLOCK_M):
        k = desc_k.load([start_block_kv_idx, 0])
        v = desc_v.load([start_block_kv_idx, 0])
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            k2 = desc_k2.load([start_block_kv_idx, 0])
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

        k_idx = (start_block_kv_idx + tl.arange(0, BLOCK_M)) * stride + kernel_size - 1
        attn_score = tl.where(q_idx[:, None] >= k_idx[None, :], attn_score * sm_scale, float('-inf'))

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - new_m_i)

        exp_attn_score = tl.exp2(attn_score - new_m_i[:, None])

        l_i = tl.fma(l_i, alpha, tl.sum(exp_attn_score, axis=-1))
        acc = acc * alpha[:, None]

        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)
        m_i = new_m_i

    acc /= l_i[:, None]
    lse = m_i + tl.log2(l_i)
    if cp_start_n == 0:
        acc = tl.where(q_idx[:, None]>=(kernel_size-1), acc, 0)
        lse = tl.where(q_idx>=(kernel_size-1), lse, 0)
    desc_o.store([cp_start_n, 0], acc.to(desc_o.dtype)) 
    tl.store(LSE + cp_idx * lse_stride_n, lse, mask=cp_idx < cp_len)

# @triton.autotune([triton.Config({'BLOCK_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [16, 32, 64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['VD'])
@triton.jit
def _bwd_preprocess(
    O, 
    DO, 
    Delta,
    o_stride_n, o_stride_h, o_stride_d,
    do_stride_n, do_stride_h, do_stride_d,
    delta_stride_h, delta_stride_n,
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
    tl.store(Delta + off_n * delta_stride_n, delta, mask=off_n < T)


# @triton.autotune([triton.Config({'BLOCK_N': bsn, 'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=["D1", "D2", "VD"])
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
    Y_CU_SEQLENS,
    CP:tl.constexpr, 
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    v_stride_m, v_stride_h, v_stride_d,
    dq_stride_n, dq_stride_h, dq_stride_d,
    dk_stride_m, dk_stride_h, dk_stride_d,
    dv_stride_m, dv_stride_h, dv_stride_d,
    do_stride_n, do_stride_h, do_stride_d,
    lse_stride_h, lse_stride_n,
    sm_scale, 
    kernel_size, 
    stride,
    ATOMIC: tl.constexpr, 
    REPEAT_DKDV: tl.constexpr, 
    COMPUTE_DQ: tl.constexpr,
    G, 
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_N: tl.constexpr=64, 
    BLOCK_M: tl.constexpr=64
):
    start_m = tl.program_id(1) * BLOCK_M
    cp_off_b = tl.program_id(2)
    off_qh = tl.program_id(0)
    off_kh = off_qh // G

    if not CP:
        off_b = cp_off_b
        y_bos, y_eos = tl.load(Y_CU_SEQLENS + off_b), tl.load(Y_CU_SEQLENS + off_b + 1)
        y_len = y_eos - y_bos
        if start_m >= y_len:
            return
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        cp_bos = x_bos
        cp_len = x_len
        cp_offset = 0
    else:
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        y_bos, y_eos = tl.load(Y_CU_SEQLENS + off_b), tl.load(Y_CU_SEQLENS + off_b + 1)
        y_len = y_eos - y_bos
        if start_m >= y_len:
            return
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        if (start_m * stride + kernel_size - 1) >= (cp_offset + cp_len):
            return

    if REPEAT_DKDV:
        off_dkdvh = off_qh
    else:
        off_dkdvh = off_kh
    if COMPUTE_DQ:
        DQ += cp_bos * dq_stride_n + off_qh * dq_stride_h
    Q += cp_bos * q_stride_n + off_qh * q_stride_h
    K += y_bos * k_stride_m + off_kh * k_stride_h 
    V += y_bos * v_stride_m + off_kh * v_stride_h
    DK += y_bos * dk_stride_m + off_dkdvh * dk_stride_h 
    DV += y_bos * dv_stride_m + off_dkdvh * dv_stride_h
    DO += cp_bos * do_stride_n + off_qh * do_stride_h
    Lse += off_qh * lse_stride_h + cp_bos * lse_stride_n
    Delta += off_qh * lse_stride_h + cp_bos * lse_stride_n

    off_m = start_m + tl.arange(0, BLOCK_M)
    if COMPUTE_DQ:
        desc_dq = tl.make_tensor_descriptor(DQ, (cp_len, D1), (dq_stride_n, dq_stride_d), (BLOCK_N, D1))
    desc_q = tl.make_tensor_descriptor(Q, (cp_len, D1), (q_stride_n, q_stride_d), (BLOCK_N, D1))
    desc_k = tl.make_tensor_descriptor(K, (y_len, D1), (k_stride_m, k_stride_d), (BLOCK_M, D1))
    desc_v = tl.make_tensor_descriptor(V, (y_len, VD), (v_stride_m, v_stride_d), (BLOCK_M, VD))
    desc_do = tl.make_tensor_descriptor(DO, (cp_len, VD), (do_stride_n, do_stride_d), (BLOCK_N, VD))
    desc_dk = tl.make_tensor_descriptor(DK, (y_len, D1), (dk_stride_m, dk_stride_d), (BLOCK_M, D1))
    desc_dv = tl.make_tensor_descriptor(DV, (y_len, VD), (dv_stride_m, dv_stride_d), (BLOCK_M, VD))
    if D2 > 0:
        if COMPUTE_DQ:
            desc_dq2 = tl.make_tensor_descriptor(DQ + D1, (cp_len, D2), (dq_stride_n, dq_stride_d), (BLOCK_N, D2))
        desc_q2 = tl.make_tensor_descriptor(Q + D1, (cp_len, D2), (q_stride_n, q_stride_d), (BLOCK_N, D2))
        desc_k2 = tl.make_tensor_descriptor(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (BLOCK_M, D2))
        desc_dk2 = tl.make_tensor_descriptor(DK + D1, (y_len, D2), (dk_stride_m, dk_stride_d), (BLOCK_M, D2))

    k = desc_k.load([start_m, 0])
    v = desc_v.load([start_m, 0])
    acc_dk = tl.zeros((BLOCK_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_M, VD), dtype=tl.float32)

    if D2 > 0:
        k2 = desc_k2.load([start_m, 0])
        acc_dk2 = tl.zeros((BLOCK_M, D2), dtype=tl.float32)

    sm_scale_ln2 = sm_scale * 1.44269504
    k_idx = off_m * stride + kernel_size - 1
    begin = tl.maximum(start_m * stride + kernel_size - 1 - cp_offset, 0)
    mid = tl.minimum(begin + tl.cdiv((BLOCK_M-1) * stride, BLOCK_N) * BLOCK_N, cp_len)
    for cp_start_n in range(begin, cp_len, BLOCK_N):
        cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
        q_idx = cp_idx + cp_offset
        q = desc_q.load([cp_start_n, 0])
        do = desc_do.load([cp_start_n, 0])
        lse = tl.load(Lse + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)
        delta = tl.load(Delta + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)

        attn_score = tl.dot(q, tl.permute(k, 1, 0)) 
        if D2 > 0:
            q2 = desc_q2.load([cp_start_n, 0])
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        attn_score = tl.where(q_idx[:, None] >= k_idx[None, :], attn_score, float('-inf'))
        p = tl.exp2(attn_score * sm_scale_ln2 -lse[:, None])

        acc_dv = tl.dot(tl.trans(p, 1, 0).to(do.dtype), do, acc_dv)

        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])

        acc_dk = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q, acc_dk)
        if D2 > 0:
            acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)

        if COMPUTE_DQ:
            dq = tl.dot(ds.to(k.dtype), k) * sm_scale
            desc_dq.atomic_add([cp_start_n, 0], dq.to(desc_dq.dtype))
            if D2 > 0:
                dq2 = tl.dot(ds.to(k.dtype), k2) * sm_scale
                desc_dq2.atomic_add([cp_start_n, 0], dq2.to(desc_dq.dtype))

    # for cp_start_n in range(mid, cp_len, BLOCK_N):
    #     cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
    #     # q_idx = cp_idx + cp_offset
    #     q = desc_q.load([cp_start_n, 0])
    #     do = desc_do.load([cp_start_n, 0])
    #     lse = tl.load(Lse + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)
    #     delta = tl.load(Delta + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)

    #     attn_score = tl.dot(q, tl.permute(k, 1, 0)) 

    #     if D2 > 0:
    #         q2 = desc_q2.load([cp_start_n, 0])
    #         attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

    #     # attn_score = tl.where(q_idx[:, None] >= k_idx[None, :], attn_score, float('-inf'))
    #     p = tl.exp2(attn_score * sm_scale_ln2 -lse[:, None])

    #     acc_dv = tl.dot(tl.trans(p, 1, 0).to(do.dtype), do, acc_dv)

    #     dp = tl.dot(do, tl.permute(v, 1, 0))
    #     ds = p * (dp - delta[:, None])

    #     acc_dk = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q, acc_dk)
    #     if D2 > 0:
    #         acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)

    #     if COMPUTE_DQ:
    #         dq = tl.dot(ds.to(k.dtype), k) * sm_scale
    #         desc_dq.atomic_add([cp_start_n, 0], dq.to(desc_dq.dtype))
    #         if D2 > 0:
    #             dq2 = tl.dot(ds.to(k.dtype), k2) * sm_scale
    #             desc_dq2.atomic_add([cp_start_n, 0], dq2.to(desc_dq.dtype))

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


# @triton.autotune([triton.Config({'BLOCK_N': bsn, 'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=["D1", "D2", "VD", "ATOMIC"])
@triton.jit
def _dq_kernel(
    DQ, 
    DO, 
    Q, 
    K, 
    V, 
    Lse,
    Delta,
    X_CU_SEQLENS, 
    Y_CU_SEQLENS,
    CP:tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    v_stride_m, v_stride_h, v_stride_d,
    dq_stride_n, dq_stride_h, dq_stride_d,
    do_stride_n, do_stride_h, do_stride_d,
    lse_stride_h, lse_stride_n,
    sm_scale,  
    kernel_size, 
    stride,
    ATOMIC: tl.constexpr,
    G, 
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_N: tl.constexpr=64,
    BLOCK_M: tl.constexpr=64
):

    cp_start_n = tl.program_id(0) * BLOCK_N
    cp_off_b = tl.program_id(1)
    off_qh = tl.program_id(2)
    off_kh = off_qh // G
    if not CP:
        off_b = cp_off_b
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if cp_start_n >= x_len:
            return
        cp_bos = x_bos
        cp_len = x_len
        cp_offset = 0
        start_n = cp_start_n
    else:
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if cp_start_n >= cp_len:
            return
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        start_n = cp_start_n + cp_offset

    y_bos, y_eos = tl.load(Y_CU_SEQLENS + off_b), tl.load(Y_CU_SEQLENS + off_b + 1)
    y_len = y_eos - y_bos

    Q += cp_bos * q_stride_n + off_qh * q_stride_h
    K += y_bos * k_stride_m + off_kh * k_stride_h 
    V += y_bos * v_stride_m + off_kh * v_stride_h
    DQ += cp_bos * dq_stride_n + off_qh * dq_stride_h
    DO += cp_bos * do_stride_n + off_qh * do_stride_h
    Lse += off_qh * lse_stride_h + cp_bos * lse_stride_n
    Delta += off_qh * lse_stride_h + cp_bos * lse_stride_n

    cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
    q_idx =  cp_idx + cp_offset
    lse = tl.load(Lse + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)
    delta = tl.load(Delta + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)

    desc_q = tl.make_tensor_descriptor(Q, (cp_len, D1), (q_stride_n, q_stride_d), (BLOCK_N, D1))
    desc_k = tl.make_tensor_descriptor(K, (y_len, D1), (k_stride_m, k_stride_d), (BLOCK_M, D1))
    desc_v = tl.make_tensor_descriptor(V, (y_len, VD), (v_stride_m, v_stride_d), (BLOCK_M, VD))
    desc_do = tl.make_tensor_descriptor(DO, (cp_len, D1), (do_stride_n, do_stride_d), (BLOCK_N, VD))
    desc_dq = tl.make_tensor_descriptor(DQ, (cp_len, D1), (dq_stride_n, dq_stride_d), (BLOCK_N, D1))
    if D2 > 0:
        desc_q2 = tl.make_tensor_descriptor(Q + D1, (cp_len, D2), (q_stride_n, q_stride_d), (BLOCK_N, D2))
        desc_k2 = tl.make_tensor_descriptor(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (BLOCK_M, D2))
        desc_dq2 = tl.make_tensor_descriptor(DQ + D1, (cp_len, D2), (dq_stride_n, dq_stride_d), (BLOCK_N, D2))

    q = desc_q.load([cp_start_n, 0])
    do = desc_do.load([cp_start_n, 0])
    if D2 > 0:
        q2 = desc_q2.load([cp_start_n, 0])

    acc_dq = tl.zeros((BLOCK_N, D1), dtype=tl.float32)
    if D2 > 0:
        acc_dq2 = tl.zeros((BLOCK_N, D2), dtype=tl.float32)

    sm_scale_ln2 = sm_scale * 1.44269504
    mid = tl.minimum((start_n - kernel_size) // stride + 1, y_len).to(tl.int32)
    mid = (mid // BLOCK_M) * BLOCK_M
    end = tl.minimum((start_n + BLOCK_N - kernel_size) // stride + 1, y_len).to(tl.int32)
    for start_m in range(0, mid, BLOCK_M):
        # block_idx = start_block_kv_idx + tl.arange(0, BLOCK_M)
        k = desc_k.load([start_m, 0])
        v = desc_v.load([start_m, 0])
        attn_score = tl.dot(q, tl.permute(k, 1, 0)) 
        if D2 > 0:
            k2 = desc_k2.load([start_m, 0])
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

        # k_idx = block_idx * stride + kernel_size - 1
        # attn_score = tl.where(q_idx[:, None] >= k_idx[None, :], attn_score, float('-inf'))
        p = tl.exp2(attn_score * sm_scale_ln2 - lse[:, None])

        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])

        acc_dq = tl.dot(ds.to(k.dtype), k, acc_dq)
        if D2 > 0:
            acc_dq2 = tl.dot(ds.to(k.dtype), k2, acc_dq2)

    for start_m in range(mid, end, BLOCK_M):
        k = desc_k.load([start_m, 0])
        v = desc_v.load([start_m, 0])
        attn_score = tl.dot(q, tl.permute(k, 1, 0)) 
        if D2 > 0:
            k2 = desc_k2.load([start_m, 0])
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

        k_idx = (start_m + tl.arange(0, BLOCK_M)) * stride + kernel_size - 1
        attn_score = tl.where(q_idx[:, None] >= k_idx[None, :], attn_score * sm_scale_ln2, float('-inf'))
        p = tl.exp2(attn_score - lse[:, None])

        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])

        acc_dq = tl.dot(ds.to(k.dtype), k, acc_dq)
        if D2 > 0:
            acc_dq2 = tl.dot(ds.to(k.dtype), k2, acc_dq2)

    if not ATOMIC:
        desc_dq.store([cp_start_n, 0], (acc_dq * sm_scale).to(desc_dq.dtype))
        if D2 > 0:
            desc_dq2.store([cp_start_n, 0], (acc_dq2 * sm_scale).to(desc_dq2.dtype))
    else:
        desc_dq.atomic_add([cp_start_n, 0], (acc_dq * sm_scale).to(desc_dq.dtype))
        if D2 > 0:
            desc_dq2.atomic_add([cp_start_n, 0], (acc_dq2 * sm_scale).to(desc_dq2.dtype))

@use_tma
def cmp_attn_fwd(q, k, v, sm_scale=None, o=None, helper=NSAHelper):
    T, QH, D = q.shape
    T2, KH, D2 = k.shape
    T3, KH2, VD = v.shape
    kernel_size, stride = NSAHelper.kernel_size, NSAHelper.stride

    assert D == D2 and KH == KH2 and T2 == T3
    assert QH % KH == 0
    assert math.log2(VD).is_integer()
    D1, D2 = NSAHelper.split_d(D)
    G = QH // KH

    if sm_scale is None:
        sm_scale = D**-0.5

    x_cu_seqlens, y_cu_seqlens, x_maxlen = helper.x_cu_seqlens, helper.y_cu_seqlens, helper.x_maxlen
    cp_cu_seqlens, cp_maxlen = helper.cp_cu_seqlens, helper.cp_maxlen
    cp_bacth_idx, cp_offset = helper.cp_batch_idx, helper.cp_offset
    CP = cp_cu_seqlens is not None and NSAHelper.is_context_parallel_enable
    S = x_maxlen if not CP else cp_maxlen
    B = len(x_cu_seqlens) - 1 if not CP else len(cp_cu_seqlens) - 1

    if o is None:
        o = torch.empty(T, QH, VD, device=q.device, dtype=q.dtype)
    lse = torch.empty(QH, T, dtype=torch.float32, device=q.device,)

    if D <= 128:
        kwargs = {"BLOCK_N": 128, "BLOCK_M": 64, "num_warps": 4, "num_stages": 2}
    else:
        kwargs = {"BLOCK_N": 128, "BLOCK_M": 64, "num_warps": 4, "num_stages": 1}

    grid = lambda meta: (triton.cdiv(S, meta['BLOCK_N']), B, QH)
    func = _fwd_kernel if NSAHelper.use_tma_kernel() else _fwd_kernel_no_tma
    func[grid](
        q, 
        k, 
        v, 
        o, 
        lse,
        x_cu_seqlens, 
        y_cu_seqlens,
        CP,
        cp_cu_seqlens,
        cp_bacth_idx,
        cp_offset,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        *lse.stride(),
        sm_scale, 
        kernel_size, 
        stride, 
        G, 
        D1, 
        D2, 
        VD,
        **kwargs
    )
    return o, lse

@use_tma
def cmp_attn_bwd(q, k, v, o, lse, do, dq=None, sm_scale=None, fuse_dqdkdv=False, dkdv_dtype=torch.float, dkdv_repeat=False, async_dq=False, helper=NSAHelper):
    '''
    dkdv_repeat=False is non-deterministic
    '''

    T, QH, D = q.shape
    T2, KH, D2 = k.shape
    T3, KH2, VD = v.shape
    kernel_size, stride = NSAHelper.kernel_size, NSAHelper.stride

    assert D == D2 and KH == KH2 and T2 == T3
    assert QH % KH == 0
    assert math.log2(VD).is_integer()
    D1, D2 = NSAHelper.split_d(D)
    G = QH // KH

    if sm_scale is None:
        sm_scale = D**-0.5 

    x_cu_seqlens, y_cu_seqlens = helper.x_cu_seqlens, helper.y_cu_seqlens
    x_maxlen, y_maxlen = helper.x_maxlen, helper.y_maxlen
    cp_cu_seqlens, cp_maxlen = helper.cp_cu_seqlens, helper.cp_maxlen
    cp_bacth_idx, cp_offset = helper.cp_batch_idx, helper.cp_offset
    CP = cp_cu_seqlens is not None and NSAHelper.is_context_parallel_enable
    S = x_maxlen if not CP else cp_maxlen
    B = len(x_cu_seqlens) - 1 if not CP else len(cp_bacth_idx)

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

    dkdv_dtype = dkdv_dtype if G > 1 else k.dtype
    dkdv_atomic = not dkdv_repeat or helper.atomic_add_dkdv
    DKDVH = QH if dkdv_repeat else KH

    dk = torch.zeros(T2, DKDVH, D1+D2, device=q.device, dtype=dkdv_dtype)
    dv = torch.zeros(T2, DKDVH, VD, device=q.device, dtype=dkdv_dtype)
    if dq is None:
        dq = torch.zeros_like(q)
        dq_atomic = False
    else:
        dq_atomic  = True

    if D <= 128:
        kwargs = {"BLOCK_N": 64, "BLOCK_M": 64, "num_warps": 4, "num_stages": 2}
    else:
        kwargs = {"BLOCK_N": 64, "BLOCK_M": 64, "num_warps": 4, "num_stages": 3}
    grid = lambda meta: (QH, triton.cdiv(y_maxlen, meta["BLOCK_M"]), B)
    func = _dkdv_kernel if NSAHelper.use_tma_kernel() else _dkdv_kernel_no_tma
    func[grid](
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
        y_cu_seqlens,
        CP,
        cp_cu_seqlens,
        cp_bacth_idx,
        cp_offset,
        *q.stride(), 
        *k.stride(),
        *v.stride(),
        *dq.stride(),
        *dk.stride(),
        *dv.stride(),
        *do.stride(),
        *lse.stride(),
        sm_scale, 
        kernel_size, 
        stride, 
        dkdv_atomic, 
        dkdv_repeat, 
        fuse_dqdkdv,
        G, 
        D1, 
        D2, 
        VD,
        **kwargs
    )
    if dkdv_repeat:
        dk = dk.view(T2, KH, G, D).sum(2)
        dv = dv.view(T2, KH, G, VD).sum(2)
    if dk.dtype != k.dtype:
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)

    if fuse_dqdkdv:
        return dq, dk, dv

    if D <= 128:
        kwargs = {"BLOCK_N": 128, "BLOCK_M": 32, "num_warps": 8, "num_stages": 3}
    else:
        kwargs = {"BLOCK_N": 64, "BLOCK_M": 32, "num_warps": 4, "num_stages": 3}
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_N"]), B, QH)
    def func():
        dq_func = _dq_kernel if NSAHelper.use_tma_kernel() else _dq_kernel_no_tma
        dq_func[grid](
            dq, 
            do, 
            q, 
            k, 
            v,
            lse, 
            delta,
            x_cu_seqlens,
            y_cu_seqlens,
            CP,
            cp_cu_seqlens,
            cp_bacth_idx,
            cp_offset,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *dq.stride(),
            *do.stride(),
            *lse.stride(),
            sm_scale, 
            kernel_size, 
            stride, 
            dq_atomic,
            G, 
            D1, 
            D2, 
            VD,
            **kwargs
        )   
    if not async_dq:
        func()
        return dq, dk, dv
    else:
        return dq, dk, dv, func

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        o, lse = cmp_attn_fwd(q, k, v, sm_scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.helper = NSAHelper.get_bwd_helper()
        return o, lse

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = cmp_attn_bwd(q, k, v, o, lse, do, dkdv_repeat=False, sm_scale=ctx.sm_scale, helper=ctx.helper)
        return dq, dk, dv, None

def compress_attn(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    sm_scale: float=None
) -> torch.Tensor:
    """
    Compress attention

    Args:
        q (torch.Tensor): [t, num_q_head, qk_head_dim]
        k (torch.Tensor): [total_num_blocks, num_kv_head, qk_head_dim]
        v (torch.Tensor): [total_num_blocks, num_kv_head, v_head_dim]
        sm_scale (float): softmax_scale
    Return:
        o (torch.Tensor): [t, num_q_head, v_head_dim]
    """
    return _attention.apply(q, k, v, sm_scale)


# @triton.autotune([triton.Config({'BLOCK_N': bsn, 'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['D1', "D2", "VD"])
@triton.jit
def _fwd_kernel_no_tma(
    Q, 
    K, 
    V, 
    O, 
    LSE, 
    X_CU_SEQLENS, 
    Y_CU_SEQLENS,
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    v_stride_m, v_stride_h, v_stride_d,
    o_stride_n, o_stride_h, o_stride_d,
    lse_stride_h, lse_stride_n,
    sm_scale, 
    kernel_size, 
    stride,
    G, 
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_N: tl.constexpr=64, 
    BLOCK_M: tl.constexpr=128
):

    cp_start_n = tl.program_id(0) * BLOCK_N
    cp_off_b = tl.program_id(1)
    off_qh = tl.program_id(2)
    off_kh = off_qh // G
    if not CP:
        off_b = cp_off_b
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if cp_start_n >= x_len:
            return
        cp_bos = x_bos
        cp_len = x_len
        cp_offset = 0
        start_n = cp_start_n
    else:
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if cp_start_n >= cp_len:
            return
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        start_n = cp_start_n + cp_offset

    y_bos, y_eos = tl.load(Y_CU_SEQLENS + off_b), tl.load(Y_CU_SEQLENS + off_b + 1)
    y_len = y_eos - y_bos

    Q += cp_bos * q_stride_n + off_qh * q_stride_h
    K += y_bos * k_stride_m + off_kh * k_stride_h
    V += y_bos * v_stride_m + off_kh * v_stride_h
    O += cp_bos * o_stride_n + off_qh * o_stride_h
    LSE += off_qh * lse_stride_h + cp_bos * lse_stride_n

    cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
    q_idx = cp_idx + cp_offset

    desc_q = tl.make_block_ptr(Q, (cp_len, D1), (q_stride_n, q_stride_d), (cp_start_n, 0), (BLOCK_N, D1), order=(1, 0))
    desc_o = tl.make_block_ptr(O, (cp_len, D1), (o_stride_n, o_stride_d), (cp_start_n, 0), (BLOCK_N, VD), order=(1, 0))
    desc_k = tl.make_block_ptr(K, (y_len, D1), (k_stride_m, k_stride_d), (0, 0), (BLOCK_M, D1), order=(1, 0))
    desc_v = tl.make_block_ptr(V, (y_len, VD), (v_stride_m, v_stride_d), (0, 0), (BLOCK_M, VD), order=(1, 0))
    if D2 > 0:
        desc_q2 = tl.make_block_ptr(Q + D1, (cp_len, D2), (q_stride_n, q_stride_d), (cp_start_n, 0), (BLOCK_N, D2), order=(1, 0))
        desc_k2 = tl.make_block_ptr(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (0, 0), (BLOCK_M, D2), order=(1, 0))
    q = tl.load(desc_q, boundary_check=(0, 1))
    if D2 > 0:
        q2 = tl.load(desc_q2, boundary_check=(0, 1))

    sm_scale *= 1.44269504
    m_i = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_N, VD], dtype=tl.float32)

    mid = tl.minimum((start_n - kernel_size) // stride + 1, y_len).to(tl.int32)
    mid = (mid // BLOCK_M) * BLOCK_M
    end = tl.minimum((start_n + BLOCK_N - kernel_size) // stride + 1, y_len).to(tl.int32)
    for start_block_kv_idx in range(0, mid, BLOCK_M):
        k = tl.load(desc_k, boundary_check=(0, 1))
        v = tl.load(desc_v, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            k2 = tl.load(desc_k2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

        attn_score *= sm_scale

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - new_m_i)

        exp_attn_score = tl.exp2(attn_score - new_m_i[:, None])

        l_i = tl.fma(l_i, alpha, tl.sum(exp_attn_score, axis=-1))
        acc = acc * alpha[:, None]

        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)
        m_i = new_m_i

        desc_k = tl.advance(desc_k, (BLOCK_M, 0))
        desc_v = tl.advance(desc_v, (BLOCK_M, 0))
        if D2 > 0:
            desc_k2 = tl.advance(desc_k2, (BLOCK_M, 0))

    for start_block_kv_idx in range(mid, end, BLOCK_M):
        k = tl.load(desc_k, boundary_check=(0, 1))
        v = tl.load(desc_v, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            k2 = tl.load(desc_k2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

        k_idx = (start_block_kv_idx + tl.arange(0, BLOCK_M)) * stride + kernel_size - 1
        attn_score = tl.where(q_idx[:, None] >= k_idx[None, :], attn_score * sm_scale, float('-inf'))

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - new_m_i)

        exp_attn_score = tl.exp2(attn_score - new_m_i[:, None])

        l_i = tl.fma(l_i, alpha, tl.sum(exp_attn_score, axis=-1))
        acc = acc * alpha[:, None]

        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)
        m_i = new_m_i

        desc_k = tl.advance(desc_k, (BLOCK_M, 0))
        desc_v = tl.advance(desc_v, (BLOCK_M, 0))
        if D2 > 0:
            desc_k2 = tl.advance(desc_k2, (BLOCK_M, 0))

    acc /= l_i[:, None]
    lse = m_i + tl.log2(l_i)
    if cp_start_n == 0:
        acc = tl.where(q_idx[:, None]>=(kernel_size-1), acc, 0)
        lse = tl.where(q_idx>=(kernel_size-1), lse, 0)
    tl.store(desc_o, acc.to(desc_o.type.element_ty)) 
    tl.store(LSE + cp_idx * lse_stride_n, lse, mask=cp_idx < cp_len)
    
# @triton.autotune([triton.Config({'BLOCK_N': bsn, 'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=["D1", "D2", "VD"])
@triton.jit
def _dkdv_kernel_no_tma(
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
    Y_CU_SEQLENS,
    CP:tl.constexpr, 
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    v_stride_m, v_stride_h, v_stride_d,
    dq_stride_n, dq_stride_h, dq_stride_d,
    dk_stride_m, dk_stride_h, dk_stride_d,
    dv_stride_m, dv_stride_h, dv_stride_d,
    do_stride_n, do_stride_h, do_stride_d,
    lse_stride_h, lse_stride_n,
    sm_scale, 
    kernel_size, 
    stride,
    ATOMIC: tl.constexpr, 
    REPEAT_DKDV: tl.constexpr, 
    COMPUTE_DQ: tl.constexpr,
    G, 
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_N: tl.constexpr=64, 
    BLOCK_M: tl.constexpr=64
):
    start_m = tl.program_id(1) * BLOCK_M
    cp_off_b = tl.program_id(2)
    off_qh = tl.program_id(0)
    off_kh = off_qh // G

    if not CP:
        off_b = cp_off_b
        y_bos, y_eos = tl.load(Y_CU_SEQLENS + off_b), tl.load(Y_CU_SEQLENS + off_b + 1)
        y_len = y_eos - y_bos
        if start_m >= y_len:
            return
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        cp_bos = x_bos
        cp_len = x_len
        cp_offset = 0
    else:
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        y_bos, y_eos = tl.load(Y_CU_SEQLENS + off_b), tl.load(Y_CU_SEQLENS + off_b + 1)
        y_len = y_eos - y_bos
        if start_m >= y_len:
            return
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        if (start_m * stride + kernel_size - 1) >= (cp_offset + cp_len):
            return

    if REPEAT_DKDV:
        off_dkdvh = off_qh
    else:
        off_dkdvh = off_kh
    if COMPUTE_DQ:
        DQ += cp_bos * dq_stride_n + off_qh * dq_stride_h
    Q += cp_bos * q_stride_n + off_qh * q_stride_h
    K += y_bos * k_stride_m + off_kh * k_stride_h 
    V += y_bos * v_stride_m + off_kh * v_stride_h
    DK += y_bos * dk_stride_m + off_dkdvh * dk_stride_h 
    DV += y_bos * dv_stride_m + off_dkdvh * dv_stride_h
    DO += cp_bos * do_stride_n + off_qh * do_stride_h
    Lse += off_qh * lse_stride_h + cp_bos * lse_stride_n
    Delta += off_qh * lse_stride_h + cp_bos * lse_stride_n

    begin = tl.maximum(start_m * stride + kernel_size - 1 - cp_offset, 0)
    
    off_m = start_m + tl.arange(0, BLOCK_M)
    if COMPUTE_DQ:
        desc_dq = tl.make_block_ptr(DQ, (cp_len, D1), (dq_stride_n, dq_stride_d), (begin, 0), (BLOCK_N, D1), (1, 0))
    desc_q = tl.make_block_ptr(Q, (cp_len, D1), (q_stride_n, q_stride_d), (begin, 0), (BLOCK_N, D1), (1, 0))
    desc_k = tl.make_block_ptr(K, (y_len, D1), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D1), (1, 0))
    desc_v = tl.make_block_ptr(V, (y_len, VD), (v_stride_m, v_stride_d), (start_m, 0), (BLOCK_M, VD), (1, 0))
    desc_do = tl.make_block_ptr(DO, (cp_len, VD), (do_stride_n, do_stride_d), (begin, 0), (BLOCK_N, VD), (1, 0))
    if D2 > 0:
        if COMPUTE_DQ:
            desc_dq2 = tl.make_block_ptr(DQ + D1, (cp_len, D2), (dq_stride_n, dq_stride_d), (begin, 0), (BLOCK_N, D2), (1, 0))
        desc_q2 = tl.make_block_ptr(Q + D1, (cp_len, D2), (q_stride_n, q_stride_d), (begin, 0), (BLOCK_N, D2), (1, 0))
        desc_k2 = tl.make_block_ptr(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D2), (1, 0))

    k = tl.load(desc_k, boundary_check=(0, 1))
    v = tl.load(desc_v, boundary_check=(0, 1))
    acc_dk = tl.zeros((BLOCK_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_M, VD), dtype=tl.float32)

    if D2 > 0:
        k2 = tl.load(desc_k2, boundary_check=(0, 1))
        acc_dk2 = tl.zeros((BLOCK_M, D2), dtype=tl.float32)

    sm_scale_ln2 = sm_scale * 1.44269504
    k_idx = off_m * stride + kernel_size - 1
    
    mid = tl.minimum(begin + tl.cdiv((BLOCK_M-1) * stride, BLOCK_N) * BLOCK_N, cp_len)
    for cp_start_n in range(begin, cp_len, BLOCK_N):
        cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
        q_idx = cp_idx + cp_offset
        q = tl.load(desc_q, boundary_check=(0, 1))
        do = tl.load(desc_do, boundary_check=(0, 1))
        lse = tl.load(Lse + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)
        delta = tl.load(Delta + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)

        attn_score = tl.dot(q, tl.permute(k, 1, 0)) 
        if D2 > 0:
            q2 = tl.load(desc_q2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        attn_score = tl.where(q_idx[:, None] >= k_idx[None, :], attn_score, float('-inf'))
        p = tl.exp2(attn_score * sm_scale_ln2 -lse[:, None])

        acc_dv = tl.dot(tl.trans(p, 1, 0).to(do.dtype), do, acc_dv)

        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])

        acc_dk = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q, acc_dk)
        if D2 > 0:
            acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)

        if COMPUTE_DQ:
            pass
            # dq = tl.dot(ds.to(k.dtype), k) * sm_scale
            # tl.atomic_add(desc_dq, dq.to(desc_dq.type.element_ty))
            # if D2 > 0:
            #     dq2 = tl.dot(ds.to(k.dtype), k2) * sm_scale
            #     tl.atomic_add(desc_dq2, dq2.to(desc_dq2.type.element_ty))
        desc_q = tl.advance(desc_q, (BLOCK_N, 0))
        desc_do = tl.advance(desc_do, (BLOCK_N, 0))
        if D2 > 0:
            desc_q2 = tl.advance(desc_q2, (BLOCK_N, 0))
        
    # for cp_start_n in range(mid, cp_len, BLOCK_N):
    #     cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
    #     # q_idx = cp_idx + cp_offset
    #     q = desc_q.load([cp_start_n, 0])
    #     do = desc_do.load([cp_start_n, 0])
    #     lse = tl.load(Lse + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)
    #     delta = tl.load(Delta + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)

    #     attn_score = tl.dot(q, tl.permute(k, 1, 0)) 

    #     if D2 > 0:
    #         q2 = desc_q2.load([cp_start_n, 0])
    #         attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

    #     # attn_score = tl.where(q_idx[:, None] >= k_idx[None, :], attn_score, float('-inf'))
    #     p = tl.exp2(attn_score * sm_scale_ln2 -lse[:, None])

    #     acc_dv = tl.dot(tl.trans(p, 1, 0).to(do.dtype), do, acc_dv)

    #     dp = tl.dot(do, tl.permute(v, 1, 0))
    #     ds = p * (dp - delta[:, None])

    #     acc_dk = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q, acc_dk)
    #     if D2 > 0:
    #         acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)

    #     if COMPUTE_DQ:
    #         dq = tl.dot(ds.to(k.dtype), k) * sm_scale
    #         desc_dq.atomic_add([cp_start_n, 0], dq.to(desc_dq.dtype))
    #         if D2 > 0:
    #             dq2 = tl.dot(ds.to(k.dtype), k2) * sm_scale
    #             desc_dq2.atomic_add([cp_start_n, 0], dq2.to(desc_dq.dtype))

    if not ATOMIC:
        desc_dk = tl.make_block_ptr(DK, (y_len, D1), (dk_stride_m, dk_stride_d), (start_m, 0), (BLOCK_M, D1), (1, 0))
        desc_dv = tl.make_block_ptr(DV, (y_len, VD), (dv_stride_m, dv_stride_d), (start_m, 0), (BLOCK_M, VD), (1, 0))
        tl.store(desc_dk, (acc_dk * sm_scale).to(desc_dk.type.element_ty), boundary_check=(0, 1))
        tl.store(desc_dv, acc_dv.to(desc_dv.type.element_ty), boundary_check=(0, 1))
        if D2 > 0:
            desc_dk2 = tl.make_block_ptr(DK + D1, (y_len, D2), (dk_stride_m, dk_stride_d), (start_m, 0), (BLOCK_M, D2), (1, 0))
            tl.store(desc_dk2, (acc_dk2 * sm_scale).to(desc_dk2.type.element_ty), boundary_check=(0, 1))
    else:
        mask = off_m < y_len
        tl.atomic_add(DK + off_m[:, None] * dk_stride_m + tl.arange(0, D1)[None, :], (acc_dk * sm_scale).to(DK.type.element_ty), mask=mask[:, None])
        tl.atomic_add(DV + off_m[:, None] * dv_stride_m + tl.arange(0, VD)[None, :], acc_dv.to(DV.type.element_ty), mask=mask[:, None])
        if D2 > 0:
            tl.atomic_add(DK + D1 + off_m[:, None] * dk_stride_m + tl.arange(0, D2)[None, :], (acc_dk2 * sm_scale).to(DK.type.element_ty), mask=mask[:, None])
            
            
# @triton.autotune([triton.Config({'BLOCK_N': bsn, 'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=["D1", "D2", "VD", "ATOMIC"])
@triton.jit
def _dq_kernel_no_tma(
    DQ, 
    DO, 
    Q, 
    K, 
    V, 
    Lse,
    Delta,
    X_CU_SEQLENS, 
    Y_CU_SEQLENS,
    CP:tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    v_stride_m, v_stride_h, v_stride_d,
    dq_stride_n, dq_stride_h, dq_stride_d,
    do_stride_n, do_stride_h, do_stride_d,
    lse_stride_h, lse_stride_n,
    sm_scale,  
    kernel_size, 
    stride,
    ATOMIC: tl.constexpr,
    G, 
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_N: tl.constexpr=64,
    BLOCK_M: tl.constexpr=64
):

    cp_start_n = tl.program_id(0) * BLOCK_N
    cp_off_b = tl.program_id(1)
    off_qh = tl.program_id(2)
    off_kh = off_qh // G
    if not CP:
        off_b = cp_off_b
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if cp_start_n >= x_len:
            return
        cp_bos = x_bos
        cp_len = x_len
        cp_offset = 0
        start_n = cp_start_n
    else:
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if cp_start_n >= cp_len:
            return
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        start_n = cp_start_n + cp_offset

    y_bos, y_eos = tl.load(Y_CU_SEQLENS + off_b), tl.load(Y_CU_SEQLENS + off_b + 1)
    y_len = y_eos - y_bos

    Q += cp_bos * q_stride_n + off_qh * q_stride_h
    K += y_bos * k_stride_m + off_kh * k_stride_h 
    V += y_bos * v_stride_m + off_kh * v_stride_h
    DQ += cp_bos * dq_stride_n + off_qh * dq_stride_h
    DO += cp_bos * do_stride_n + off_qh * do_stride_h
    Lse += off_qh * lse_stride_h + cp_bos * lse_stride_n
    Delta += off_qh * lse_stride_h + cp_bos * lse_stride_n

    cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
    q_idx =  cp_idx + cp_offset
    lse = tl.load(Lse + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)
    delta = tl.load(Delta + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)

    desc_q = tl.make_block_ptr(Q, (cp_len, D1), (q_stride_n, q_stride_d), (cp_start_n, 0), (BLOCK_N, D1), (1, 0))
    desc_k = tl.make_block_ptr(K, (y_len, D1), (k_stride_m, k_stride_d), (0, 0), (BLOCK_M, D1), (1, 0))
    desc_v = tl.make_block_ptr(V, (y_len, VD), (v_stride_m, v_stride_d), (0, 0), (BLOCK_M, VD), (1, 0))
    desc_do = tl.make_block_ptr(DO, (cp_len, D1), (do_stride_n, do_stride_d), (cp_start_n, 0), (BLOCK_N, VD), (1, 0))
    if D2 > 0:
        desc_q2 = tl.make_block_ptr(Q + D1, (cp_len, D2), (q_stride_n, q_stride_d), (cp_start_n, 0), (BLOCK_N, D2), (1, 0))
        desc_k2 = tl.make_block_ptr(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (0, 0), (BLOCK_M, D2), (1, 0))

    q = tl.load(desc_q, boundary_check=(0, 1))
    do = tl.load(desc_do, boundary_check=(0, 1))
    if D2 > 0:
        q2 = tl.load(desc_q2, boundary_check=(0, 1))

    acc_dq = tl.zeros((BLOCK_N, D1), dtype=tl.float32)
    if D2 > 0:
        acc_dq2 = tl.zeros((BLOCK_N, D2), dtype=tl.float32)

    sm_scale_ln2 = sm_scale * 1.44269504
    mid = tl.minimum((start_n - kernel_size) // stride + 1, y_len).to(tl.int32)
    mid = (mid // BLOCK_M) * BLOCK_M
    end = tl.minimum((start_n + BLOCK_N - kernel_size) // stride + 1, y_len).to(tl.int32)
    for start_m in range(0, mid, BLOCK_M):
        # block_idx = start_block_kv_idx + tl.arange(0, BLOCK_M)
        k = tl.load(desc_k, boundary_check=(0, 1))
        v = tl.load(desc_v, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0)) 
        if D2 > 0:
            k2 = tl.load(desc_k2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

        # k_idx = block_idx * stride + kernel_size - 1
        # attn_score = tl.where(q_idx[:, None] >= k_idx[None, :], attn_score, float('-inf'))
        p = tl.exp2(attn_score * sm_scale_ln2 - lse[:, None])

        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])

        acc_dq = tl.dot(ds.to(k.dtype), k, acc_dq)
        if D2 > 0:
            acc_dq2 = tl.dot(ds.to(k.dtype), k2, acc_dq2)

        desc_k = tl.advance(desc_k, (BLOCK_M, 0))
        desc_v = tl.advance(desc_v, (BLOCK_M, 0))
        if D2 > 0:
            desc_k2 = tl.advance(desc_k2, (BLOCK_M, 0))
        

    for start_m in range(mid, end, BLOCK_M):
        k = tl.load(desc_k, boundary_check=(0, 1))
        v = tl.load(desc_v, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0)) 
        if D2 > 0:
            k2 = tl.load(desc_k2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

        k_idx = (start_m + tl.arange(0, BLOCK_M)) * stride + kernel_size - 1
        attn_score = tl.where(q_idx[:, None] >= k_idx[None, :], attn_score * sm_scale_ln2, float('-inf'))
        p = tl.exp2(attn_score - lse[:, None])

        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])

        acc_dq = tl.dot(ds.to(k.dtype), k, acc_dq)
        if D2 > 0:
            acc_dq2 = tl.dot(ds.to(k.dtype), k2, acc_dq2)
            
        desc_k = tl.advance(desc_k, (BLOCK_M, 0))
        desc_v = tl.advance(desc_v, (BLOCK_M, 0))
        if D2 > 0:
            desc_k2 = tl.advance(desc_k2, (BLOCK_M, 0))

    if not ATOMIC:
        desc_dq = tl.make_block_ptr(DQ, (cp_len, D1), (dq_stride_n, dq_stride_d), (cp_start_n, 0), (BLOCK_N, D1), (1, 0))
        tl.store(desc_dq, (acc_dq * sm_scale).to(desc_dq.type.element_ty), boundary_check=(0, 1))
        if D2 > 0:
            desc_dq2 = tl.make_block_ptr(DQ + D1, (cp_len, D2), (dq_stride_n, dq_stride_d), (cp_start_n, 0), (BLOCK_N, D2), (1, 0))
            tl.store(desc_dq2, (acc_dq2 * sm_scale).to(desc_dq.type.element_ty), boundary_check=(0, 1))
    else:
        pass
        mask = cp_idx < cp_len
        tl.atomic_add(DQ + cp_idx[:, None] * dq_stride_n + tl.arange(0, D1)[None, :], (acc_dq * sm_scale).to(DQ.type.element_ty), mask=mask[:, None])
        if D2 > 0:
            tl.atomic_add(DQ + D1 + cp_idx[:, None] * dq_stride_n + tl.arange(0, D2)[None, :], (acc_dq2 * sm_scale).to(DQ.type.element_ty), mask=mask[:, None])

