# Copyright (c) 2025 Duyue Ma

import math

import torch
import triton
import triton.language as tl

from ..utils import NSAHelper


# @triton.autotune([triton.Config({'BLOCK_N': bsn, 'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['D1', "D2", "VD"])
@triton.jit
def _cmp_fwd_kernel(
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
    tl.store(desc_o, acc.to(desc_o.type.element_ty), boundary_check=(0, 1)) 
    tl.store(LSE + cp_idx * lse_stride_n, lse, mask=cp_idx < cp_len)
    
# @triton.autotune([triton.Config({'BLOCK_N': bsn, 'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=["D1", "D2", "VD"])
@triton.jit
def _cmp_dkdv_kernel(
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

        desc_q = tl.advance(desc_q, (BLOCK_N, 0))
        desc_do = tl.advance(desc_do, (BLOCK_N, 0))
        if D2 > 0:
            desc_q2 = tl.advance(desc_q2, (BLOCK_N, 0))

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
def _cmp_dq_kernel(
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
        tl.atomic_add(DQ + cp_idx[:, None] * dq_stride_n + tl.arange(0, D1)[None, :], (acc_dq * sm_scale), mask=mask[:, None])
        if D2 > 0:
            tl.atomic_add(DQ + D1 + cp_idx[:, None] * dq_stride_n + tl.arange(0, D2)[None, :], (acc_dq2 * sm_scale), mask=mask[:, None])

# @triton.autotune([triton.Config({'BLOCK_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [16, 32, 64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['VD'])
@triton.jit
def _cmp_bwd_preprocess(
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
        kwargs = {"BLOCK_N": 128, "BLOCK_M": 64, "num_warps": 8, "num_stages": 3}
    else:
        kwargs = {"BLOCK_N": 128, "BLOCK_M": 64, "num_warps": 4, "num_stages": 1}

    grid = lambda meta: (triton.cdiv(S, meta['BLOCK_N']), B, QH)
    
    _cmp_fwd_kernel[grid](
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
    _cmp_bwd_preprocess[grid](
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
        kwargs = {"BLOCK_N": 64, "BLOCK_M": 64, "num_warps": 4, "num_stages": 1}
    else:
        kwargs = {"BLOCK_N": 64, "BLOCK_M": 64, "num_warps": 4, "num_stages": 3}
    grid = lambda meta: (QH, triton.cdiv(y_maxlen, meta["BLOCK_M"]), B)
    _cmp_dkdv_kernel[grid](
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
        kwargs = {"BLOCK_N": 128, "BLOCK_M": 32, "num_warps": 4, "num_stages": 2}
    else:
        kwargs = {"BLOCK_N": 64, "BLOCK_M": 32, "num_warps": 4, "num_stages": 3}
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_N"]), B, QH)
    def func():
        _cmp_dq_kernel[grid](
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
    

ALIGN_FACTOR=8

# @triton.autotune([triton.Config({'BLOCK_M': bsm, 'BLOCK_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsm in [64, 128]
#                  for bsn in [64, 128]
#                  for ns in [1, 2,3, 4]
#                  for nw in [4, 8]
#                  ], key=['D1', "D2"])
@triton.jit
def _attn_probs_kernel(
    Q, 
    K, 
    Lse, 
    P,
    X_CU_SEQLENS, 
    Y_CU_SEQLENS,
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    lse_stride_h, lse_stride_n,
    p_stride_h, p_stride_n, p_stride_m,
    sm_scale, 
    kernel_size, 
    stride,
    KH, 
    G, 
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    BLOCK_N: tl.constexpr=64, 
    BLOCK_M: tl.constexpr=64
):
    cp_start_n = tl.program_id(2) * BLOCK_N
    start_m = tl.program_id(1) * BLOCK_M
    off_bh = tl.program_id(0)
    off_kh = off_bh % KH
    cp_off_b = off_bh // KH
    if not CP:
        off_b = cp_off_b
        start_n = cp_start_n
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if start_n >= x_len:
            return
        cp_bos = x_bos
        cp_len = x_len
        cp_offset = 0
    else:
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        start_n = cp_start_n + cp_offset
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if cp_start_n >= cp_len:
            return

    if (start_n + BLOCK_N) < (start_m * stride + kernel_size):
        return  

    y_bos, y_eos = tl.load(Y_CU_SEQLENS + off_b), tl.load(Y_CU_SEQLENS + off_b + 1)
    y_len = y_eos - y_bos

    if y_len == 0:
        return

    Q += cp_bos * q_stride_n
    K += y_bos * k_stride_m  + off_kh * k_stride_h
    Lse += cp_bos * lse_stride_n
    P += off_kh * p_stride_h.to(tl.int64) + cp_bos * p_stride_n.to(tl.int64)
    # P += off_kh * p_stride_h + cp_bos * p_stride_n

    cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
    q_idx = cp_idx + cp_offset
    desc_p = tl.make_block_ptr(P, (cp_len, y_len), (p_stride_n, p_stride_m), (cp_start_n, start_m), (BLOCK_N, BLOCK_M), (1, 0))
    desc_k = tl.make_block_ptr(K, (y_len, D1), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D1), (1, 0))
    k = tl.load(desc_k, boundary_check=(1, 0))
    if D2 > 0:
        desc_k2 = tl.make_block_ptr(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D2), (1, 0))
        k2 = tl.load(desc_k2, boundary_check=(1, 0))

    sm_scale *= 1.44269504
    p = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for off_qh in range(off_kh * G, off_kh * G + G):
        lse = tl.load(Lse + off_qh * lse_stride_h + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)
        desc_q = tl.make_block_ptr(Q + off_qh * q_stride_h, (cp_len, D1), (q_stride_n, q_stride_d), (cp_start_n, 0), (BLOCK_N, D1), (1, 0))
        q = tl.load(desc_q, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            desc_q2 = tl.make_block_ptr(Q + off_qh * q_stride_h + D1, (cp_len, D2), (q_stride_n, q_stride_d), (cp_start_n, 0), (BLOCK_N, D2), (1, 0))
            q2 = tl.load(desc_q2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        p += tl.exp2(tl.fma(attn_score, sm_scale,  -lse[:, None]))
        # p += tl.exp2(attn_score * sm_scale -lse[:, None])

    if start_n < ((start_m + BLOCK_M - 1) * stride + kernel_size - 1):
        k_idx = (start_m + tl.arange(0, BLOCK_M)) * stride + kernel_size - 1
        causal_mask = q_idx[:, None] >= k_idx[None, :]
        p = tl.where(causal_mask, p, 0.)
    tl.store(desc_p, p.to(desc_p.type.element_ty), boundary_check=(0, 1))

# @triton.autotune([triton.Config({'BLOCK_N': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [2, 4, 8, 16, 32]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=["BLOCK_K"])
@triton.jit
def _slc_probs_topk_kernel(
    AP, 
    SP, 
    Ind,
    X_CU_SEQLENS,
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    ap_stride_h, ap_stride_n, ap_stride_m,
    sp_stride_h, sp_stride_n, sp_stride_k,
    ind_stride_h, ind_stride_n, ind_stride_k,
    kernel_size, 
    stride, 
    block_size, 
    top_n, 
    num_inital:tl.constexpr, 
    num_local:tl.constexpr,
    RETURN_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr=16,
):
    cp_start_n = tl.program_id(0) * BLOCK_N
    cp_off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    if not CP:
        off_b = cp_off_b
        start_n = cp_start_n
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if start_n >= x_len:
            return
        cp_bos = x_bos
        cp_len = x_len
        cp_offset = 0
    else:
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        start_n = cp_start_n + cp_offset
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if cp_start_n >= cp_len:
            return
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos

    y_len = (x_len - kernel_size) // stride + 1

    cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
    q_idx = cp_idx + cp_offset
    cp_mask = cp_idx < cp_len

    AP += off_h * ap_stride_h.to(tl.int64) + cp_bos * ap_stride_n.to(tl.int64)
    # AP += off_h * ap_stride_h + cp_bos * ap_stride_n
    Ind += off_h * ind_stride_h + cp_bos * ind_stride_n

    acc_p = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    select_idx = tl.arange(0, BLOCK_K)

    select_start = 0
    select_end = block_size
    compress_start = stride - kernel_size 
    # num_loops = (block_size + 2 * (kernel_size - stride) - kernel_size) // stride + 1
    num_loops = (block_size + kernel_size - stride) // stride
    compress_idx = (select_idx * block_size - kernel_size) // stride + 1
    for _ in range(num_loops):
        compress_end = compress_start + kernel_size
        area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
        w = area / stride
        mask = (compress_idx >= 0) & (compress_idx < y_len)
        p = tl.load(AP + cp_idx[:, None] * ap_stride_n.to(tl.int64) + compress_idx[None, :] * ap_stride_m, 
                    mask=cp_mask[:, None] & mask[None, :], other=0.) * w
        acc_p += p
        compress_idx += 1
        compress_start += stride

    if RETURN_P:
        SP += off_h * sp_stride_h.to(tl.int64) + cp_bos * sp_stride_n.to(tl.int64)
        tl.store(SP + cp_idx[:, None] * sp_stride_n.to(tl.int64) + select_idx[None, :] * sp_stride_k, 
                  acc_p, mask=cp_mask[:, None] & (select_idx[None, :] < tl.cdiv(x_len, block_size)))

    top_n = tl.minimum(top_n, (start_n + BLOCK_N - 1) // block_size + 1)

    num_k = q_idx // block_size
    for i in range(0, num_inital):
        tl.store(Ind + cp_idx * ind_stride_n + i, i, mask=cp_mask & (i <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K)[None, :] == i,
                        -1., acc_p)

    for i in range(0, num_local):
        tl.store(Ind + cp_idx * ind_stride_n + i + num_inital, num_k - i, mask=cp_mask & (i + num_inital <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K)[None, :] == (num_k - i)[:, None],
                        -1., acc_p)

    for i in range(num_inital+num_local, top_n):
        max_idx = tl.argmax(acc_p, axis=-1)
        tl.store(Ind + cp_idx * ind_stride_n + i * ind_stride_k, max_idx, mask=cp_mask & (i <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K)[None, :] == max_idx[:, None],
                    -1., acc_p)

# @triton.autotune([triton.Config({'BLOCK_M': bsm, 'BLOCK_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsm in [64, 128]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=["D1", "D2"])
@triton.jit
def _slc_probs_kernel(
    Q, 
    K, 
    Lse, 
    P,
    X_CU_SEQLENS, 
    Y_CU_SEQLENS,
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    lse_stride_h, lse_stride_n,
    p_stride_h, p_stride_n, p_stride_k,
    sm_scale, 
    SCALE:tl.constexpr,
    kernel_size: tl.constexpr, 
    stride: tl.constexpr,
    KH, 
    G, 
    D1: tl.constexpr,
    D2: tl.constexpr, 
    BLOCK_N: tl.constexpr=64, 
    BLOCK_M: tl.constexpr=64
):
    cp_start_n = tl.program_id(2) * BLOCK_N
    start_m = tl.program_id(1) * BLOCK_M
    off_bh = tl.program_id(0)
    off_kh = off_bh % KH
    cp_off_b = off_bh // KH
    if not CP:
        off_b = cp_off_b
        start_n = cp_start_n
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if (start_n >= x_len) or (x_len < kernel_size):
            return
        cp_bos = x_bos
        cp_len = x_len
        cp_offset = 0
    else:
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        start_n = cp_start_n + cp_offset
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if (cp_start_n >= cp_len) or (cp_len + cp_offset < kernel_size):
            return

    if tl.cdiv(start_n + BLOCK_N, 64) < start_m//4:
        return  

    y_bos, y_eos = tl.load(Y_CU_SEQLENS + off_b), tl.load(Y_CU_SEQLENS + off_b + 1)
    y_len = y_eos - y_bos
    
    if y_len == 0:
        return

    Q += cp_bos * q_stride_n + (off_kh * G) * q_stride_h
    K += y_bos * k_stride_m + off_kh * k_stride_h
    Lse += cp_bos * lse_stride_n + (off_kh * G) * lse_stride_h
    P += off_kh * p_stride_h.to(tl.int64) + cp_bos * p_stride_n.to(tl.int64) 

    block_idx = start_m + tl.arange(0, BLOCK_M)
    cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
    q_idx = cp_idx + cp_offset

    BLOCK_K: tl.constexpr = BLOCK_M // 4
    left_k_block_idx = start_m + tl.arange(0, BLOCK_K) * 4 - 1

    desc_p = tl.make_block_ptr(P, (cp_len, tl.cdiv(cp_len + cp_offset, 64)), (p_stride_n , p_stride_k), (cp_start_n, start_m//4), (BLOCK_N, BLOCK_K), (1, 0))
    desc_k = tl.make_block_ptr(K, (y_len, D1), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D1), (1, 0))
    desc_left_k = tl.make_block_ptr(K, (y_len, D1), (k_stride_m, k_stride_d), (start_m - 1, 0), (BLOCK_M, D1), (1, 0))
    left_k = tl.load(desc_left_k, boundary_check=(0, 1)).reshape(BLOCK_K, 4, D1)
    k = tl.load(desc_k, boundary_check=(0, 1))
    left_k = tl.where(tl.arange(0,4)[None, :, None] < 1, left_k, 0)
    left_k = tl.sum(left_k.to(tl.float32), 1).to(k.dtype)

    if D2 > 0:
        desc_k2 = tl.make_block_ptr(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D2), (1, 0))
        desc_left_k2 = tl.make_block_ptr(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (start_m - 1, 0), (BLOCK_M, D2), (1, 0))
        k2 = tl.load(desc_k2, boundary_check=(0, 1))
        left_k2 = tl.load(desc_left_k2, boundary_check=(0, 1)).reshape(BLOCK_K, 4, D2)
        left_k2 = tl.where(tl.arange(0,4)[None, :, None] < 1, left_k2, 0)
        left_k2 = tl.sum(left_k2, 1)

    sm_scale *= 1.44269504
    weight = (1 + (tl.arange(0, 4)<3))[None, None, :]
    main_p = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    left_p = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for off_qh in range(0, G):
        lse = tl.load(Lse + off_qh * lse_stride_h + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)
        desc_q = tl.make_block_ptr(Q + off_qh * q_stride_h, (cp_len, D1), (q_stride_n, q_stride_d), (cp_start_n, 0), (BLOCK_N, D1), (1, 0))
        q = tl.load(desc_q, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        left_score = tl.dot(q, tl.permute(left_k, 1, 0))
        if D2 > 0:
            desc_q2 = tl.make_block_ptr(Q + off_qh * q_stride_h + D1, (cp_len, D2), (q_stride_n, q_stride_d), (cp_start_n, 0), (BLOCK_N, D2), (1, 0))
            q2 = tl.load(desc_q2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
            left_score = tl.dot(q2, tl.permute(left_k2, 1, 0), left_score)
        # main_p += tl.exp2(attn_score * sm_scale - lse[:, None])
        # left_p += tl.exp2(left_score * sm_scale - lse[:, None])
        main_p += tl.exp2(tl.fma(attn_score, sm_scale, -lse[:, None]))
        left_p += tl.exp2(tl.fma(left_score, sm_scale, -lse[:, None]))

    if start_n < ((start_m + BLOCK_M - 1) * stride + kernel_size - 1):
        k_idx = block_idx * stride + kernel_size - 1
        left_k_idx = left_k_block_idx * stride + kernel_size - 1
        left_k_idx = tl.where(left_k_idx <= stride, 999999999, left_k_idx)
        causal_mask = q_idx[:, None] >= k_idx[None, :]
        left_mask = q_idx[:, None] >= left_k_idx[None, :]
        main_p = tl.where(causal_mask, main_p, 0.)
        left_p = tl.where(left_mask, left_p, 0.)

    p = left_p + tl.sum(main_p.reshape(BLOCK_N, BLOCK_K, 4) * weight, -1)
    if SCALE:
        tl.store(desc_p, (p * p * p).to(desc_p.type.element_ty), boundary_check=(0, 1))
    else:
        tl.store(desc_p, p.to(desc_p.type.element_ty), boundary_check=(0, 1))

# @triton.autotune([triton.Config({'BLOCK_N': bs}, num_stages=ns, num_warps=nw)
#                  for bs in [2, 4]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [1, 2, 4, 8]
#                  ], key=["BLOCK_K"])
@triton.jit
def _topk_kernel(
    P, 
    Ind,
    X_CU_SEQLENS,
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    p_stride_h, p_stride_n, p_stride_k,
    ind_stride_h, ind_stride_n, ind_stride_k,
    block_size: tl.constexpr,
    top_n: tl.constexpr, 
    num_inital: tl.constexpr, 
    num_local: tl.constexpr,
    H:tl.constexpr,
    BLOCK_K: tl.constexpr, 
    BLOCK_N: tl.constexpr=1,
):

    cp_start_n = tl.program_id(0) * BLOCK_N
    off_bh = tl.program_id(1)
    cp_off_b = off_bh // H
    off_h = off_bh % H
    # cp_off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    if not CP:
        off_b = cp_off_b
        start_n = cp_start_n
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if start_n >= x_len:
            return
        cp_bos = x_bos
        cp_len = x_len
        cp_offset = 0
    else:
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        start_n = cp_start_n + cp_offset
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if cp_start_n >= cp_len:
            return

    P += off_h * p_stride_h.to(tl.int64) + cp_bos * p_stride_n.to(tl.int64) 
    Ind += off_h * ind_stride_h + cp_bos * ind_stride_n

    cp_idx = cp_start_n + tl.arange(0, BLOCK_N)
    q_idx = cp_idx + cp_offset
    cp_mask = cp_idx < cp_len

    desc_p = tl.make_block_ptr(P, (cp_len, tl.cdiv(cp_len + cp_offset, 64)), (p_stride_n, p_stride_k), (cp_start_n, 0), (BLOCK_N, BLOCK_K), (1, 0))
    acc_p = tl.load(desc_p, boundary_check=(0, 1)).to(tl.float32)

    top_n = tl.minimum(top_n, (start_n + BLOCK_N - 1) // block_size + 1)
    num_k = q_idx // block_size

    for i in range(0, num_inital):
        tl.store(Ind + cp_idx * ind_stride_n + i, i, mask=cp_mask & (i <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K)[None, :] == i,
                        -1., acc_p)

    for i in range(0, num_local):
        tl.store(Ind + cp_idx * ind_stride_n + i + num_inital, num_k - i, mask=cp_mask & (i + num_inital <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K)[None, :] == (num_k - i)[:, None],
                        -1., acc_p)

    for i in range(num_inital+num_local, top_n):
        max_idx = tl.argmax(acc_p, axis=-1)
        tl.store(Ind + cp_idx * ind_stride_n + i, max_idx, mask=cp_mask & (i <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K)[None, :] == max_idx[:, None],
                    -1., acc_p)

@torch.no_grad()       
def slc_topk_indices_for_32_16_64(
    q, 
    k, 
    lse, 
    sm_scale=None, 
    return_slc_prob=False, 
    align=True, 
    ignore_index=None, 
    scale=False, 
    fp32=False
):
    QT, QH, D = q.shape
    KT, KH, D2 = k.shape
    assert QH % KH == 0
    G = QH // KH
    D1, D2 = NSAHelper.split_d(D)
    if sm_scale is None:
        sm_scale = D**-0.5
    top_n, num_inital, num_local = NSAHelper.top_n, NSAHelper.num_init_blocks, NSAHelper.num_local_blocks
    x_cu_seqlens, y_cu_seqlens, k_cu_seqlens = NSAHelper.x_cu_seqlens, NSAHelper.y_cu_seqlens, NSAHelper.k_cu_seqlens
    x_maxlen, y_maxlen = NSAHelper.x_maxlen, NSAHelper.y_maxlen
    cp_cu_seqlens, cp_maxlen = NSAHelper.cp_cu_seqlens, NSAHelper.cp_maxlen
    cp_batch_idx, cp_offset = NSAHelper.cp_batch_idx, NSAHelper.cp_offset
    CP = cp_cu_seqlens is not None and NSAHelper.is_context_parallel_enable
    S = x_maxlen if not CP else cp_maxlen
    B = len(x_cu_seqlens)-1 if not CP else len(cp_batch_idx)

    num_slc_blocks = triton.cdiv(x_maxlen, 64)

    if align:
        # pad_max_slc_blocks = triton.cdiv(num_slc_blocks, ALIGN_FACTOR) * ALIGN_FACTOR
        pad_max_slc_blocks = triton.next_power_of_2(num_slc_blocks)
    else:
        pad_max_slc_blocks = num_slc_blocks

    pad_max_slc_blocks = max(pad_max_slc_blocks, 8)
    slc_probs = torch.zeros(KH, QT, pad_max_slc_blocks, device=q.device, dtype=torch.float16 if not fp32 else torch.float)
    scale = False if fp32 else scale
    scale = False if pad_max_slc_blocks > 2048 else scale

    if D<=128:
        kwargs = {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages":3}
    else:
        kwargs = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages":2}

    grid = lambda meta: (B * KH, triton.cdiv(y_maxlen, meta['BLOCK_M']), triton.cdiv(S, meta['BLOCK_N']))
    _slc_probs_kernel[grid](
        q, 
        k, 
        lse, 
        slc_probs,
        x_cu_seqlens,
        y_cu_seqlens,
        CP,
        cp_cu_seqlens,
        cp_batch_idx,
        cp_offset,
        *q.stride(),
        *k.stride(),
        *lse.stride(),
        *slc_probs.stride(),
        sm_scale, 
        scale, 
        32, 
        16,
        KH, 
        G,
        D1, 
        D2,
        **kwargs
    )

    assert ignore_index is None or ignore_index >= num_slc_blocks
    ignore_index = ignore_index if ignore_index is not None else num_slc_blocks
    topk_indices = torch.full((KH, QT, top_n), ignore_index, dtype=torch.int32, device=q.device)

    BLOCK_K = max(triton.next_power_of_2(pad_max_slc_blocks), 16)
    BLOCK_N = 4
    if BLOCK_K >= 2048:
        BLOCK_N = 1

    grid=lambda meta: (triton.cdiv(S, meta['BLOCK_N']), B*KH)
    # 256k
    if BLOCK_K <= 4096:
        kwargs = {"BLOCK_N":BLOCK_N, "num_warps": 1, "num_stages": 1}
    else:
        # for cp 1M test 
        kwargs = {"BLOCK_N":BLOCK_N, "num_warps": 4, "num_stages": 4}
    _topk_kernel[grid](
        slc_probs, 
        topk_indices,
        x_cu_seqlens,
        CP,
        cp_cu_seqlens,
        cp_batch_idx,
        cp_offset,
        *slc_probs.stride(),
        *topk_indices.stride(),
        64, 
        top_n, 
        num_inital, 
        num_local,
        KH,
        BLOCK_K,
        **kwargs
    )
    # grid=lambda meta: (S, B, KH)
    # kwargs = {"num_warps": 1, "num_stages": 3}
    # _topk_one_row_kernel[grid](
    #     slc_probs, 
    #     topk_indices,
    #     x_cu_seqlens,
    #     CP,
    #     cp_cu_seqlens,
    #     cp_batch_idx,
    #     cp_offset,
    #     *slc_probs.stride(),
    #     *topk_indices.stride(),
    #     64, 
    #     pad_max_slc_blocks, 
    #     top_n, 
    #     num_inital, 
    #     num_local,
    #     BLOCK_K,
    #     **kwargs
    # )
    if not return_slc_prob:
        NSAHelper.clear_tensor_data(slc_probs)
        slc_probs = None
    return topk_indices, slc_probs
  
@torch.no_grad()
def slc_topk_indices(
    q: torch.Tensor, 
    k: torch.Tensor, 
    lse: torch.Tensor, 
    sm_scale: float = None, 
    return_slc_prob: bool = False, 
    align: bool = True, 
    ignore_index: int = None, 
    fp32: bool = False, 
    maybe_efficient_version: bool = False, 
    scale_slc_p: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    compute the topk indices for slc_attn
    q2k topk indices for forward and dq, named it as fwd_ind(find)
    k2q indices for dk and dv, named it as bwd_ind(bind)

    Args:
        q (torch.Tensor): [t, h, d].

        k (torch.Tensor): [totat_num_blocks, h, d].

        lse (torch.Tensor): [h, t].

        sm_scale (float): softmax scale.

        return_slc_prob (bool): Whether return slc probs.

        align (bool): Because the p.size(-1) is not fixed. Align to 8 bytes for hardware optimization.

        ignore_index (int): It's only for verifying the result's correctness.

        fp32 (bool): When write p to global memory, whether use fp32 dtype. If set False, the default dtype if fp16.

        maybe_efficient_version (bool): kernel_size=32, stride=16, block_size=64, it will choose another function, 
            it's only write slc_prob to global memory, save more memory and run fast.

        scale_slc_p (bool): when maybe_efficient_version=True and fp32=False, the slc_p = slc_p ** 3, the topk result have higher precision,
    '''


    kernel_size, stride, block_size = NSAHelper.kernel_size, NSAHelper.stride, NSAHelper.block_size
    top_n, num_inital, num_local = NSAHelper.top_n, NSAHelper.num_init_blocks, NSAHelper.num_local_blocks
    x_cu_seqlens, y_cu_seqlens = NSAHelper.x_cu_seqlens, NSAHelper.y_cu_seqlens
    x_maxlen, y_maxlen = NSAHelper.x_maxlen, NSAHelper.y_maxlen
    cp_cu_seqlens, cp_maxlen = NSAHelper.cp_cu_seqlens, NSAHelper.cp_maxlen
    cp_batch_idx, cp_offset = NSAHelper.cp_batch_idx, NSAHelper.cp_offset
    CP = cp_cu_seqlens is not None and NSAHelper.is_context_parallel_enable
    S = x_maxlen if not CP else cp_maxlen
    B = len(x_cu_seqlens)-1 if not CP else len(cp_batch_idx)

    if maybe_efficient_version and kernel_size == 32 and stride == 16 and block_size == 64 and num_inital >= 1:
        return slc_topk_indices_for_32_16_64(q, k, lse, sm_scale, return_slc_prob, align, ignore_index, scale_slc_p, fp32)

    QT, QH, D = q.shape
    KT, KH, D2 = k.shape
    assert QH % KH == 0
    G = QH // KH
    D1, D2 = NSAHelper.split_d(D)
    if sm_scale is None:
        sm_scale = D**-0.5

    if align:
        pad_y_maxlen = max(triton.next_power_of_2(y_maxlen), 8)
    else:
        pad_y_maxlen = y_maxlen

    attn_probs = torch.zeros(KH, QT, pad_y_maxlen, device=q.device, dtype=torch.float16 if not fp32 else torch.float)
    if D <= 128:
        kwargs = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}
    else:
        kwargs = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}
    grid = lambda meta: (B * KH, triton.cdiv(y_maxlen, meta['BLOCK_M']), triton.cdiv(S, meta['BLOCK_N']))
    _attn_probs_kernel[grid](
        q, 
        k, 
        lse, 
        attn_probs,
        x_cu_seqlens,
        y_cu_seqlens,
        CP,
        cp_cu_seqlens,
        cp_batch_idx,
        cp_offset,
        *q.stride(),
        *k.stride(),
        *lse.stride(),
        *attn_probs.stride(),
        sm_scale, 
        kernel_size, 
        stride,
        KH, 
        G,
        D1, 
        D2,
        **kwargs
    )
    # return attn_probs

    num_slc_blocks = triton.cdiv(x_maxlen, block_size)
    if align:
        pad_max_slc_blocks = max(triton.next_power_of_2(num_slc_blocks), 8)
    else:
        pad_max_slc_blocks = num_slc_blocks

    BLOCK_K = max(triton.next_power_of_2(pad_max_slc_blocks), 16)
    assert ignore_index is None or ignore_index >= num_slc_blocks
    ignore_index = ignore_index if ignore_index is not None else num_slc_blocks

    slc_probs = torch.zeros(KH, QT, pad_max_slc_blocks, device=q.device, dtype=torch.float16) if return_slc_prob else None
    topk_indices = torch.full((KH, QT, top_n), ignore_index, dtype=torch.int32, device=q.device)
    BLOCK_N = 8
    if BLOCK_K >= 1024:
        BLOCK_N = 4
    elif BLOCK_K >= 2048:
        BLOCK_N = 1

    grid=lambda meta: (triton.cdiv(S, meta['BLOCK_N']), B, KH)
    kwargs = {"BLOCK_N": BLOCK_N, "num_warps": 4, "num_stages": 2}
    _slc_probs_topk_kernel[grid](
        attn_probs, 
        slc_probs, 
        topk_indices,
        x_cu_seqlens,
        CP,
        cp_cu_seqlens,
        cp_batch_idx,
        cp_offset,
        *attn_probs.stride(),
        *(slc_probs.stride() if slc_probs is not None else (0,0,0)),
        *topk_indices.stride(),
        kernel_size, 
        stride, 
        block_size, 
        top_n, 
        num_inital, 
        num_local,
        return_slc_prob,
        BLOCK_K,
        **kwargs
    )
    NSAHelper.clear_tensor_data(attn_probs)
    return topk_indices, slc_probs

# @triton.autotune([triton.Config({'BLOCK_N': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [256, 512, 1024]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['top_n'])
@triton.jit
def _find_to_bind_kernel(
    FInd, 
    BInd,
    X_CU_SEQLENS, 
    K_CU_SEQLENS, 
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_K_CU_SEQLENS,
    find_stride_h, find_stride_n, find_stride_k,
    bind_stride_h, bind_stride_k, bind_stride_n,
    top_n: tl.constexpr,
    BLOCK_N: tl.constexpr=256,
):
    cp_start_n = tl.program_id(0) * BLOCK_N
    cp_off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    if not CP:
        off_b = cp_off_b
        start_n = cp_start_n
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if start_n >= (x_len * top_n):
            return  
        k_bos, k_eos = tl.load(K_CU_SEQLENS + off_b), tl.load(K_CU_SEQLENS + off_b + 1)
        k_len = k_eos - k_bos
        cp_bos = x_bos
        cp_len = x_len
        cp_k_bos = k_bos
        cp_k_len = k_len
    else:
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if cp_start_n >= (cp_len * top_n):
            return  
        cp_k_bos, cp_k_eos = tl.load(CP_K_CU_SEQLENS + cp_off_b), tl.load(CP_K_CU_SEQLENS + cp_off_b + 1)
        cp_k_len = cp_k_eos - cp_k_bos

    FInd += off_h * find_stride_h + cp_bos * find_stride_n
    BInd += off_h * bind_stride_h.to(tl.int64) + cp_k_bos * bind_stride_k.to(tl.int64) 

    off_n = cp_start_n + tl.arange(0, BLOCK_N)
    cp_idx = off_n // top_n

    find_idx = tl.load(FInd + off_n, mask=off_n < (cp_len*top_n), other=cp_k_len)
    tl.store(BInd + find_idx * bind_stride_k.to(tl.int64) + cp_idx, cp_idx + 1, find_idx < cp_k_len)

# @triton.autotune([triton.Config({'BLOCK_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [256, 512, 1024]
#                  for ns in [1, 2,3, 4]
#                  for nw in [4, 8]
#                  ], key=['N']) 
@triton.jit
def _reorder_bind(
    SRC_Ind, 
    DST_Ind, 
    Cnt, 
    K_CU_SEQLENS, 
    CP: tl.constexpr,
    CP_K_CU_SEQLENS, 
    CP_OFFSET, 
    ind_stride_h, ind_stride_k, ind_stride_n,
    cnt_stride_h, cnt_stride_k,
    block_size: tl.constexpr,
    N,
    BLOCK_N: tl.constexpr=1024, 
):
    off_h = tl.program_id(0)
    cp_off_b = tl.program_id(1)
    cp_off_k = tl.program_id(2)

    if not CP:
        off_b = cp_off_b
        off_k = cp_off_k
        k_bos, k_eos = tl.load(K_CU_SEQLENS + off_b), tl.load(K_CU_SEQLENS + off_b + 1)
        k_len = k_eos - k_bos
        if off_k >= k_len:
            return  
        cp_offset = 0
        cp_k_bos = k_bos
        cp_k_len = k_len
    else:
        cp_k_bos, cp_k_eos = tl.load(CP_K_CU_SEQLENS + cp_off_b), tl.load(CP_K_CU_SEQLENS + cp_off_b + 1)
        cp_k_len = cp_k_eos - cp_k_bos
        if cp_off_k >= cp_k_len:
            return  
        cp_offset = tl.load(CP_OFFSET + cp_off_b)

    SRC_Ind += (cp_k_bos + cp_off_k) * ind_stride_k.to(tl.int64) + off_h * ind_stride_h.to(tl.int64) 
    DST_Ind += (cp_k_bos + cp_off_k) * ind_stride_k.to(tl.int64) + off_h * ind_stride_h.to(tl.int64) 
    Cnt += (cp_k_bos + cp_off_k) * cnt_stride_k + off_h * cnt_stride_h

    last_cnt = 0
    cols = tl.arange(0, BLOCK_N)
    start = tl.maximum(cp_off_k * block_size - cp_offset, 0)
    end = tl.minimum(block_size * cp_k_len - cp_offset, N)
    for cp_start_n in range(start, end, BLOCK_N):
        cp_off_n = cp_start_n + cols
        ind = tl.load(SRC_Ind + cp_off_n, mask=cp_off_n < end, other=0)
        this_cnt = tl.sum(ind)
        if this_cnt > 0:
            this_cnt = tl.sum(tl.where(ind == 0, 0, 1))
            ind = tl.sort(ind, descending=False)
            tl.store(DST_Ind + last_cnt + cols - (BLOCK_N - this_cnt), ind - 1, mask=cols >= (BLOCK_N - this_cnt))
            last_cnt += this_cnt
    tl.store(Cnt, last_cnt)

def get_bind_from_find(fwd_ind, align=True, inplace=True, helper=NSAHelper):
    H, T, top_n = fwd_ind.shape

    block_size = NSAHelper.block_size
    x_cu_seqlens, k_cu_seqlens, x_maxlen, k_maxlen = helper.x_cu_seqlens, helper.k_cu_seqlens, helper.x_maxlen, helper.k_maxlen
    cp_cu_seqlens, cp_maxlen = helper.cp_cu_seqlens, helper.cp_maxlen
    cp_batch_idx, cp_offset = helper.cp_batch_idx, helper.cp_offset
    cp_k_cu_seqlens, cp_k_maxlen = helper.cp_k_cu_seqlens, helper.cp_k_maxlen
    CP = cp_cu_seqlens is not None and NSAHelper.is_context_parallel_enable
    S = x_maxlen if not CP else cp_maxlen
    S2 = k_maxlen if not CP else cp_k_maxlen
    B = len(x_cu_seqlens)-1 if not CP else len(cp_batch_idx)
    total_num_slc_blocks = helper.k_len if not CP else helper.cp_k_len 

    pad_x_maxlen = S
    if align:
        pad_x_maxlen = max(triton.next_power_of_2(S), 8)
        total_num_slc_blocks = max(triton.next_power_of_2(total_num_slc_blocks), 8)

    bwd_ind = torch.zeros(H, total_num_slc_blocks, pad_x_maxlen, dtype=torch.int32, device=fwd_ind.device)
    grid = lambda meta: (triton.cdiv(S * top_n, meta['BLOCK_N']), B, H)
    kwargs = {"BLOCK_N": 512, "num_warps": 4, "num_stages": 3}
    _find_to_bind_kernel[grid](
        fwd_ind,
        bwd_ind,
        x_cu_seqlens,
        k_cu_seqlens,
        CP,
        cp_cu_seqlens,
        cp_k_cu_seqlens,
        *fwd_ind.stride(),
        *bwd_ind.stride(),
        top_n,
        **kwargs
    )
    # inplace = False
    src_ind = bwd_ind
    dst_ind = bwd_ind if inplace else torch.empty_like(bwd_ind)

    count = torch.empty(H, total_num_slc_blocks, dtype=torch.int32, device=bwd_ind.device)
    grid = (H, B, S2)
    kwargs = {"BLOCK_N": 256, "num_warps": 4, "num_stages": 4}
    _reorder_bind[grid](
        src_ind, 
        dst_ind, 
        count, 
        k_cu_seqlens,
        CP,
        cp_k_cu_seqlens, 
        cp_offset, 
        *bwd_ind.stride(),
        *count.stride(),
        block_size,
        pad_x_maxlen,
        **kwargs
        )
    return dst_ind, count


# @triton.autotune([triton.Config({}, num_warps=nw, num_stages=ns)
#                  for nw in [1, 2, 4, 8]
#                  for ns in [1,2,3,4]
#                  ], key=["D1", "D2", "VD" 'BLOCK_H', 'BLOCK_M'])
@triton.jit
def _slc_fwd_kernel(
    Q,
    K,
    V,
    O,
    Lse, 
    Ind,
    X_CU_SEQLENS, 
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    v_stride_m, v_stride_h, v_stride_d,
    o_stride_n, o_stride_h, o_stride_d,
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

    Q += start_qh * q_stride_h + (cp_bos + cp_idx) * q_stride_n
    O += start_qh * o_stride_h + (cp_bos + cp_idx) * o_stride_n
    K += off_kh * k_stride_h + cp_bos * k_stride_m
    V += off_kh * v_stride_h + cp_bos * v_stride_m
    Ind += off_kh * ind_stride_h + (cp_bos + cp_idx) * ind_stride_n
    Lse += (cp_bos + cp_idx) * lse_stride_n + start_qh * lse_stride_h

    desc_q = tl.make_block_ptr(Q, (G, D1), (q_stride_h, q_stride_d), (0, 0), (BLOCK_H, D1), (1, 0))
    desc_o = tl.make_block_ptr(O, (G, D1), (o_stride_h, o_stride_d), (0, 0), (BLOCK_H, D1), (1, 0))
    q = tl.load(desc_q, boundary_check=(0, 1))
    if D2 > 0:
        desc_q2 = tl.make_block_ptr(Q + D1, (G, D2), (q_stride_h, q_stride_d), (0, 0), (BLOCK_H, D2), (1, 0))
        q2 = tl.load(desc_q2, boundary_check=(0, 1))

    sm_scale *= 1.44269504
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, VD], dtype=tl.float32)

    stop_n = tl.constexpr(tl.minimum(top_n, tl.cdiv(q_idx+1, BLOCK_M)))
    for i in tl.range(0, stop_n):
        start_m = tl.load(Ind + i) * BLOCK_M
        desc_k = tl.make_block_ptr(K, (x_len, D1), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D1), (1, 0))
        k = tl.load(desc_k, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            desc_k2 = tl.make_block_ptr(K + D1, (x_len, D2), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D2), (1, 0))
            k2 = tl.load(desc_k2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        desc_v = tl.make_block_ptr(V, (x_len, VD), (v_stride_m, v_stride_d), (start_m, 0), (BLOCK_M, VD), (1, 0))
        v = tl.load(desc_v, boundary_check=(0, 1))
        attn_score = tl.where(q_idx >= (start_m + tl.arange(0, BLOCK_M))[None, :], attn_score * sm_scale, float('-inf'))
        new_m_i = tl.maximum(m_i, tl.max(attn_score, axis=1))
        alpha = tl.exp2(m_i - new_m_i)
        exp_attn_score = tl.exp2(attn_score - new_m_i[:, None])
        l_i = tl.fma(l_i, alpha, tl.sum(exp_attn_score, axis=-1))
        acc = acc * alpha[:, None] + tl.dot(exp_attn_score.to(v.dtype), v)
        m_i = new_m_i

    acc /= l_i[:, None]
    lse = m_i + tl.log2(l_i)

    tl.store(desc_o, acc.to(desc_o.type.element_ty), boundary_check=(0, 1))
    tl.store(Lse + tl.arange(0, BLOCK_H), lse, mask=tl.arange(0, BLOCK_H) < G)

# @triton.autotune([triton.Config({'BLOCK_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [16, 32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['T'])
@triton.jit
def _slc_bwd_preprocess(
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
def _slc_dkdv_kernel(
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
    desc_k = tl.make_block_ptr(K, (x_len, D1), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D1), (1, 0))
    desc_v = tl.make_block_ptr(V, (x_len, VD), (v_stride_m, v_stride_d), (start_m, 0), (BLOCK_M, VD), (1, 0))
    if D2 > 0:
        desc_k2 = tl.make_block_ptr(K + D1, (x_len, D2), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D2), (1, 0))
    k = tl.load(desc_k, boundary_check=(0, 1))
    v = tl.load(desc_v, boundary_check=(0, 1))
    acc_dk = tl.zeros((BLOCK_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_M, VD), dtype=tl.float32)

    if D2 > 0:
        k2 = tl.load(desc_k2, boundary_check=(0, 1))
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
        desc_dk = tl.make_block_ptr(DK, (x_len, D1), (dk_stride_m, dk_stride_d), (start_m, 0), (BLOCK_M, D1), (1, 0))
        desc_dv = tl.make_block_ptr(DV, (x_len, D1), (dv_stride_m, dv_stride_d), (start_m, 0), (BLOCK_M, D1), (1, 0))
        tl.store(desc_dk, (acc_dk * sm_scale).to(desc_dk.type.element_ty), boundary_check=(0, 1))
        tl.store(desc_dv, acc_dv.to(desc_dv.type.element_ty), boundary_check=(0, 1))
        if D2 > 0:
            desc_dk2 = tl.make_block_ptr(DK + D1, (x_len, D2), (dk_stride_m, dk_stride_d), (start_m, 0), (BLOCK_M, D2), (1, 0))
            tl.store(desc_dk2, (acc_dk2 * sm_scale).to(desc_dk2.type.element_ty), boundary_check=(0, 1))
    else:
        dk_ptrs = DK + off_m[:, None] * dk_stride_m + tl.arange(0, D1)[None, :] * dk_stride_d
        dv_ptrs = DV + off_m[:, None] * dv_stride_m + tl.arange(0, D1)[None, :] * dv_stride_d
        mask = off_m < x_len
        tl.atomic_add(dk_ptrs, acc_dk * sm_scale, mask=mask[:, None])
        tl.atomic_add(dv_ptrs, acc_dv, mask=mask[:, None])
        if D2 > 0:
            dk_ptrs2 = DK + off_m[:, None] * dk_stride_m + (tl.arange(0, D2)[None, :] + D1) * dk_stride_d
            tl.atomic_add(dk_ptrs2, acc_dk2 * sm_scale, mask=mask[:, None])

# @triton.autotune([triton.Config({}, num_warps=nw, num_stages=ns)
#                  for nw in [1, 2, 4, 8]
#                  for ns in [1, 2, 3, 4]
#                  ], key=['D1',"D2", "BLOCK_H", "BLOCK_M", "ATOMIC"])
@triton.jit
def _slc_dq_kernel( 
    Q,
    K,
    V,
    DO,
    DQ,
    Lse, 
    Delta,
    Ind,
    X_CU_SEQLENS,
    CP: tl.constexpr,
    CP_CU_SEQLENS, 
    CP_BATCH_IDX, 
    CP_OFFSET,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    v_stride_m, v_stride_h, v_stride_d,
    do_stride_n, do_stride_h, do_stride_d,
    dq_stride_n, dq_stride_h, dq_stride_d,
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

    Q += start_qh * q_stride_h + (cp_bos + cp_idx) * q_stride_n
    DO += start_qh * do_stride_h + (cp_bos + cp_idx) * do_stride_n
    K += off_kh * k_stride_h + cp_bos * k_stride_m
    V += off_kh * v_stride_h + cp_bos * v_stride_m
    DQ += start_qh * dq_stride_h + (cp_bos + cp_idx) * dq_stride_n
    Ind += off_kh * ind_stride_h + (cp_bos + cp_idx) * ind_stride_n
    Lse += (cp_bos + cp_idx) * lse_stride_n + start_qh * lse_stride_h
    Delta += (cp_bos + cp_idx) * lse_stride_n + start_qh * lse_stride_h

    desc_q = tl.make_block_ptr(Q, (G, D1), (q_stride_h, q_stride_d), (0, 0), (BLOCK_H, D1), (1, 0))
    desc_do = tl.make_block_ptr(DO, (G, D1), (do_stride_h, do_stride_d), (0, 0), (BLOCK_H, D1), (1, 0))
    q = tl.load(desc_q, boundary_check=(0, 1))
    do = tl.load(desc_do, boundary_check=(0, 1))
    lse = tl.load(Lse + tl.arange(0, BLOCK_H) * lse_stride_h, mask=tl.arange(0, BLOCK_H)<G, other=0.)
    delta = tl.load(Delta + tl.arange(0, BLOCK_H) * lse_stride_h, mask=tl.arange(0, BLOCK_H)<G, other=0.)
    acc_dq = tl.zeros([BLOCK_H, D1], dtype=tl.float32)

    if D2 > 0:
        desc_q2 = tl.make_block_ptr(Q + D1, (G, D2), (q_stride_h, q_stride_d), (0, 0), (BLOCK_H, D2), (1, 0))
        q2 = tl.load(desc_q2, boundary_check=(0, 1))
        acc_dq2 = tl.zeros([BLOCK_H, D2], dtype=tl.float32)

    sm_scale_ln2 = sm_scale * 1.44269504
    stop_n = tl.minimum(top_n, tl.cdiv(q_idx+1, BLOCK_M))
    for i in range(0, stop_n):
        start_m = tl.load(Ind + i).to(tl.int32) * BLOCK_M
        desc_k = tl.make_block_ptr(K, (x_len, D1), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D1), (1, 0))
        k = tl.load(desc_k, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            desc_k2 = tl.make_block_ptr(K + D1, (x_len, D2), (k_stride_m, k_stride_d), (start_m, 0), (BLOCK_M, D2), (1, 0))
            k2 = tl.load(desc_k2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

        attn_score = tl.where(q_idx >= (start_m + tl.arange(0, BLOCK_M))[None, :], attn_score * sm_scale_ln2, float('-inf'))
        p = tl.exp2(attn_score - lse[:, None])
        desc_v = tl.make_block_ptr(V, (x_len, VD), (v_stride_m, v_stride_d), (start_m, 0), (BLOCK_M, VD), (1, 0))
        v = tl.load(desc_v, boundary_check=(0, 1))
        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None]) 
        acc_dq = tl.dot(ds.to(k.dtype), k, acc_dq)
        if D2 > 0:
            acc_dq2 = tl.dot(ds.to(k.dtype), k2, acc_dq2)

    if not ATOMIC:
        desc_dq = tl.make_block_ptr(DQ, (G, D1), (dq_stride_h, dq_stride_d), (0, 0), (BLOCK_H, D1), (1, 0))
        tl.store(desc_dq, (acc_dq * sm_scale).to(desc_dq.type.element_ty), boundary_check=(0, 1))
        if D2 > 0:
            desc_dq2 = tl.make_block_ptr(DQ + D1, (G, D2), (dq_stride_h, dq_stride_d), (0, 0), (BLOCK_H, D2), (1, 0))
            tl.store(desc_dq2, (acc_dq2 * sm_scale).to(desc_dq2.type.element_ty), boundary_check=(0, 1))
    else:
        dq_ptrs = DQ + tl.arange(0, BLOCK_H)[:, None] * dq_stride_h + tl.arange(0, D1)[None, :] * dq_stride_d
        mask = tl.arange(0, BLOCK_H) < G
        tl.atomic_add(dq_ptrs, acc_dq * sm_scale, mask=mask[:, None])
        if D2 > 0:
            dq2_ptrs = DQ + tl.arange(0, BLOCK_H)[:, None] * dq_stride_h + tl.arange(0, D2)[None, :] * dq_stride_d + D1
            tl.atomic_add(dq2_ptrs, acc_dq2 * sm_scale, mask=mask[:, None])

def slc_attn_fwd(q, k, v, topk, sm_scale=None, o=None, helper=NSAHelper):
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
    if D <= 128:
        kwargs = {"num_warps": 4, "num_stages": 2}
    else:
        kwargs = {"num_warps": 1, "num_stages": 1}
    grid = lambda meta: (S, B * KH)
    _slc_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse, 
        topk,
        x_cu_seqlens,
        CP,
        cp_cu_seqlens,
        cp_batch_idx,
        cp_offset,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
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
    fuse_dqdkdv = False
    # if dq is None:
    #     dkdv_repeat = True
    # dkdv_repeat = False
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
    _slc_bwd_preprocess[grid](
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
        kwargs = {"BLOCK_N":64, "num_warps": 4, "num_stages": 1}
    else:
        kwargs = {"BLOCK_N":64, "num_warps": 4, "num_stages": 1}
    grid = (B, QH, triton.cdiv(x_maxlen, BLOCK_M))
    _slc_dkdv_kernel[grid](
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

    if D <= 128:
        kwargs = {"num_warps": 4, "num_stages": 3}
    else:
        kwargs = {"num_warps": 2, "num_stages": 2}
    grid = lambda meta: (S, B * KH)
    def func():
        _slc_dq_kernel[grid](
            q,
            k,
            v,
            do,
            dq,
            lse, 
            delta, 
            topk,
            x_cu_seqlens,
            CP,
            cp_cu_seqlens,
            cp_batch_idx,
            cp_offset,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *do.stride(),
            *dq.stride(),
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
    