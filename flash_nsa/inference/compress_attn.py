# Copyright (c) 2025 Duyue Ma

import math

import torch
import triton
import triton.language as tl

from .utils import split_d, use_tma

# @triton.autotune([triton.Config({'BLOCK_N': bsn, 'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['D1', "D2", "VD"])
@triton.jit
def _prefill_kernel(
    Q, 
    K, 
    V, 
    O, 
    LSE, 
    X_CU_SEQLENS, 
    CONTEXT_LENS,
    TABLES,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_n, k_stride_h, k_stride_d,
    v_stride_n, v_stride_h, v_stride_d,
    o_stride_n, o_stride_h, o_stride_d,
    lse_stride_h, lse_stride_n,
    table_stride,
    sm_scale, 
    kernel_size, 
    stride,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    G: tl.constexpr, 
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_N: tl.constexpr=64, 
    BLOCK_M: tl.constexpr=128
):

    start_n = tl.program_id(0) * BLOCK_N
    off_b = tl.program_id(1)
    off_qh = tl.program_id(2)
    off_kh = off_qh // G

    x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
    x_len = x_eos - x_bos
    if start_n >= x_len:
        return
    cached_tokens = tl.load(CONTEXT_LENS + off_b) - x_len
    
    if x_len + cached_tokens <= kernel_size - 1:
        y_len = 0
    else:
        y_len = (x_len + cached_tokens - kernel_size) // stride + 1


    Q += x_bos * q_stride_n + off_qh * q_stride_h
    K += off_kh * k_stride_h
    V += off_kh * v_stride_h
    O += x_bos * o_stride_n + off_qh * o_stride_h
    LSE += off_qh * lse_stride_h + x_bos * lse_stride_n

    off_n = start_n + tl.arange(0, BLOCK_N)
    q_idx = off_n + cached_tokens

    desc_q = tl.make_tensor_descriptor(Q, (x_len, D1), (q_stride_n, q_stride_d), (BLOCK_N, D1))
    desc_o = tl.make_tensor_descriptor(O, (x_len, VD), (o_stride_n, o_stride_d), (BLOCK_N, VD))
    desc_k = tl.make_tensor_descriptor(K, (num_blocks * PAGE_SIZE, D1), (k_stride_n, k_stride_d), (BLOCK_M, D1))
    desc_v = tl.make_tensor_descriptor(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (BLOCK_M, VD))
    if D2 > 0:
        desc_q2 = tl.make_tensor_descriptor(Q + D1, (x_len, D2), (q_stride_n, q_stride_d), (BLOCK_N, D1))
        desc_k2 = tl.make_tensor_descriptor(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (BLOCK_M, D2))
    q = desc_q.load([start_n, 0])
    if D2 > 0:
        q2 = desc_q2.load([start_n, 0])

    sm_scale *= 1.44269504
    m_i = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_N, VD], dtype=tl.float32)

    mid = tl.minimum((start_n + cached_tokens - kernel_size) // stride + 1, y_len).to(tl.int32)
    mid = (mid // BLOCK_M) * BLOCK_M
    end = tl.minimum((start_n + cached_tokens + BLOCK_N - kernel_size) // stride + 1, y_len).to(tl.int32)
    for i in range(0, mid, BLOCK_M):
        offset = i % PAGE_SIZE
        table_idx = i // PAGE_SIZE
        cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
        k = desc_k.load([cache_idx * PAGE_SIZE + offset, 0])
        v = desc_v.load([cache_idx * PAGE_SIZE + offset, 0])
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            k2 = desc_k2.load([cache_idx * PAGE_SIZE + offset, 0])
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

    for i in range(mid, end, BLOCK_M):
        offset = i % PAGE_SIZE
        table_idx = i // PAGE_SIZE
        cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
        k = desc_k.load([cache_idx * PAGE_SIZE + offset, 0])
        v = desc_v.load([cache_idx * PAGE_SIZE + offset, 0])
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            k2 = desc_k2.load([cache_idx * PAGE_SIZE + offset, 0])
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

        k_idx = (i + tl.arange(0, BLOCK_M)) * stride + kernel_size - 1
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
    if start_n == 0:
        acc = tl.where(q_idx[:, None]>=(kernel_size-1), acc, 0)
        lse = tl.where(q_idx>=(kernel_size-1), lse, 0)
    desc_o.store([start_n, 0], acc.to(desc_o.dtype)) 
    tl.store(LSE + off_n * lse_stride_n, lse, mask=off_n < x_len)
    
# @triton.autotune([triton.Config({'BLOCK_M': bsm, "NUM_SPLITS": num_splits}, num_stages=ns, num_warps=nw)
#                 for num_splits in [4, 8]
#                  for bsm in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [2, 4, 8]
#                  ], key=['D1', "D2", "BLOCK_H"])
@triton.jit
def _decode_stage1_kernel(
    Q, 
    K, 
    V, 
    O, 
    LSE, 
    CONTEXT_LENS,
    TABLES,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_n, k_stride_h, k_stride_d,
    v_stride_n, v_stride_h, v_stride_d,
    o_stride_k, o_stride_n, o_stride_h, o_stride_d,
    lse_stride_k, lse_stride_n, lse_stride_h,
    table_stride,
    sm_scale, 
    kernel_size, 
    stride,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    G: tl.constexpr, 
    G2: tl.constexpr,
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_H: tl.constexpr=16, 
    BLOCK_M: tl.constexpr=128,
    NUM_SPLITS: tl.constexpr=4,
):

    off_b = tl.program_id(0)
    start_qh = tl.program_id(1) * G2
    off_k = tl.program_id(2)
    off_kh = start_qh // G

    x_len = tl.load(CONTEXT_LENS + off_b)
    if x_len <= kernel_size - 1:
        return
    y_len = tl.maximum((x_len - kernel_size) // stride + 1, 0)
    q_idx = x_len - 1

    if NUM_SPLITS != 0:
        num_splits = NUM_SPLITS
    else:
        if x_len < 64 * 1024:
            num_splits = 2
        else:
            num_splits = 4

    split_size = tl.cdiv(tl.cdiv(y_len, num_splits), BLOCK_M) * BLOCK_M
    kv_start = off_k * split_size
    
    if kv_start < y_len:
        kv_end = tl.minimum(kv_start + split_size, y_len)
        kv_mid = tl.minimum((y_len // BLOCK_M) * BLOCK_M, kv_end)
        
        Q += off_b * q_stride_n + start_qh * q_stride_h
        K += off_kh * k_stride_h
        V += off_kh * v_stride_h
        O += off_k * o_stride_k + off_b * o_stride_n + start_qh * o_stride_h
        LSE += off_k * lse_stride_k + start_qh * lse_stride_h + off_b * lse_stride_n

        q = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D1)[None, :], tl.arange(0, BLOCK_H)[:, None] < G2)
        desc_k = tl.make_tensor_descriptor(K, (num_blocks * PAGE_SIZE, D1), (k_stride_n, k_stride_d), (BLOCK_M, D1))
        desc_v = tl.make_tensor_descriptor(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (BLOCK_M, VD))
        if D2 > 0:
            q2 = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, tl.arange(0, BLOCK_H)[:, None] < G2)
            desc_k2 = tl.make_tensor_descriptor(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (BLOCK_M, D2))
           
        sm_scale *= 1.44269504
        m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, VD], dtype=tl.float32)

        for i in range(kv_start, kv_mid, BLOCK_M):
            offset = i % PAGE_SIZE
            table_idx = i // PAGE_SIZE
            cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
            k = desc_k.load([cache_idx * PAGE_SIZE + offset, 0])
            v = desc_v.load([cache_idx * PAGE_SIZE + offset, 0])
            attn_score = tl.dot(q, tl.permute(k, 1, 0))
            if D2>0:
                k2 = desc_k2.load([cache_idx * PAGE_SIZE + offset, 0])
                attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

            attn_score *= sm_scale

            m_ij = tl.max(attn_score, axis=1)
            new_m_i = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - new_m_i)

            exp_attn_score = tl.exp2(attn_score - new_m_i[:, None])

            l_i = tl.fma(l_i, alpha, tl.sum(exp_attn_score, axis=-1))
            acc = tl.fma(acc, alpha[:, None], tl.dot(exp_attn_score.to(v.dtype), v))
            m_i = new_m_i

        for i in range(kv_mid, kv_end, BLOCK_M):
            offset = i % PAGE_SIZE
            table_idx = i // PAGE_SIZE
            cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
            k = desc_k.load([cache_idx * PAGE_SIZE + offset, 0])
            v = desc_v.load([cache_idx * PAGE_SIZE + offset, 0])
            attn_score = tl.dot(q, tl.permute(k, 1, 0))
            if D2>0:
                k2 = desc_k2.load([cache_idx * PAGE_SIZE + offset, 0]).reshape(BLOCK_M, D2)
                attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

            k_idx = (i + tl.arange(0, BLOCK_M)) * stride + kernel_size - 1
            attn_score = tl.where(q_idx >= k_idx[None, :], attn_score * sm_scale, float('-inf'))

            m_ij = tl.max(attn_score, axis=1)
            new_m_i = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - new_m_i)

            exp_attn_score = tl.exp2(attn_score - new_m_i[:, None])

            l_i = tl.fma(l_i, alpha, tl.sum(exp_attn_score, axis=-1))
            acc = tl.fma(acc, alpha[:, None], tl.dot(exp_attn_score.to(v.dtype), v))
            m_i = new_m_i

        acc /= l_i[:, None]
        lse = m_i + tl.log2(l_i)
        if q_idx < (kernel_size - 1):
            acc = tl.zeros_like(acc)
            lse = tl.zeros_like(lse)
        tl.store(O + tl.arange(0, BLOCK_H)[:, None] * o_stride_h + tl.arange(0, D1)[None, :], acc, tl.arange(0, BLOCK_H)[:, None] < G2)
        tl.store(LSE + tl.arange(0, BLOCK_H) * lse_stride_h, lse, mask=tl.arange(0, BLOCK_H) < G2)

# @triton.autotune([triton.Config({'BLOCK_M': bsm, "NUM_SPLITS": num_splits}, num_stages=ns, num_warps=nw)
#                 for num_splits in [2, 4, 8]
#                  for bsm in [64]
#                  for ns in [2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['D1', "D2", "BLOCK_H"])
@triton.jit
def _decode_stage1_no_tma_kernel(
    Q, 
    K, 
    V, 
    O, 
    LSE, 
    CONTEXT_LENS,
    TABLES,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_n, k_stride_h, k_stride_d,
    v_stride_n, v_stride_h, v_stride_d,
    o_stride_k, o_stride_n, o_stride_h, o_stride_d,
    lse_stride_k, lse_stride_n, lse_stride_h,
    table_stride,
    sm_scale, 
    kernel_size, 
    stride,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    G: tl.constexpr, 
    G2: tl.constexpr,
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_H: tl.constexpr=16, 
    BLOCK_M: tl.constexpr=128,
    NUM_SPLITS: tl.constexpr=4,
):

    off_b = tl.program_id(0).to(tl.int64)
    start_qh = tl.program_id(1).to(tl.int64) * G2
    off_k = tl.program_id(2).to(tl.int64)
    off_kh = start_qh // G

    x_len = tl.load(CONTEXT_LENS + off_b)
    if x_len <= kernel_size - 1:
        return
    y_len = tl.maximum((x_len - kernel_size) // stride + 1, 0)
    q_idx = x_len - 1
    # num_splits = tl.load(NUM_SPLITS + off_b)
    num_splits = NUM_SPLITS
    split_size = tl.cdiv(tl.cdiv(y_len, num_splits), BLOCK_M) * BLOCK_M
    kv_start = off_k * split_size
    
    k_stride_n = tl.cast(k_stride_n, tl.int64)
    v_stride_n = tl.cast(v_stride_n, tl.int64)
    if kv_start < y_len:
        kv_end = tl.minimum(kv_start + split_size, y_len)
        kv_mid = tl.minimum((y_len // BLOCK_M) * BLOCK_M, kv_end)
        
        Q += off_b * q_stride_n + start_qh * q_stride_h
        K += off_kh * k_stride_h
        V += off_kh * v_stride_h
        O += off_k * o_stride_k + off_b * o_stride_n + start_qh * o_stride_h
        LSE += off_k * lse_stride_k + start_qh * lse_stride_h + off_b * lse_stride_n

        q = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D1)[None, :])
        if D2 > 0:
            q2 = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1)

        sm_scale *= 1.44269504
        m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, VD], dtype=tl.float32)

        for i in range(kv_start, kv_mid, BLOCK_M):
            offset = (i % PAGE_SIZE).to(tl.int64)
            table_idx = i // PAGE_SIZE
            cache_idx = tl.load(TABLES + off_b * table_stride + table_idx).to(tl.int64)
            k = tl.load(K + (cache_idx * PAGE_SIZE + offset + tl.arange(0, BLOCK_M))[:, None] * k_stride_n + tl.arange(0, D1)[None, :])
            v = tl.load(V + (cache_idx * PAGE_SIZE + offset + tl.arange(0, BLOCK_M))[:, None] * v_stride_n + tl.arange(0, D1)[None, :])
            attn_score = tl.dot(q, tl.permute(k, 1, 0))
            if D2>0:
                k2 = tl.load(K + (cache_idx * PAGE_SIZE + offset + tl.arange(0, BLOCK_M))[:, None] * k_stride_n + tl.arange(0, D2)[None, :] + D1)
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

        for i in range(kv_mid, kv_end, BLOCK_M):
            offset = (i % PAGE_SIZE).to(tl.int64)
            table_idx = i // PAGE_SIZE
            cache_idx = tl.load(TABLES + off_b * table_stride + table_idx).to(tl.int64)
            k = tl.load(K + (cache_idx * PAGE_SIZE + offset + tl.arange(0, BLOCK_M))[:, None] * k_stride_n + tl.arange(0, D1)[None, :])
            v = tl.load(V + (cache_idx * PAGE_SIZE + offset + tl.arange(0, BLOCK_M))[:, None] * v_stride_n + tl.arange(0, D1)[None, :])
            attn_score = tl.dot(q, tl.permute(k, 1, 0))
            if D2>0:
                k2 = tl.load(K + (cache_idx * PAGE_SIZE + offset + tl.arange(0, BLOCK_M))[:, None] * k_stride_n + tl.arange(0, D2)[None, :] + D1)
                attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

            k_idx = (i + tl.arange(0, BLOCK_M)) * stride + kernel_size - 1
            attn_score = tl.where(q_idx >= k_idx[None, :], attn_score * sm_scale, float('-inf'))

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
        if q_idx < (kernel_size - 1):
            acc = tl.zeros_like(acc)
            lse = tl.zeros_like(lse) + 1.
        tl.store(O + tl.arange(0, BLOCK_H)[:, None] * o_stride_h + tl.arange(0, VD)[None, :], acc)
        tl.store(LSE + tl.arange(0, BLOCK_H) * lse_stride_h, lse, mask=tl.arange(0, BLOCK_H) < G2)
        
        
        
# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                  for ns in [1, 2, 3, 4]
#                  for nw in [1, 2, 4, 8]
#                  ], key=['D1', "D2", "VD"])
@triton.jit
def _decode_stage2_kernel(
    MID_O,
    O,
    LSE, 
    CONTEXT_LENS,
    mo_stride_k, mo_stride_n, mo_stride_h, mo_stride_d,
    o_stride_n, o_stride_h, o_stride_d,
    lse_stride_k, lse_stride_n, lse_stride_h,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    VD: tl.constexpr=128, 
    BLOCK_M: tl.constexpr=64,
    NUM_SPLITS: tl.constexpr=8
):

    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_d = tl.arange(0, VD)

    x_len = tl.load(CONTEXT_LENS + off_b)
    y_len = tl.maximum((x_len - kernel_size) // stride + 1, 0)

    o_ptrs = O + off_b * o_stride_n + off_h * o_stride_h + off_d
    mid_o_ptrs = MID_O + off_b * mo_stride_n + off_h * mo_stride_h + off_d
    lse_ptrs = LSE + off_b * lse_stride_n + off_h * lse_stride_h
    
    if x_len <= kernel_size - 1:
        # o = tl.zeros((VD,), dtype=tl.float32)
        tl.store(o_ptrs, 0.)
        tl.store(lse_ptrs, 0.)
        return
    
    if NUM_SPLITS != 0:
        num_splits = NUM_SPLITS
    else:
        if x_len < 64 * 1024:
            num_splits = 2
        else:
            num_splits = 4

    split_size = tl.cdiv(tl.cdiv(y_len, num_splits), BLOCK_M) * BLOCK_M
    
    m_i = float("-inf")
    l_i = 0.
    acc = tl.zeros([VD], dtype=tl.float32)
    off_k = 0
    for kv_start in range(0, y_len, split_size):
        o_i = tl.load(mid_o_ptrs + off_k * mo_stride_k)
        m_ij = tl.load(lse_ptrs + off_k * lse_stride_k)
        
        new_m_i = tl.maximum(m_ij, m_i)
        old_scale = tl.exp2(m_i - new_m_i)
        this_scale = tl.exp2(m_ij - new_m_i)
        acc = tl.fma(acc, old_scale, this_scale * o_i)
        
        l_i = tl.fma(l_i, old_scale, this_scale)
        m_i = new_m_i
        off_k += 1
        
    tl.store(o_ptrs, acc/l_i)
    tl.store(lse_ptrs, m_i + tl.log2(l_i))

@use_tma
def cmp_attn_prefill(q, k, v, x_cu_seqlens, x_maxlen, block_tables, context_lens, kernel_size=32, stride=16, sm_scale=None):
    T, QH, D = q.shape
    num_blocks, PAGE_SIZE, KH, _ = k.shape
    
    VD = v.size(-1)
    D1, D2 = split_d(D)
    G = QH // KH
    B = len(x_cu_seqlens) - 1
    if sm_scale is None:
        sm_scale = D**-0.5
        
    k = k.view(-1, KH, D)
    v = v.view(-1, KH, VD)
    
    o = torch.empty(T, QH, VD, device=q.device, dtype=q.dtype)
    lse = torch.empty(QH, T, dtype=torch.float32, device=q.device)

    if D <= 128:
        kwargs = {"BLOCK_N": 128, "BLOCK_M": 64, "num_warps": 4, "num_stages": 2}
    else:
        kwargs = {"BLOCK_N": 128, "BLOCK_M": 64, "num_warps": 4, "num_stages": 1}
    grid = lambda meta: (triton.cdiv(x_maxlen, meta['BLOCK_N']), B, QH)
    _prefill_kernel[grid](
        q, 
        k, 
        v, 
        o, 
        lse,
        x_cu_seqlens, 
        context_lens, 
        block_tables,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        *lse.stride(),
        block_tables.stride(0),
        sm_scale, 
        kernel_size, 
        stride, 
        num_blocks,
        PAGE_SIZE,
        G, 
        D1, 
        D2, 
        VD,
        **kwargs
    )
    return o, lse

@use_tma
def cmp_attn_decode(q, k, v, block_tables, context_lens, out=None, num_splits=0, max_num_splits=4, kernel_size=32, stride=16, sm_scale=None, tma=True, bench=False):
    B, QH, D = q.shape
    num_blocks, PAGE_SIZE, KH, _ = k.shape
    VD = v.size(-1)
    D1, D2 = split_d(D)
    G = QH // KH
    if sm_scale is None:
        sm_scale = D**-0.5
        
    assert max_num_splits >= num_splits
    assert num_splits >= 0
        
    k = k.view(-1, KH, D)
    v = v.view(-1, KH, VD)
    
    BLOCK_H = 16
    
    mid_o = torch.empty(max_num_splits, B, QH, VD, device=q.device, dtype=q.dtype)
    lse = torch.empty(max_num_splits, B, QH, dtype=torch.float32, device=q.device)
    if out is None:
        out = torch.empty(B, QH, VD, device=q.device, dtype=q.dtype)

    NUM_SPLITS = num_splits
    if tma:
        BLOCK_M = 64
        kwargs = {"BLOCK_M": BLOCK_M, "NUM_SPLITS":NUM_SPLITS, "num_warps": 2, "num_stages": 2}
        func = _decode_stage1_kernel
    else:
        BLOCK_M = 64
        kwargs = {"BLOCK_M": BLOCK_M, "NUM_SPLITS":NUM_SPLITS, "num_warps": 4, "num_stages": 3}
        func = _decode_stage1_no_tma_kernel

    G2 = min(G, BLOCK_H)
    grid = lambda meta: (B, QH // G2, max_num_splits)
    func[grid](
        q, 
        k, 
        v, 
        mid_o, 
        lse,
        context_lens, 
        block_tables,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *mid_o.stride(),
        *lse.stride(),
        block_tables.stride(0),
        sm_scale, 
        kernel_size, 
        stride, 
        num_blocks,
        PAGE_SIZE,
        G, 
        G2,
        D1, 
        D2, 
        VD,
        BLOCK_H=BLOCK_H,
        **kwargs
    )
    if bench:
        print(triton.testing.do_bench(lambda:        func[grid](
            q, 
            k, 
            v, 
            mid_o, 
            lse,
            context_lens, 
            block_tables,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *mid_o.stride(),
            *lse.stride(),
            block_tables.stride(0),
            sm_scale, 
            kernel_size, 
            stride, 
            num_blocks,
            PAGE_SIZE,
            G, 
            G2,
            D1, 
            D2, 
            VD,
            BLOCK_H=BLOCK_H,
            **kwargs
        )))


    kwargs = {"num_warps": 1, "num_stages": 3}
    grid = lambda meta: (B, QH)
    _decode_stage2_kernel[grid](
        mid_o,
        out,
        lse,
        context_lens, 
        *mid_o.stride(),
        *out.stride(),
        *lse.stride(),
        kernel_size,
        stride,
        VD,
        BLOCK_M=BLOCK_M,
        NUM_SPLITS=NUM_SPLITS,
        **kwargs
    )
    if bench:
        print(triton.testing.do_bench(lambda: _decode_stage2_kernel[grid](
        mid_o,
        out,
        lse,
        context_lens, 
        *mid_o.stride(),
        *out.stride(),
        *lse.stride(),
        kernel_size,
        stride,
        VD,
        BLOCK_M=BLOCK_M,
        NUM_SPLITS=NUM_SPLITS,
        **kwargs
        )))
    return out, lse[0]

# @use_tma
# def cmp_attn_decod2(q, k, v, block_tables, context_lens, num_splits=None, kernel_size=32, stride=16, max_num_splits=4, sm_scale=None):
#     B, QH, D = q.shape
#     num_blocks, PAGE_SIZE, KH, _ = k.shape
#     VD = v.size(-1)
#     D1, D2 = split_d(D)
#     G = QH // KH
#     if sm_scale is None:
#         sm_scale = D**-0.5
        
#     k = k.view(-1, KH, D)
#     v = v.view(-1, KH, VD)
    
#     BLOCK_H = 16
    
#     o = torch.zeros(max_num_splits, B, QH, VD, device=q.device, dtype=q.dtype)
#     lse = torch.zeros(max_num_splits, B, QH, dtype=torch.float32, device=q.device)

#     BLOCK_M = 64
#     NUM_SPLITS = 4
#     kwargs = {"BLOCK_M": BLOCK_M, "NUM_SPLITS":NUM_SPLITS, "num_warps": 2, "num_stages": 2}
#     func = _decode_stage1_kernel

#     G2 = min(G, BLOCK_H)
#     grid = lambda meta: (B, QH // G2, meta['NUM_SPLITS'])
#     _decode_stage1_kernel[grid](
#         q, 
#         k, 
#         v, 
#         o, 
#         lse,
#         context_lens, 
#         block_tables,
#         *q.stride(),
#         *k.stride(),
#         *v.stride(),
#         *o.stride(),
#         *lse.stride(),
#         block_tables.stride(0),
#         sm_scale, 
#         kernel_size, 
#         stride, 
#         num_blocks,
#         PAGE_SIZE,
#         G, 
#         G2,
#         D1, 
#         D2, 
#         VD,
#         BLOCK_H=BLOCK_H,
#         **kwargs
#     )

#     kwargs = {"num_warps": 1, "num_stages": 3}
#     grid = lambda meta: (B, QH)
#     _decode_stage2_kernel[grid](
#         o, 
#         lse,
#         context_lens, 
#         *o.stride(),
#         *lse.stride(),
#         kernel_size,
#         stride,
#         VD,
#         BLOCK_M=BLOCK_M,
#         NUM_SPLITS=NUM_SPLITS,
#         **kwargs
#     )
#     return o[0], lse[0]


