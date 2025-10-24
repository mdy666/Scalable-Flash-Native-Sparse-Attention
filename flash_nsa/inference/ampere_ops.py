# Copyright (c) 2025 Duyue Ma

import math

import torch
import triton
import triton.language as tl

from .utils import split_d

# @triton.autotune([triton.Config({'BLOCK_N': bsn, 'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['D1', "D2", "VD"])
@triton.jit
def _cmp_prefill_kernel(
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

    desc_q = tl.make_block_ptr(Q, (x_len, D1), (q_stride_n, q_stride_d), (start_n, 0), (BLOCK_N, D1), (1, 0))
    desc_o = tl.make_block_ptr(O, (x_len, VD), (o_stride_n, o_stride_d), (start_n, 0), (BLOCK_N, VD), (1, 0))
    if D2 > 0:
        desc_q2 = tl.make_block_ptr(Q + D1, (x_len, D2), (q_stride_n, q_stride_d), (start_n, 0), (BLOCK_N, D1), (1, 0))
    q = tl.load(desc_q, boundary_check=(0, 1))
    if D2 > 0:
        q2 = tl.load(desc_q2, boundary_check=(0, 1))

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
        desc_k = tl.make_block_ptr(K, (num_blocks * PAGE_SIZE, D1), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D1), (1, 0))
        desc_v = tl.make_block_ptr(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, VD), (1, 0))
        k = tl.load(desc_k, boundary_check=(0, 1))
        v = tl.load(desc_v, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            desc_k2 = tl.make_block_ptr(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D2), (1, 0))
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

    for i in range(mid, end, BLOCK_M):
        offset = i % PAGE_SIZE
        table_idx = i // PAGE_SIZE
        cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
        desc_k = tl.make_block_ptr(K, (num_blocks * PAGE_SIZE, D1), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D1), (1, 0))
        desc_v = tl.make_block_ptr(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, VD), (1, 0))
        k = tl.load(desc_k, boundary_check=(0, 1))
        v = tl.load(desc_v, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            desc_k2 = tl.make_block_ptr(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D2), (1, 0))
            k2 = tl.load(desc_k2, boundary_check=(0, 1))
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
    tl.store(desc_o, acc.to(desc_o.type.element_ty), boundary_check=(0, 1))
    tl.store(LSE + off_n * lse_stride_n, lse, mask=off_n < x_len)
    

# @triton.autotune([triton.Config({'BLOCK_M': bsm, "NUM_SPLITS": num_splits}, num_stages=ns, num_warps=nw)
#                 for num_splits in [2, 4, 8]
#                  for bsm in [64]
#                  for ns in [2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['D1', "D2", "BLOCK_H"])
@triton.jit
def _cmp_decode_stage1_kernel(
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

        mask = tl.arange(0, BLOCK_H) < G2
        q = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D1)[None, :], mask[:, None])
        if D2 > 0:
            q2 = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, mask[:, None])

        sm_scale *= 1.44269504
        m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, VD], dtype=tl.float32)

        for i in range(kv_start, kv_mid, BLOCK_M):
            offset = (i % PAGE_SIZE)
            table_idx = i // PAGE_SIZE
            cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
            desc_k = tl.make_block_ptr(K, (num_blocks * PAGE_SIZE, D1), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D1), (1, 0))
            desc_v = tl.make_block_ptr(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, VD), (1, 0))
            k = tl.load(desc_k, boundary_check=(0, 1))
            v = tl.load(desc_v, boundary_check=(0, 1))
            attn_score = tl.dot(q, tl.permute(k, 1, 0))
            if D2>0:
                desc_k2 = tl.make_block_ptr(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D2), (1, 0))
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

        for i in range(kv_mid, kv_end, BLOCK_M):
            offset = (i % PAGE_SIZE)
            table_idx = i // PAGE_SIZE
            cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
            desc_k = tl.make_block_ptr(K, (num_blocks * PAGE_SIZE, D1), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D1), (1, 0))
            desc_v = tl.make_block_ptr(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, VD), (1, 0))
            k = tl.load(desc_k, boundary_check=(0, 1))
            v = tl.load(desc_v, boundary_check=(0, 1))
            attn_score = tl.dot(q, tl.permute(k, 1, 0))
            if D2>0:
                desc_k2 = tl.make_block_ptr(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D2), (1, 0))
                k2 = tl.load(desc_k2, boundary_check=(0, 1))
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
            lse = tl.zeros_like(lse)
        tl.store(O + tl.arange(0, BLOCK_H)[:, None] * o_stride_h + tl.arange(0, VD)[None, :], acc, mask[:, None])
        tl.store(LSE + tl.arange(0, BLOCK_H) * lse_stride_h, lse, mask)
        
        
        
# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                  for ns in [1, 2, 3, 4]
#                  for nw in [1, 2, 4, 8]
#                  ], key=['D1', "D2", "VD"])
@triton.jit
def _cmp_decode_stage2_kernel(
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
        kwargs = {"BLOCK_N": 128, "BLOCK_M": 64, "num_warps": 8, "num_stages": 3}
    else:
        kwargs = {"BLOCK_N": 128, "BLOCK_M": 64, "num_warps": 4, "num_stages": 1}
    grid = lambda meta: (triton.cdiv(x_maxlen, meta['BLOCK_N']), B, QH)
    _cmp_prefill_kernel[grid](
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
    BLOCK_M = 64
    kwargs = {"BLOCK_M": BLOCK_M, "NUM_SPLITS":NUM_SPLITS, "num_warps": 2, "num_stages": 2}

    G2 = min(G, BLOCK_H)
    grid = lambda meta: (B, QH // G2, max_num_splits)
    _cmp_decode_stage1_kernel[grid](
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

    kwargs = {"num_warps": 1, "num_stages": 3}
    grid = lambda meta: (B, QH)
    _cmp_decode_stage2_kernel[grid](
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
    return out, lse[0]


# # @triton.autotune([triton.Config({'BLOCK_M': bsm, 'BLOCK_N': bsn}, num_stages=ns, num_warps=nw)
# #                  for bsm in [64, 128]
# #                  for bsn in [64, 128]
# #                  for ns in [1, 2,3, 4]
# #                  for nw in [4, 8]
#                  ], key=['D1', "D2"])
@triton.jit
def _attn_probs_prefill_kernel(
    Q, 
    K, 
    Lse, 
    P,
    X_CU_SEQLENS, 
    CONTEXT_LENS,
    TABLES,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    lse_stride_h, lse_stride_n,
    p_stride_h, p_stride_n, p_stride_m,
    table_stride,
    sm_scale, 
    kernel_size, 
    stride,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    KH: tl.constexpr, 
    G: tl.constexpr, 
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    BLOCK_N: tl.constexpr=64, 
    BLOCK_M: tl.constexpr=64
):
    start_n = tl.program_id(2) * BLOCK_N
    start_m = tl.program_id(1) * BLOCK_M
    off_bh = tl.program_id(0)
    off_kh = off_bh % KH
    off_b = off_bh // KH

    x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
    x_len = x_eos - x_bos
    if (start_n >= x_len):
        return
    cached_tokens = tl.load(CONTEXT_LENS + off_b) - x_len
    if x_len + cached_tokens <= kernel_size - 1:
        return
    y_len = tl.maximum((x_len + cached_tokens - kernel_size) // stride + 1, 0)

    if (start_n + cached_tokens + BLOCK_N) < (start_m * stride + kernel_size):
        return  
    
    Q += x_bos * q_stride_n
    K += off_kh * k_stride_h
    Lse += x_bos * lse_stride_n
    P += off_kh * p_stride_h.to(tl.int64) + x_bos * p_stride_n.to(tl.int64)

    off_n = start_n + tl.arange(0, BLOCK_N)

    offset = start_m % PAGE_SIZE
    table_idx = start_m // PAGE_SIZE
    cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
    
    desc_p = tl.make_block_ptr(P, (x_len, y_len), (p_stride_n, p_stride_m), (start_n, start_m), (BLOCK_N, BLOCK_M), (1,0))
    desc_k = tl.make_block_ptr(K, (num_blocks * PAGE_SIZE, D1), (k_stride_m, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D1), (1,0))
    k = tl.load(desc_k, boundary_check=(0, 1))
    if D2 > 0:
        desc_q2 = tl.make_block_ptr(Q + D1, (G * KH, x_len, D2), (q_stride_h, q_stride_n, q_stride_d), (1, BLOCK_N, D2), (1,0))
        desc_k2 = tl.make_block_ptr(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_m, k_stride_d),(cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D2), (1,0))
        k2 = tl.load(desc_k2, boundary_check=(0, 1))

    sm_scale *= 1.44269504
    p = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for off_qh in range(off_kh * G, off_kh * G + G):
        lse = tl.load(Lse + off_qh * lse_stride_h + off_n * lse_stride_n, mask=off_n < x_len, other=0.)
        desc_q = tl.make_block_ptr(Q + off_qh * q_stride_h, (x_len, D1), (q_stride_n, q_stride_d), (start_n, 0), (BLOCK_N, D1), (1,0))
        q = tl.load(desc_q, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            desc_q2 = tl.make_block_ptr(Q + off_qh * q_stride_h + D1, (x_len, D2), (q_stride_n, q_stride_d), (start_n, 0), (BLOCK_N, D2), (1,0))
            q2 = tl.load(desc_q2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        p += tl.exp2(tl.fma(attn_score, sm_scale,  -lse[:, None]))
        # p += tl.exp2(attn_score * sm_scale -lse[:, None])

    # if (start_n + cached_tokens) < ((start_m + BLOCK_M - 1) * stride + kernel_size - 1):
    #     k_idx = (start_m + tl.arange(0, BLOCK_M)) * stride + kernel_size - 1
    #     causal_mask = q_idx[:, None] >= k_idx[None, :]
    #     p = tl.where(causal_mask, p, 0.)
    tl.store(desc_p, p.to(desc_p.type.element_ty), boundary_check=(0, 1))

# @triton.autotune([triton.Config({'BLOCK_N': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [2, 4, 8, 16, 32]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=["BLOCK_K"])
@triton.jit
def _topk_prefill_kernel(
    AP, 
    Ind,
    X_CU_SEQLENS,
    CONTEXT_LENS,
    ap_stride_h, ap_stride_n, ap_stride_m,
    ind_stride_h, ind_stride_n, ind_stride_k,
    kernel_size, 
    stride, 
    block_size, 
    top_n, 
    num_inital:tl.constexpr, 
    num_local:tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr=16,
):
    start_n = tl.program_id(0) * BLOCK_N
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
    x_len = x_eos - x_bos
    if start_n >= x_len:
        return

    cached_tokens = tl.load(CONTEXT_LENS + off_b) - x_len
    # y_len = tl.maximum((x_len + cached_tokens - kernel_size) // stride + 1, 0)

    AP += off_h * ap_stride_h.to(tl.int64) + x_bos * ap_stride_n.to(tl.int64)
    Ind += off_h * ind_stride_h + x_bos * ind_stride_n

    acc_p = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    select_idx = tl.arange(0, BLOCK_K)

    off_n = start_n + tl.arange(0, BLOCK_N)
    q_idx = off_n + cached_tokens
    x_mask = off_n < x_len

    select_start = 0
    select_end = block_size
    compress_start = stride - kernel_size 
    # num_loops = (block_size + 2 * (kernel_size - stride) - kernel_size) // stride + 1
    num_loops = (block_size + kernel_size - stride) // stride
    compress_idx = (select_idx * block_size - kernel_size) // stride + 1
    end_len = (start_n + BLOCK_N + cached_tokens - kernel_size) // stride + 1
    for _ in range(num_loops):
        compress_end = compress_start + kernel_size
        area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
        w = area / stride
        y_mask = (compress_idx >= 0) & (compress_idx < end_len)
        p = tl.load(AP + off_n[:, None] * ap_stride_n.to(tl.int64) + compress_idx[None, :] * ap_stride_m, 
                    mask=x_mask[:, None] & y_mask[None, :], other=0.) * w
        acc_p += p
        compress_idx += 1
        compress_start += stride


    top_n = tl.minimum(top_n, (start_n + cached_tokens + BLOCK_N - 1) // block_size + 1)

    num_k = q_idx // block_size
    for i in range(0, num_inital):
        tl.store(Ind + off_n * ind_stride_n + i, i, mask=x_mask & (i <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K)[None, :] == i,
                        -1., acc_p)

    for i in range(0, num_local):
        tl.store(Ind + off_n * ind_stride_n + i + num_inital, num_k - i, mask=x_mask & (i + num_inital <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K)[None, :] == (num_k - i)[:, None],
                        -1., acc_p)

    for i in range(num_inital+num_local, top_n):
        max_idx = tl.argmax(acc_p, axis=-1)
        tl.store(Ind + off_n * ind_stride_n + i * ind_stride_k, max_idx, mask=x_mask & (i <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K)[None, :] == max_idx[:, None],
                    -1., acc_p)
          
@torch.no_grad()
def topk_prefill(
    q: torch.Tensor, 
    k: torch.Tensor, 
    lse: torch.Tensor,
    x_cu_seqlens,
    x_maxlen,
    y_maxlen,
    block_tables,
    context_lens,
    fixed_y_maxlen: int = 8192,
    fixed_num_slc_blocks: int = 2048,
    kernel_size: int = 32,
    stride: int = 16,
    block_size: int = 64,
    top_n:int = 16,
    num_inital: int = 1, 
    num_local: int = 2,
    sm_scale: float = None, 
    fp32: bool = False, 

) -> tuple[torch.Tensor, torch.Tensor]:

    T, QH, D = q.shape
    num_blocks, PAGE_SIZE, KH, _ = k.shape
    D1, D2 = split_d(D)
    G = QH // KH
    B = len(x_cu_seqlens) - 1
    if sm_scale is None:
        sm_scale = D**-0.5
        
    k = k.view(-1, KH, D)

    pad_y_maxlen = max(triton.next_power_of_2(y_maxlen), 8)
    attn_probs = torch.empty(KH, T, pad_y_maxlen, device=q.device, dtype=torch.float16 if not fp32 else torch.float)
    if D <= 128:
        kwargs = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}
    else:
        kwargs = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}
    grid = lambda meta: (B * KH, triton.cdiv(y_maxlen, meta['BLOCK_M']), triton.cdiv(x_maxlen, meta['BLOCK_N']))
    _attn_probs_prefill_kernel[grid](
        q, 
        k, 
        lse, 
        attn_probs,
        x_cu_seqlens,
        context_lens,
        block_tables,
        *q.stride(),
        *k.stride(),
        *lse.stride(),
        *attn_probs.stride(),
        block_tables.stride(0),
        sm_scale, 
        kernel_size, 
        stride,
        num_blocks,
        PAGE_SIZE,
        KH, 
        G,
        D1, 
        D2,
        **kwargs
    )

    ignore_index = -1
    BLOCK_K = max(triton.next_power_of_2(triton.cdiv(pad_y_maxlen * stride, block_size)), 16)

    topk_indices = torch.full((KH, T, top_n), ignore_index, dtype=torch.int32, device=q.device)
    BLOCK_N = 8
    if BLOCK_K >= 1024:
        BLOCK_N = 4
    elif BLOCK_K >= 2048:
        BLOCK_N = 1

    grid=lambda meta: (triton.cdiv(x_maxlen, meta['BLOCK_N']), B, KH)
    kwargs = {"BLOCK_N": BLOCK_N, "num_warps": 4, "num_stages": 2}
    _topk_prefill_kernel[grid](
        attn_probs, 
        topk_indices,
        x_cu_seqlens,
        context_lens,
        *attn_probs.stride(),
        *topk_indices.stride(),
        kernel_size, 
        stride, 
        block_size, 
        top_n, 
        num_inital, 
        num_local,
        BLOCK_K,
        **kwargs
    )
    return topk_indices


# @triton.autotune([triton.Config({'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [1, 2, 4, 8]
#                  ], key=['D1', "D2"])
@triton.jit
def _attn_probs_decode_kernel(
    Q, 
    K, 
    Lse, 
    P,
    CONTEXT_LENS,
    TABLES,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_m, k_stride_h, k_stride_d,
    lse_stride_n, lse_stride_h,
    p_stride_h, p_stride_n, p_stride_m,
    table_stride,
    sm_scale, 
    kernel_size, 
    stride,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    G: tl.constexpr, 
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    BLOCK_H: tl.constexpr=16, 
    BLOCK_M: tl.constexpr=64
):
    off_b = tl.program_id(2)
    start_m = tl.program_id(0) * BLOCK_M
    off_kh = tl.program_id(1)
    start_qh = off_kh * G

    x_len = tl.load(CONTEXT_LENS + off_b)
    if x_len <= kernel_size - 1:
        return
    y_len = tl.maximum((x_len - kernel_size) // stride + 1, 0)
    if start_m >= y_len:
        return

    Q += off_b * q_stride_n + start_qh * q_stride_h
    K += off_kh * k_stride_h
    Lse += off_b * lse_stride_n + start_qh * lse_stride_h
    P += off_kh * p_stride_h + off_b * p_stride_n

    offset = start_m % PAGE_SIZE
    table_idx = start_m // PAGE_SIZE
    cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
    sm_scale *= 1.44269504
    lse = tl.load(Lse + tl.arange(0, BLOCK_H) * lse_stride_h, mask=tl.arange(0, BLOCK_H) < G)
    q = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D1)[None, :], mask=tl.arange(0, BLOCK_H)[:, None] < G)
    desc_k = tl.make_block_ptr(K, (num_blocks * PAGE_SIZE, D1), (k_stride_m, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0),  (BLOCK_M, D1), (1, 0))
    k = tl.load(desc_k, boundary_check=(0, 1))
    attn_score = tl.dot(q, tl.permute(k, 1, 0))
    if D2 > 0:
        q2 = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, tl.arange(0, BLOCK_H)[:, None] < G)
        desc_k2 = tl.make_block_ptr(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_m, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D2), (1, 0))
        k2 = tl.load(desc_k2, boundary_check=(0, 1))
        attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
    p = tl.sum(tl.exp2(tl.fma(attn_score, sm_scale, -lse[:, None])), 0)
    tl.store(P + start_m + tl.arange(0, BLOCK_M), p, mask=(start_m + tl.arange(0, BLOCK_M)) < y_len)
    
# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                  for ns in [1, 2, 3, 4]
#                  for nw in [1, 2, 4, 8]
#                  ], key=["BLOCK_K"])
@triton.jit
def _topk_decode_kernel(
    AP, 
    Ind,
    CONTEXT_LENS,
    ap_stride_h, ap_stride_n, ap_stride_m,
    ind_stride_h, ind_stride_n, ind_stride_k,
    kernel_size, 
    stride, 
    block_size, 
    top_n, 
    num_inital:tl.constexpr, 
    num_local:tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    x_len = tl.load(CONTEXT_LENS + off_b)
    y_len = tl.maximum((x_len - kernel_size) // stride + 1, 0)

    AP += off_h * ap_stride_h + off_b * ap_stride_n
    Ind += off_h * ind_stride_h + off_b * ind_stride_n
    
    acc_p = tl.zeros((BLOCK_K, ), dtype=tl.float32)
    select_idx = tl.arange(0, BLOCK_K)

    q_idx = x_len - 1

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
        y_mask = (compress_idx >= 0) & (compress_idx < y_len)
        p = tl.load(AP + compress_idx * ap_stride_m, mask=y_mask, other=0.) * w
        acc_p += p
        compress_idx += 1
        compress_start += stride

    top_n = tl.minimum(top_n, q_idx // block_size + 1)

    num_k = q_idx // block_size
    for i in range(0, num_inital):
        tl.store(Ind + i, i, mask=(i <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K) == i,
                        -1., acc_p)

    for i in range(0, num_local):
        tl.store(Ind + i + num_inital, num_k - i, mask=(i + num_inital <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K) == (num_k - i),
                        -1., acc_p)

    for i in range(num_inital + num_local, top_n):
        max_idx = tl.argmax(acc_p, 0)
        tl.store(Ind + i, max_idx, mask=(i <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K) == max_idx,
                    -1., acc_p)
         
def topk_decode(
    q: torch.Tensor, 
    k: torch.Tensor, 
    lse: torch.Tensor,
    block_tables,
    context_lens,
    kernel_size: int = 32,
    stride: int = 16,
    block_size: int = 64,
    top_n:int = 16,
    num_inital: int = 1, 
    num_local: int = 2,
    fixed_num_slc_blocks: int = 2048,
    fixed_y_maxlen: int = 8192,
    sm_scale: float = None, 
    persistent: bool = False,
    workers: int = 4,

) -> tuple[torch.Tensor, torch.Tensor]:

    B, QH, D = q.shape
    num_blocks, PAGE_SIZE, KH, _ = k.shape
    D1, D2 = split_d(D)
    G = QH // KH
    if sm_scale is None:
        sm_scale = D**-0.5
        
    k = k.view(-1, KH, D)

    BLOCK_H = max(triton.next_power_of_2(G), 16)
    fixed_y_maxlen = triton.cdiv(fixed_y_maxlen, 8) * 8
    attn_probs = torch.empty(KH, B, fixed_y_maxlen, device=q.device, dtype=torch.float32)

    kwargs = {"BLOCK_M": 64, "num_warps": 4, "num_stages": 1}
    grid = lambda meta: (triton.cdiv(fixed_y_maxlen, meta['BLOCK_M']), KH, B)
    _attn_probs_decode_kernel[grid](
        q, 
        k, 
        lse, 
        attn_probs,
        context_lens,
        block_tables,
        *q.stride(),
        *k.stride(),
        *lse.stride(),
        *attn_probs.stride(),
        block_tables.stride(0),
        sm_scale, 
        kernel_size, 
        stride,
        num_blocks,
        PAGE_SIZE,
        G,
        D1, 
        D2,
        BLOCK_H=BLOCK_H,
        **kwargs
    )

    ignore_index = -1
    BLOCK_K = triton.next_power_of_2(fixed_num_slc_blocks)

    topk_indices = torch.full((KH, B, top_n), ignore_index, dtype=torch.int32, device=q.device)

    grid=lambda meta: (B, KH)
    # kwargs = {"num_warps": 4, "num_stages": 2}
    kwargs = {"num_warps": 8, "num_stages": 2}
    _topk_decode_kernel[grid](
        attn_probs, 
        topk_indices,
        context_lens,
        *attn_probs.stride(),
        *topk_indices.stride(),
        kernel_size, 
        stride, 
        block_size, 
        top_n, 
        num_inital, 
        num_local,
        BLOCK_K,
        **kwargs
    )
    return topk_indices

# @triton.autotune([triton.Config({}, num_warps=nw, num_stages=ns)
#                  for nw in [1, 2, 4, 8]
#                  for ns in [1,2,3,4]
#                  ], key=["D1", "D2", "VD" 'BLOCK_H', 'BLOCK_M'])
@triton.jit
def _slc_prefill_kernel(
    Q,
    K,
    V,
    O,
    Lse, 
    Ind,
    X_CU_SEQLENS, 
    CONTEXT_LENS,
    TABLES,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_n, k_stride_h, k_stride_d,
    v_stride_n, v_stride_h, v_stride_d,
    o_stride_n, o_stride_h, o_stride_d,
    lse_stride_n, lse_stride_h,
    ind_stride_h, ind_stride_n, ind_stride_k,
    table_stride,
    sm_scale, 
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    top_n,
    KH: tl.constexpr,
    G: tl.constexpr,
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    BLOCK_H: tl.constexpr=16, 
    BLOCK_M: tl.constexpr=64,
):
    off_n = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // KH
    off_kh = off_bh % KH
    start_qh = off_kh * G

    x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
    x_len = x_eos - x_bos
    if off_n >= x_len:
        return

    cached_tokens = tl.load(CONTEXT_LENS + off_b) - x_len
    q_idx = off_n + cached_tokens


    Q += start_qh * q_stride_h + (x_bos + off_n) * q_stride_n
    O += start_qh * o_stride_h + (x_bos + off_n) * o_stride_n
    K += off_kh * k_stride_h
    V += off_kh * v_stride_h
    Ind += off_kh * ind_stride_h + (x_bos + off_n) * ind_stride_n
    Lse += (x_bos + off_n) * lse_stride_n + start_qh * lse_stride_h

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
        offset = start_m % PAGE_SIZE
        table_idx = start_m // PAGE_SIZE
        cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
        desc_k = tl.make_block_ptr(K, (num_blocks * PAGE_SIZE, D1), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D1), (1, 0))
        desc_v = tl.make_block_ptr(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, VD), (1, 0))
        k = tl.load(desc_k, boundary_check=(0, 1))
        v = tl.load(desc_v, boundary_check=(0, 1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            desc_k2 = tl.make_block_ptr(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D2), (1, 0))
            k2 = tl.load(desc_k2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        desc_v = tl.make_block_ptr(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, VD), (1, 0))
        v = tl.load(desc_v, boundary_check=(0, 1))
        attn_score = tl.where(q_idx >= (start_m + tl.arange(0, BLOCK_M))[None, :], attn_score * sm_scale, float('-inf'))
        new_m_i = tl.maximum(m_i, tl.max(attn_score, axis=1))
        alpha = tl.exp2(m_i - new_m_i)
        exp_attn_score = tl.exp2(attn_score - new_m_i[:, None])
        l_i = tl.fma(l_i, alpha, tl.sum(exp_attn_score, axis=-1))
        acc = acc * alpha[:, None]
        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)
        # acc = tl.fma(acc, alpha[:, None], tl.dot(exp_attn_score.to(v.dtype), v))
        m_i = new_m_i


    acc /= l_i[:, None]
    lse = m_i + tl.log2(l_i)

    tl.store(desc_o, acc.to(desc_o.type.element_ty), boundary_check=(0, 1))
    tl.store(Lse + tl.arange(0, BLOCK_H), lse, mask=tl.arange(0, BLOCK_H) < G)

def slc_attn_prefill(q, k, v, topk, x_cu_seqlens, x_maxlen, block_tables, context_lens, block_size=64, top_n=16, sm_scale=None):
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

    o = torch.empty(T, QH, VD, device=q.device, dtype=q.dtype)
    lse = torch.empty(T, QH, dtype=torch.float32, device=q.device,)

    BLOCK_H = max(triton.next_power_of_2(G), 16)
    BLOCK_M = block_size

    if D <= 128:
        kwargs = {"num_warps": 4, "num_stages": 2}
    else:
        kwargs = {"num_warps": 1, "num_stages": 1}
    grid = lambda meta: (x_maxlen, B * KH)
    _slc_prefill_kernel[grid](
        q,
        k,
        v,
        o,
        lse, 
        topk,
        x_cu_seqlens,
        context_lens,
        block_tables,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        *lse.stride(),
        *topk.stride(),
        block_tables.stride(0),
        sm_scale,
        num_blocks,
        PAGE_SIZE,
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
        
# @triton.autotune([triton.Config({"NUM_SPLITS": num_splits}, num_stages=ns, num_warps=nw)
#                 for num_splits in [1, 2, 4]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [1, 2, 4, 8]
#                  ], key=['D1', "D2", "BLOCK_H", "VD"])
@triton.jit
def _fused_decode_stage1_kernel(
    Q, 
    K, 
    V, 
    O, 
    LSE, 
    Ind,
    CONTEXT_LENS,
    TABLES,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_n, k_stride_h, k_stride_d,
    v_stride_n, v_stride_h, v_stride_d,
    o_stride_a, o_stride_k, o_stride_n, o_stride_h, o_stride_d,
    lse_stride_a, lse_stride_k, lse_stride_n, lse_stride_h,
    ind_stride_h, ind_stride_n, ind_stride_k,
    table_stride,
    sm_scale, 
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    top_n,
    G: tl.constexpr,
    G2: tl.constexpr,
    D1: tl.constexpr, 
    D2: tl.constexpr, 
    VD: tl.constexpr, 
    WINDOW_SIZE: tl.constexpr=512,
    NUM_SPLITS: tl.constexpr=1,
    BLOCK_H: tl.constexpr=16, 
    BLOCK_M: tl.constexpr=64,
):
    off_b = tl.program_id(0)
    start_qh = tl.program_id(1) * G2
    off_k = tl.program_id(2)

    off_kh = start_qh // G

    x_len = tl.load(CONTEXT_LENS + off_b)
    q_idx = x_len - 1
    
    Q += off_b * q_stride_n + start_qh * q_stride_h
    K += off_kh * k_stride_h
    V += off_kh * v_stride_h
    O += off_k * o_stride_k + off_b * o_stride_n + start_qh * o_stride_h
    LSE += off_k * lse_stride_k + start_qh * lse_stride_h + off_b * lse_stride_n
    Ind += off_kh * ind_stride_h + off_b * ind_stride_n

    q = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D1)[None, :], tl.arange(0, BLOCK_H)[:, None] < G2)
    if D2 > 0:
        q2 = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, tl.arange(0, BLOCK_H)[:, None] < G2)
        desc_k2 = tl.make_block_ptr(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (BLOCK_M, D2), (1, 0))

    sm_scale *= 1.44269504
    if NUM_SPLITS != 0:
        num_splits = NUM_SPLITS
    else:
        num_splits = 4

    left = tl.maximum(q_idx - WINDOW_SIZE, 0)
    left = (left // BLOCK_M) * BLOCK_M
    num_tiles = tl.cdiv(x_len - left, BLOCK_M)

    if off_k < num_tiles:
        m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, VD], dtype=tl.float32)

        for i in range(off_k, num_tiles, num_splits):
            start_m = left + i * BLOCK_M
            offset = start_m % PAGE_SIZE
            table_idx = start_m // PAGE_SIZE
            cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
            desc_k = tl.make_block_ptr(K, (num_blocks * PAGE_SIZE, D1), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D1), (1, 0))
            desc_v = tl.make_block_ptr(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, VD), (1, 0))
            k = tl.load(desc_k, boundary_check=(0, 1))
            v = tl.load(desc_v, boundary_check=(0, 1))
            attn_score = tl.dot(q, tl.permute(k, 1, 0))
            if D2>0:
                desc_k2 = tl.make_block_ptr(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D2), (1, 0))
                k2 = tl.load(desc_k2, boundary_check=(0, 1))
                attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
            k_idx = start_m + tl.arange(0, BLOCK_M)
            mask = ((q_idx - WINDOW_SIZE) <= k_idx[None, :]) & (q_idx >= k_idx[None, :])
            # mask = (q_idx >= k_idx[None, :])
            attn_score = tl.where(mask, attn_score * sm_scale, float('-inf'))
            m_ij = tl.max(attn_score, axis=1)
            new_m_i = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - new_m_i)

            exp_attn_score = tl.exp2(attn_score - new_m_i[:, None])

            l_i = tl.fma(l_i, alpha, tl.sum(exp_attn_score, axis=-1))
            acc = tl.fma(acc, alpha[:, None], tl.dot(exp_attn_score.to(v.dtype), v))
            m_i = new_m_i
            
        acc /= l_i[:, None]
        lse = m_i + tl.log2(l_i)
        tl.store(O + o_stride_a + tl.arange(0, BLOCK_H)[:, None] * o_stride_h + tl.arange(0, D1)[None, :], acc, tl.arange(0, BLOCK_H)[:, None] < G2)
        tl.store(LSE + lse_stride_a + tl.arange(0, BLOCK_H) * lse_stride_h, lse, mask=tl.arange(0, BLOCK_H) < G2)

    top_n = tl.minimum(top_n, tl.cdiv(x_len, BLOCK_M))
    split_size = tl.cdiv(top_n, num_splits)
    start_idx = split_size * off_k
    end_idx = tl.minimum(top_n, start_idx + split_size)

    if start_idx < top_n:
        m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, VD], dtype=tl.float32)
        for i in range(start_idx, end_idx):
            start_m = tl.load(Ind + i) * BLOCK_M
            offset = start_m % PAGE_SIZE
            table_idx = start_m // PAGE_SIZE
            cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
            desc_k = tl.make_block_ptr(K, (num_blocks * PAGE_SIZE, D1), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D1), (1, 0))
            desc_v = tl.make_block_ptr(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, VD), (1, 0))
            k = tl.load(desc_k, boundary_check=(0, 1))
            v = tl.load(desc_v, boundary_check=(0, 1))
            attn_score = tl.dot(q, tl.permute(k, 1, 0))
            if D2>0:
                desc_k2 = tl.make_block_ptr(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (cache_idx * PAGE_SIZE + offset, 0), (BLOCK_M, D2), (1, 0))
                k2 = tl.load(desc_k2, boundary_check=(0, 1))
                attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)

            attn_score = tl.where(q_idx >= (start_m + tl.arange(0, BLOCK_M))[None, :], attn_score * sm_scale, float('-inf'))

            m_ij = tl.max(attn_score, axis=1)
            new_m_i = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - new_m_i)

            exp_attn_score = tl.exp2(attn_score - new_m_i[:, None])

            l_i = tl.fma(l_i, alpha, tl.sum(exp_attn_score, axis=-1))
            acc = tl.fma(acc, alpha[:, None], tl.dot(exp_attn_score.to(v.dtype), v))
            m_i = new_m_i

        acc /= l_i[:, None]
        lse = m_i + tl.log2(l_i)

        tl.store(O + tl.arange(0, BLOCK_H)[:, None] * o_stride_h + tl.arange(0, D1)[None, :], acc, tl.arange(0, BLOCK_H)[:, None] < G2)
        tl.store(LSE + tl.arange(0, BLOCK_H) * lse_stride_h, lse, mask=tl.arange(0, BLOCK_H) < G2)

# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                  for ns in [1, 2, 3, 4]
#                  for nw in [1, 2, 4, 8]
#                  ], key=[ "VD", "NUM_SPLITS"])
@triton.jit
def _fused_decode_stage2_kernel(
    MID_O,
    O,
    LSE, 
    CONTEXT_LENS,
    mo_stride_a, mo_stride_k, mo_stride_n, mo_stride_h, mo_stride_d,
    o_stride_a, o_stride_n, o_stride_h, o_stride_d,
    lse_stride_a, lse_stride_k, lse_stride_n, lse_stride_h,
    top_n,
    VD: tl.constexpr=128, 
    BLOCK_M: tl.constexpr=64,
    NUM_SPLITS: tl.constexpr=8,
    WINDOW_SIZE: tl.constexpr=512,
):

    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    pid2 = tl.program_id(2)
    off_d = tl.arange(0, VD)

    x_len = tl.load(CONTEXT_LENS + off_b)
    if NUM_SPLITS != 0:
        num_splits = NUM_SPLITS
    else:
        num_splits = 4

    if pid2 == 0:
        top_n = tl.minimum(top_n, tl.cdiv(x_len, BLOCK_M))
        split_size = tl.cdiv(top_n, num_splits)
        num_loops = tl.minimum(num_splits, tl.cdiv(top_n, split_size))
    else:
        left = tl.maximum(x_len - 1 - WINDOW_SIZE, 0)
        left = left // BLOCK_M * BLOCK_M
        num_loops = tl.minimum(num_splits, tl.cdiv(x_len - left, BLOCK_M))
    
    o_ptrs = O + pid2 * o_stride_a + off_b * o_stride_n + off_h * o_stride_h + off_d
    mid_o_ptrs = MID_O + pid2 * mo_stride_a + off_b * mo_stride_n + off_h * mo_stride_h + off_d
    lse_ptrs = LSE + pid2 * lse_stride_a + off_b * lse_stride_n + off_h * lse_stride_h
    
    m_i = float("-inf")
    l_i = 0.
    acc = tl.zeros([VD], dtype=tl.float32)
    off_k = 0
    for _ in range(0, num_loops):
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

def fused_slc_swa_attn_decode(q, k, v, topk, block_tables, context_lens, out=None, num_splits=0, max_num_splits=4, block_size=64, top_n=16, window_size=512, sm_scale=None, bench=False):
    B, QH, D = q.shape
    num_blocks, PAGE_SIZE, KH, _ = k.shape
    VD = v.size(-1)
    D1, D2 = split_d(D)
    G = QH // KH
    if sm_scale is None:
        sm_scale = D**-0.5
    assert num_splits >= 0
    assert num_splits <= max_num_splits
        
    k = k.view(-1, KH, D)
    v = v.view(-1, KH, VD)

    mid_o = torch.empty(2, max_num_splits, B, QH, VD, device=q.device, dtype=torch.float32)
    lse = torch.empty(2, max_num_splits, B, QH, dtype=torch.float32, device=q.device)
    if out is None:
        out = torch.empty(2, B, QH, VD, device=q.device, dtype=q.dtype)

    BLOCK_H = max(triton.next_power_of_2(G), 16)
    G2 = min(BLOCK_H, G)
    BLOCK_M = block_size

    if D <= 128:
        kwargs = {"num_warps": 2, "num_stages": 2}
    else:
        kwargs = {"num_warps": 1, "num_stages": 1}
    
    NUM_SPLITS = num_splits
    grid = lambda meta: (B, QH // G2, max_num_splits)
    _fused_decode_stage1_kernel[grid](
        q,
        k,
        v,
        mid_o,
        lse, 
        topk,
        context_lens,
        block_tables,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *mid_o.stride(),
        *lse.stride(),
        *topk.stride(),
        block_tables.stride(0),
        sm_scale,
        num_blocks,
        PAGE_SIZE,
        top_n,
        G,
        G2,
        D1, 
        D2,
        VD,
        BLOCK_H=BLOCK_H,
        BLOCK_M=BLOCK_M,
        WINDOW_SIZE=window_size,
        NUM_SPLITS=NUM_SPLITS,
        **kwargs
    )

    kwargs = {"num_warps": 1, "num_stages": 3}
    grid = lambda meta: (B, QH, 2)
    _fused_decode_stage2_kernel[grid](
        mid_o,
        out,
        lse,
        context_lens, 
        *mid_o.stride(),
        *out.stride(),
        *lse.stride(),
        top_n,
        VD,
        BLOCK_M=BLOCK_M,
        NUM_SPLITS=NUM_SPLITS,
        WINDOW_SIZE=window_size,
        **kwargs
    )
    return out[0], out[1]

