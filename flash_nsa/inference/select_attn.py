# Copyright (c) 2025 Duyue Ma

import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .utils import split_d, use_tma


# @triton.autotune([triton.Config({}, num_warps=nw, num_stages=ns)
#                  for nw in [1, 2, 4, 8]
#                  for ns in [1,2,3,4]
#                  ], key=["D1", "D2", "VD" 'BLOCK_H', 'BLOCK_M'])
@triton.jit
def _prefill_kernel(
    desc_q,
    desc_q2, 
    desc_k,
    desc_k2,
    desc_v,
    desc_o, 
    Lse, 
    Ind,
    X_CU_SEQLENS, 
    CONTEXT_LENS,
    TABLES,
    lse_stride_n, lse_stride_h,
    ind_stride_h, ind_stride_n, ind_stride_k,
    table_stride,
    sm_scale, 
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

    Ind += off_kh * ind_stride_h + (x_bos + off_n) * ind_stride_n
    Lse += (x_bos + off_n) * lse_stride_n + start_qh * lse_stride_h

    q = desc_q.load([x_bos + off_n, off_kh, 0, 0]).reshape(BLOCK_H, D1)
    if D2 > 0:
        q2 = desc_q2.load([x_bos + off_n, off_kh, 0, D1]).reshape(BLOCK_H, D2)

    sm_scale *= 1.44269504
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, VD], dtype=tl.float32)

    stop_n = tl.constexpr(tl.minimum(top_n, tl.cdiv(q_idx+1, BLOCK_M)))
    for i in tl.range(0, stop_n, flatten=True):
        start_m = tl.load(Ind + i) * BLOCK_M
        offset = start_m % PAGE_SIZE
        table_idx = start_m // PAGE_SIZE
        cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
        k = desc_k.load([cache_idx * PAGE_SIZE + offset, off_kh, 0]).reshape(BLOCK_M, D1)
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2>0:
            k2 = desc_k2.load([cache_idx * PAGE_SIZE + offset, off_kh, D1]).reshape(BLOCK_M, D2)
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        v = desc_v.load([cache_idx * PAGE_SIZE + offset, off_kh, 0]).reshape(BLOCK_M, VD)
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

    desc_o.store([x_bos + off_n, off_kh, 0, 0], acc.reshape(1, 1, BLOCK_H, VD).to(desc_o.dtype))
    tl.store(Lse + tl.arange(0, BLOCK_H), lse, mask=tl.arange(0, BLOCK_H) < G)


@use_tma
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
    grid = lambda meta: (x_maxlen, B * KH)
    _prefill_kernel[grid](
        desc_q, 
        desc_q2, 
        desc_k, 
        desc_k2, 
        desc_v, 
        desc_o, 
        lse, 
        topk,
        x_cu_seqlens,
        context_lens,
        block_tables,
        *lse.stride(),
        *topk.stride(),
        block_tables.stride(0),
        sm_scale,
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
#                 for num_splits in [2, 4]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [1, 2, 4, 8]
#                  ], key=['D1', "D2", "BLOCK_H", "VD"])
@triton.jit
def _decode_stage1_kernel(
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
    o_stride_k, o_stride_n, o_stride_h, o_stride_d,
    lse_stride_k, lse_stride_n, lse_stride_h,
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

    top_n = tl.minimum(top_n, tl.cdiv(x_len, BLOCK_M))
    split_size = tl.cdiv(top_n, NUM_SPLITS)
    start_idx = split_size * off_k
    if start_idx < top_n:
        end_idx = tl.minimum(top_n, start_idx + split_size)

        q = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D1)[None, :], tl.arange(0, BLOCK_H)[:, None] < G2)
        desc_k = tl.make_tensor_descriptor(K, (num_blocks * PAGE_SIZE, D1), (k_stride_n, k_stride_d), (BLOCK_M, D1))
        desc_v = tl.make_tensor_descriptor(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (BLOCK_M, VD))
        if D2 > 0:
            q2 = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, tl.arange(0, BLOCK_H)[:, None] < G2)
            desc_k2 = tl.make_tensor_descriptor(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (BLOCK_M, D2))
                
        top_n = tl.minimum(top_n, tl.cdiv(x_len, BLOCK_M))
    
        sm_scale *= 1.44269504
        m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, VD], dtype=tl.float32)

        for i in range(start_idx, end_idx):
            start_m = tl.load(Ind + i) * BLOCK_M
            offset = start_m % PAGE_SIZE
            table_idx = start_m // PAGE_SIZE
            cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
            k = desc_k.load([cache_idx * PAGE_SIZE + offset, 0])
            v = desc_v.load([cache_idx * PAGE_SIZE + offset, 0])
            attn_score = tl.dot(q, tl.permute(k, 1, 0))
            if D2>0:
                k2 = desc_k2.load([cache_idx * PAGE_SIZE + offset, 0])
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
        tl.store(LSE + tl.arange(0, BLOCK_H) * lse_stride_h, lse, mask=tl.arange(0, BLOCK_H) < G2)
        tl.store(O + tl.arange(0, BLOCK_H)[:, None] * o_stride_h + tl.arange(0, D1)[None, :], acc, tl.arange(0, BLOCK_H)[:, None] < G2)
        
        
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
    desc_k = tl.make_tensor_descriptor(K, (num_blocks * PAGE_SIZE, D1), (k_stride_n, k_stride_d), (BLOCK_M, D1))
    desc_v = tl.make_tensor_descriptor(V, (num_blocks * PAGE_SIZE, VD), (v_stride_n, v_stride_d), (BLOCK_M, VD))
    if D2 > 0:
        q2 = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, tl.arange(0, BLOCK_H)[:, None] < G2)
        desc_k2 = tl.make_tensor_descriptor(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_n, k_stride_d), (BLOCK_M, D2))

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
            k = desc_k.load([cache_idx * PAGE_SIZE + offset, 0])
            v = desc_v.load([cache_idx * PAGE_SIZE + offset, 0])
            attn_score = tl.dot(q, tl.permute(k, 1, 0))
            if D2>0:
                k2 = desc_k2.load([cache_idx * PAGE_SIZE + offset, 0])
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
            k = desc_k.load([cache_idx * PAGE_SIZE + offset, 0])
            v = desc_v.load([cache_idx * PAGE_SIZE + offset, 0])
            attn_score = tl.dot(q, tl.permute(k, 1, 0))
            if D2>0:
                k2 = desc_k2.load([cache_idx * PAGE_SIZE + offset, 0])
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
def _decode_stage2_kernel(
    MID_O,
    O,
    LSE, 
    CONTEXT_LENS,
    mo_stride_k, mo_stride_n, mo_stride_h, mo_stride_d,
    o_stride_n, o_stride_h, o_stride_d,
    lse_stride_k, lse_stride_n, lse_stride_h,
    top_n,
    VD: tl.constexpr=128, 
    BLOCK_M: tl.constexpr=64,
    NUM_SPLITS: tl.constexpr=8
):

    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_d = tl.arange(0, VD)

    x_len = tl.load(CONTEXT_LENS + off_b)

    top_n = tl.minimum(top_n, tl.cdiv(x_len, BLOCK_M))
    # num_splits = tl.load(NUM_SPLITS + off_b)
    split_size = tl.cdiv(top_n, NUM_SPLITS)
    
    o_ptrs = O + off_b * o_stride_n + off_h * o_stride_h + off_d
    mid_o_ptrs = MID_O + off_b * mo_stride_n + off_h * mo_stride_h + off_d
    lse_ptrs = LSE + off_b * lse_stride_n + off_h * lse_stride_h
    
    m_i = float("-inf")
    l_i = 0.
    acc = tl.zeros([VD], dtype=tl.float32)
    off_k = 0
    for kv_start in range(0, top_n, split_size):
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
        
    
@use_tma
def slc_attn_decode(q, k, v, topk, block_tables, context_lens, block_size=64, top_n=16, num_splits=4, out=None, sm_scale=None, bench=False):
    B, QH, D = q.shape
    num_blocks, PAGE_SIZE, KH, _ = k.shape
    VD = v.size(-1)
    D1, D2 = split_d(D)
    G = QH // KH
    if sm_scale is None:
        sm_scale = D**-0.5
        
    k = k.view(-1, KH, D)
    v = v.view(-1, KH, VD)

    mid_o = torch.empty(num_splits, B, QH, VD, device=q.device, dtype=torch.float32)
    lse = torch.empty(num_splits, B, QH, dtype=torch.float32, device=q.device,)
    if out is None:
        out = torch.empty(B, QH, VD, device=q.device, dtype=q.dtype)

    BLOCK_H = max(triton.next_power_of_2(G), 16)
    G2 = min(BLOCK_H, G)
    BLOCK_M = block_size

    if D <= 128:
        kwargs = {"num_warps": 4, "num_stages": 2}
    else:
        kwargs = {"num_warps": 1, "num_stages": 1}
    
    NUM_SPLITS = num_splits
    grid = lambda meta: (B, QH // G2, NUM_SPLITS)
    _decode_stage1_kernel[grid](
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
        NUM_SPLITS=NUM_SPLITS,
        **kwargs
    )
    if bench:
        print(triton.testing.do_bench(lambda:     _decode_stage1_kernel[grid](
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
            NUM_SPLITS=NUM_SPLITS,
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
        top_n,
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
        top_n,
        VD,
        BLOCK_M=BLOCK_M,
        NUM_SPLITS=NUM_SPLITS,
        **kwargs
        )))
    return out, lse[0]


@use_tma
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
    if bench:
        print(triton.testing.do_bench(lambda:    
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
    )))

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
    if bench:
        print(triton.testing.do_bench(lambda: 
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
        )))
    return out[0], out[1]

