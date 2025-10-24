# Copyright (c) 2025 Duyue Ma

import torch
import triton
import triton.language as tl

from .utils import split_d, use_tma, NSAHelper
from . import ampere_ops

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
    
    desc_p = tl.make_tensor_descriptor(P, (x_len, y_len), (p_stride_n, p_stride_m), (BLOCK_N, BLOCK_M))
    desc_q = tl.make_tensor_descriptor(Q, (G * KH, x_len, D1), (q_stride_h, q_stride_n, q_stride_d), (1, BLOCK_N, D1))
    desc_k = tl.make_tensor_descriptor(K, (num_blocks * PAGE_SIZE, D1), (k_stride_m, k_stride_d), (BLOCK_M, D1))
    k = desc_k.load((cache_idx * PAGE_SIZE + offset, 0))
    if D2 > 0:
        desc_q2 = tl.make_tensor_descriptor(Q + D1, (G * KH, x_len, D2), (q_stride_h, q_stride_n, q_stride_d), (1, BLOCK_N, D2))
        desc_k2 = tl.make_tensor_descriptor(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_m, k_stride_d), (BLOCK_M, D2))
        k2 = desc_k2.load((cache_idx * PAGE_SIZE + offset, 0))

    sm_scale *= 1.44269504
    p = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for off_qh in range(off_kh * G, off_kh * G + G):
        lse = tl.load(Lse + off_qh * lse_stride_h + off_n * lse_stride_n, mask=off_n < x_len, other=0.)
        # q = desc_q.load([off_qh, start_n, 0]).reshape(BLOCK_N, D1)
        q = desc_q.load([off_qh, start_n, 0])
        q = tl.reshape(q, (BLOCK_N, D1))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            # q2 = desc_q2.load([off_qh, start_n, 0]).reshape(BLOCK_N, D2)
            q2 = desc_q2.load([off_qh, start_n, 0])
            q2 = tl.reshape(q2, (BLOCK_N, D2))
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        p += tl.exp2(tl.fma(attn_score, sm_scale,  -lse[:, None]))
        # p += tl.exp2(attn_score * sm_scale -lse[:, None])

    # if (start_n + cached_tokens) < ((start_m + BLOCK_M - 1) * stride + kernel_size - 1):
    #     k_idx = (start_m + tl.arange(0, BLOCK_M)) * stride + kernel_size - 1
    #     causal_mask = q_idx[:, None] >= k_idx[None, :]
    #     p = tl.where(causal_mask, p, 0.)
    desc_p.store([start_n, start_m], p.to(desc_p.dtype))

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
        
@use_tma    
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
    if NSAHelper.is_use_ampere_ops():
        return ampere_ops.topk_prefill(
            q,
            k, 
            lse,
            x_cu_seqlens,
            x_maxlen,
            y_maxlen,
            block_tables,
            context_lens,
            fixed_y_maxlen,
            fixed_num_slc_blocks,
            kernel_size,
            stride,
            block_size,
            top_n,
            num_inital, 
            num_local,
            sm_scale,
            fp32
        )
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
        kwargs = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}
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
    desc_k = tl.make_tensor_descriptor(K, (num_blocks * PAGE_SIZE, D1), (k_stride_m, k_stride_d), (BLOCK_M, D1))
    k = desc_k.load((cache_idx * PAGE_SIZE + offset, 0))
    attn_score = tl.dot(q, tl.permute(k, 1, 0))
    if D2 > 0:
        q2 = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, tl.arange(0, BLOCK_H)[:, None] < G)
        desc_k2 = tl.make_tensor_descriptor(K + D1, (num_blocks * PAGE_SIZE, D2), (k_stride_m, k_stride_d), (BLOCK_M, D2))
        k2 = desc_k2.load((cache_idx * PAGE_SIZE + offset, 0))
        attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
    p = tl.sum(tl.exp2(tl.fma(attn_score, sm_scale, -lse[:, None])), 0)
    tl.store(P + start_m + tl.arange(0, BLOCK_M), p, mask=(start_m + tl.arange(0, BLOCK_M)) < y_len)
    
# @triton.autotune([triton.Config({'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [2, 4, 8]
#                  ], key=['D1', "D2"])
@triton.jit
def _attn_probs_decode_kernel2(
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
    pid = tl.program_id(0)
    off_b = tl.program_id(2)
    off_kh = tl.program_id(1)
    start_qh = off_kh * G

    x_len = tl.load(CONTEXT_LENS + off_b)
    if x_len <= kernel_size - 1:
        return
    y_len = tl.maximum((x_len - kernel_size) // stride + 1, 0)

    Q += off_b * q_stride_n + start_qh * q_stride_h
    K += off_kh * k_stride_h
    Lse += off_b * lse_stride_n + start_qh * lse_stride_h
    P += off_kh * p_stride_h + off_b * p_stride_n

    sm_scale *= 1.44269504
    lse = tl.load(Lse + tl.arange(0, BLOCK_H) * lse_stride_h, mask=tl.arange(0, BLOCK_H) < G, other=0.)
    q = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D1)[None, :], tl.arange(0, BLOCK_H)[:, None] < G)
    desc_k = tl.make_tensor_descriptor(K, (num_blocks * PAGE_SIZE, D1), (k_stride_m, k_stride_d), (BLOCK_M, D1))
    if D2 > 0:
        q2 = tl.load(Q + tl.arange(0, BLOCK_H)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, tl.arange(0, BLOCK_H)[:, None] < G)
        desc_k2 = tl.make_tensor_descriptor(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (BLOCK_M, D2))
    
    for start_m in range(pid * BLOCK_M, y_len, tl.num_programs(0) * BLOCK_M):
        offset = start_m % PAGE_SIZE
        table_idx = start_m // PAGE_SIZE
        cache_idx = tl.load(TABLES + off_b * table_stride + table_idx)
        k = desc_k.load((cache_idx * PAGE_SIZE + offset, 0))
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            k2 = desc_k2.load((cache_idx * PAGE_SIZE + offset, 0))
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
        
@use_tma    
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
    if NSAHelper.is_use_ampere_ops():
        return ampere_ops.topk_decode(
                q, 
                k, 
                lse,
                block_tables,
                context_lens,
                kernel_size,
                stride,
                block_size,
                top_n,
                num_inital, 
                num_local,
                fixed_num_slc_blocks,
                fixed_y_maxlen,
                sm_scale, 
                persistent,
                workers,
            )
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

    if not persistent:
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
    else:
        kwargs = {"BLOCK_M": 64, "num_warps": 2, "num_stages": 2}
        grid = lambda meta: (workers, KH, B)
        _attn_probs_decode_kernel2[grid](
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