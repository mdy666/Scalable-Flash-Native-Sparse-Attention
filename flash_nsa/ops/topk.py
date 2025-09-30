# Copyright (c) 2025 Duyue Ma

import torch
import triton
import triton.language as tl

from ..utils import NSAHelper, use_tma

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

    desc_p = tl.make_tensor_descriptor(P, (cp_len, y_len), (p_stride_n, p_stride_m), (BLOCK_N, BLOCK_M))
    desc_q = tl.make_tensor_descriptor(Q, (G * KH, cp_len, D1), (q_stride_h, q_stride_n, q_stride_d), (1, BLOCK_N, D1))
    desc_k = tl.make_tensor_descriptor(K, (y_len, D1), (k_stride_m, k_stride_d), (BLOCK_M, D1))
    k = desc_k.load((start_m, 0))
    if D2 > 0:
        desc_q2 = tl.make_tensor_descriptor(Q + D1, (G * KH, cp_len, D2), (q_stride_h, q_stride_n, q_stride_d), (1, BLOCK_N, D2))
        desc_k2 = tl.make_tensor_descriptor(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (BLOCK_M, D2))
        k2 = desc_k2.load((start_m, 0))

    sm_scale *= 1.44269504
    p = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for off_qh in range(off_kh * G, off_kh * G + G):
        lse = tl.load(Lse + off_qh * lse_stride_h + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)
        q = desc_q.load([off_qh, cp_start_n, 0]).reshape(BLOCK_N, D1)
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            q2 = desc_q2.load([off_qh, cp_start_n, 0]).reshape(BLOCK_N, D2)
            attn_score = tl.dot(q2, tl.permute(k2, 1, 0), attn_score)
        p += tl.exp2(tl.fma(attn_score, sm_scale,  -lse[:, None]))
        # p += tl.exp2(attn_score * sm_scale -lse[:, None])

    if start_n < ((start_m + BLOCK_M - 1) * stride + kernel_size - 1):
        k_idx = (start_m + tl.arange(0, BLOCK_M)) * stride + kernel_size - 1
        causal_mask = q_idx[:, None] >= k_idx[None, :]
        p = tl.where(causal_mask, p, 0.)
    desc_p.store([cp_start_n, start_m], p.to(desc_p.dtype))

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

    desc_p = tl.make_tensor_descriptor(P, (cp_len, tl.cdiv(cp_len + cp_offset, 64)), (p_stride_n , p_stride_k), (BLOCK_N, BLOCK_K))
    desc_q = tl.make_tensor_descriptor(Q, (G, cp_len, D1), (q_stride_h, q_stride_n, q_stride_d), (1, BLOCK_N, D1))
    desc_k = tl.make_tensor_descriptor(K, (y_len, D1), (k_stride_m, k_stride_d), (BLOCK_M, D1))
    left_k = desc_k.load([start_m-1, 0]).reshape(BLOCK_K, 4, D1)
    k = desc_k.load([start_m, 0])
    left_k = tl.where(tl.arange(0,4)[None, :, None] < 1, left_k, 0)
    left_k = tl.sum(left_k.to(tl.float32), 1).to(k.dtype)

    if D2 > 0:
        desc_q2 = tl.make_tensor_descriptor(Q + D1, (G, cp_len, D2), (q_stride_h, q_stride_n, q_stride_d), (1, BLOCK_N, D2))
        desc_k2 = tl.make_tensor_descriptor(K + D1, (y_len, D2), (k_stride_m, k_stride_d), (BLOCK_M, D2))
        k2 = desc_k2.load((start_m, 0))
        left_k2 = desc_k2.load([start_m-1, 0]).reshape(BLOCK_K, 4, D2)
        left_k2 = tl.where(tl.arange(0,4)[None, :, None] < 1, left_k2, 0)
        left_k2 = tl.sum(left_k2, 1)

    sm_scale *= 1.44269504
    weight = (1 + (tl.arange(0, 4)<3))[None, None, :]
    main_p = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    left_p = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for off_qh in range(0, G):
        lse = tl.load(Lse + off_qh * lse_stride_h + cp_idx * lse_stride_n, mask=cp_idx < cp_len, other=0.)
        q = desc_q.load([off_qh, cp_start_n, 0]).reshape(BLOCK_N, D1)
        attn_score = tl.dot(q, tl.permute(k, 1, 0))
        left_score = tl.dot(q, tl.permute(left_k, 1, 0))
        if D2 > 0:
            q2 = desc_q2.load([off_qh, cp_start_n, 0]).reshape(BLOCK_N, D2)
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
        desc_p.store([cp_start_n, start_m//4], (p * p * p).to(desc_p.dtype))
    else:
        desc_p.store([cp_start_n, start_m//4], p.to(desc_p.dtype))

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

    desc_p = tl.make_tensor_descriptor(P, (cp_len, tl.cdiv(cp_len + cp_offset, 64)), (p_stride_n, p_stride_k), (BLOCK_N, BLOCK_K))
    acc_p = desc_p.load([cp_start_n, 0]).to(tl.float32)

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


# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                  for ns in [1, 2, 3, 4]
#                  for nw in [1, 2, 4, 8]
#                  ], key=["BLOCK_K"])
@triton.jit
def _topk_one_row_kernel(
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
    max_slc_blocks, 
    top_n: tl.constexpr, 
    num_inital: tl.constexpr, 
    num_local: tl.constexpr,
    BLOCK_K: tl.constexpr, 
):

    cp_idx = tl.program_id(0)
    cp_off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    if not CP:
        off_b = cp_off_b
        x_bos, x_eos = tl.load(X_CU_SEQLENS + off_b), tl.load(X_CU_SEQLENS + off_b + 1)
        x_len = x_eos - x_bos
        if cp_idx >= x_len:
            return
        cp_bos = x_bos
        cp_len = x_len
        cp_offset = 0
    else:
        off_b = tl.load(CP_BATCH_IDX + cp_off_b)
        cp_offset = tl.load(CP_OFFSET + cp_off_b)
        cp_bos, cp_eos = tl.load(CP_CU_SEQLENS + cp_off_b), tl.load(CP_CU_SEQLENS + cp_off_b + 1)
        cp_len = cp_eos - cp_bos
        if cp_idx >= cp_len:
            return

    P += off_h * p_stride_h.to(tl.int64) + cp_bos * p_stride_n.to(tl.int64) 
    Ind += off_h * ind_stride_h + cp_bos * ind_stride_n

    q_idx = cp_idx + cp_offset

    desc_p = tl.make_tensor_descriptor(P, (cp_len, max_slc_blocks), (p_stride_n, p_stride_k), (1, BLOCK_K))
    acc_p = desc_p.load([cp_idx, 0]).reshape(BLOCK_K)
    num_k = q_idx // block_size

    top_n = tl.minimum(top_n, q_idx // block_size + 1)
    for i in range(0, num_inital):
        tl.store(Ind + cp_idx * ind_stride_n + i, i, mask=i <= num_k)
        acc_p = tl.where(tl.arange(0, BLOCK_K) == i,
                        -1., acc_p)

    for i in range(0, num_local):
        tl.store(Ind + cp_idx * ind_stride_n + i + num_inital, num_k - i, mask=(i + num_inital <= num_k))
        acc_p = tl.where(tl.arange(0, BLOCK_K) == (num_k - i),
                        -1., acc_p)

    for i in range(num_inital+num_local, top_n):
        max_idx = tl.argmax(acc_p, axis=0)
        tl.store(Ind + cp_idx * ind_stride_n + i, max_idx, mask=i <= num_k)
        acc_p = tl.where(tl.arange(0, BLOCK_K) == max_idx,
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
        kwargs = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages":3}
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

@use_tma    
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
        kwargs = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}
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