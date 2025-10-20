import torch
import triton
import triton.language as tl
from ..utils import use_tma

# @triton.autotune([triton.Config({'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['D', "BLOCK_H"])
@triton.jit
def _index_socre_kernel1(
    Q,
    K,
    W,
    S,
    CU_SEQLENS,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_n, k_stride_h, k_stride_d,
    w_stride_n, w_stride_h,
    s_stride_n, s_stride_m,
    sm_scale,
    QH: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr=64,
):
    off_b = tl.program_id(0)
    q_idx = tl.program_id(2)
    start_m = tl.program_id(1) * BLOCK_M
    # causal
    if q_idx < start_m:
        return

    bos, eos = tl.load(CU_SEQLENS + off_b), tl.load(CU_SEQLENS + off_b + 1)
    seqlen = eos - bos

    if q_idx >= seqlen:
        return

    k_idx = start_m + tl.arange(0, BLOCK_M)
    desc_q = tl.make_tensor_descriptor(Q + (bos + q_idx) * q_stride_n, (QH, D), (q_stride_h, q_stride_d), (BLOCK_H, D))
    desc_k = tl.make_tensor_descriptor(K + bos * k_stride_n, (seqlen, D), (k_stride_n, k_stride_d), (BLOCK_M, D))
    # desc_s = tl.make_tensor_descriptor(S + (bos + q_idx) * s_stride_n, (seqlen, seqlen), (s_stride_n, s_stride_m), (BLOCK_H, D))
    q = desc_q.load([0, 0])
    k = desc_k.load([start_m, 0])
    w = tl.load(W + (bos + q_idx) * w_stride_n + tl.arange(0, BLOCK_H), tl.arange(0, BLOCK_H) < QH).to(tl.float32)
    w *= sm_scale
    score = tl.dot(q, tl.trans(k))
    score = tl.where(score >= 0, score * w[:, None], 0)
    score = tl.sum(score, 0)

    tl.store(S + (bos + q_idx) * s_stride_n + k_idx, score, q_idx>=k_idx)

# @triton.autotune([triton.Config({'BLOCK_M': bsm, "BLOCK_N": bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [64, 128]
#                  for bsm in [64, 128]
#                  for ns in [1, 2, 3, 4]
#                  for nw in [4, 8]
#                  ], key=['D', "BLOCK_H"])    
@triton.jit
def _index_socre_kernel2(
    Q,
    K,
    W,
    S,
    CU_SEQLENS,
    q_stride_n, q_stride_h, q_stride_d,
    k_stride_n, k_stride_h, k_stride_d,
    w_stride_n, w_stride_h,
    s_stride_n, s_stride_m,
    sm_scale,
    QH: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr=64,
    BLOCK_M: tl.constexpr=64,
):
    off_b = tl.program_id(0)
    start_n = tl.program_id(2) * BLOCK_N
    start_m = tl.program_id(1) * BLOCK_M
    # causal
    if start_n + BLOCK_N < start_m:
        return

    bos, eos = tl.load(CU_SEQLENS + off_b), tl.load(CU_SEQLENS + off_b + 1)
    seqlen = eos - bos

    if start_n >= seqlen:
        return

    q_idx = start_n + tl.arange(0, BLOCK_N)
    w_ptrs = W + (bos + q_idx) * w_stride_n
    desc_k = tl.make_tensor_descriptor(K + bos * k_stride_n, (seqlen, D), (k_stride_n, k_stride_d), (BLOCK_M, D))
    desc_s = tl.make_tensor_descriptor(S + bos * s_stride_n.to(tl.int64), (seqlen, seqlen), (s_stride_n, s_stride_m), (BLOCK_N, BLOCK_M))
    desc_q = tl.make_tensor_descriptor(Q + bos * q_stride_n, (seqlen, QH, D), (q_stride_n, q_stride_h, q_stride_d), (BLOCK_N, 1, D))
    k = tl.trans(desc_k.load([start_m, 0]))
    score = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for off_h in range(QH):
        q = desc_q.load([start_n, off_h, 0]).reshape(BLOCK_N, D)
        w = tl.load(w_ptrs, q_idx<seqlen).to(tl.float32)
        w *= sm_scale
        qk = tl.dot(q, k)
        qk = tl.where(qk >= 0, qk * w[:, None], 0)
        score += qk
        w_ptrs += 1
    if start_n < start_m + BLOCK_M - 1:
        k_idx = start_m + tl.arange(0, BLOCK_M)
        score = tl.where(q_idx[:, None]>=k_idx[None, :], score, float("-inf")) 
    desc_s.store([start_n, start_m], score.to(desc_s.dtype))

@use_tma
def index_socre(q, k, w, cu_seqlens, maxlen, topk=2048, sm_scale=None, score_dtype=torch.float32, mode=2):
    T, QH, D = q.shape
    B = len(cu_seqlens) - 1
    assert T == k.size(0) and D == k.size(2) and k.size(1) == 1
    assert T == w.size(0) and QH == w.size(1)
    assert triton.next_power_of_2(D) == D
    if sm_scale is None:
        sm_scale = D ** -0.5

    score = torch.full((T, max(triton.cdiv(maxlen, 8) * 8, topk)), float('-inf'), device=q.device, dtype=score_dtype)
    if mode == 1:
        BLOCK_H = max(triton.next_power_of_2(QH), 16)
        kwargs = {"BLOCK_M": 128, "num_warps": 4, "num_stages": 1}
        grid = lambda meta: (B, triton.cdiv(maxlen, meta["BLOCK_M"]), maxlen)
        _index_socre_kernel1[grid](
            q,
            k,
            w,
            score,
            cu_seqlens,
            *q.stride(),
            *k.stride(),
            *w.stride(),
            *score.stride(),
            sm_scale,
            QH,
            D,
            BLOCK_H,
            **kwargs
        )
    else:
        kwargs = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}
        grid = lambda meta: (B, triton.cdiv(maxlen, meta["BLOCK_M"]), triton.cdiv(maxlen, meta["BLOCK_N"]))
        _index_socre_kernel2[grid](
            q,
            k,
            w,
            score,
            cu_seqlens,
            *q.stride(),
            *k.stride(),
            *w.stride(),
            *score.stride(),
            sm_scale,
            QH,
            D,
            **kwargs
        )
    return score

@triton.autotune([triton.Config({'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
                 for bsm in [64]
                 for ns in [1, 2, 3, 4]
                 for nw in [2, 4, 8]
                 ], key=['D1', "D2", "BLOCK_H"])
@triton.jit
def _fwd_kernel(
    Q,
    KV,
    O,
    Lse,
    Ind,
    CU_SEQLENS,
    q_stride_n, q_stride_h, q_stride_d,
    kv_stride_n, kv_stride_h, kv_stride_d,
    o_stride_n, o_stride_h, o_stride_d,
    topk,
    sm_scale,
    QH: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr=32,
):
    q_idx = tl.program_id(0)
    off_b = tl.program_id(1)

    bos, eos = tl.load(CU_SEQLENS + off_b), tl.load(CU_SEQLENS + off_b + 1)
    seqlen = eos - bos

    if q_idx >= seqlen:
        return
    
    desc_q1 = tl.make_tensor_descriptor(Q + (bos + q_idx) * q_stride_n, (QH, D1), (q_stride_h, q_stride_d), (BLOCK_H, D1))
    desc_q2 = tl.make_tensor_descriptor(Q + (bos + q_idx) * q_stride_n + D1, (QH, D2), (q_stride_h, q_stride_d), (BLOCK_H, D2))
    desc_o = tl.make_tensor_descriptor(O + (bos + q_idx) * o_stride_n, (QH, D1), (o_stride_h, o_stride_d), (BLOCK_H, D1))
    KV += bos * kv_stride_n
    Ind += (bos + q_idx) * topk
    # desc_kv1 = tl.make_tensor_descriptor(KV, (seqlen, D1), (kv_stride_n, kv_stride_d), (BLOCK_M, D1))
    # desc_kv2 = tl.make_tensor_descriptor(KV + D1, (seqlen, D2), (kv_stride_n, kv_stride_d), (BLOCK_M, D2))
    
    nope_q = desc_q1.load([0, 0])
    rope_q = desc_q2.load([0, 0])
    
    acc = tl.zeros((BLOCK_H, D1), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_H,), dtype=tl.float32) - float('inf')
    
    sm_scale *= 1.44269504
    stop_k = tl.minimum(topk, q_idx + 1)
    for start in range(0, stop_k, BLOCK_M):
        off_k = start + tl.arange(0, BLOCK_M)
        k_idx = tl.load(Ind + off_k, off_k<stop_k)
        kv = tl.load(KV + k_idx[:, None] * kv_stride_n + tl.arange(0, D1)[None, :])
        rope_k = tl.load(KV + k_idx[:, None] * kv_stride_n + tl.arange(0, D2)[None, :] + D1)
        
        score = tl.zeros((BLOCK_H, BLOCK_M), dtype=tl.float32)
        k_idx = tl.where(off_k < stop_k, k_idx, seqlen)
        score = tl.where(q_idx >= k_idx[None, :], score, float('-inf'))
        score = tl.dot(nope_q, tl.trans(kv), score)
        
        # rope_k = desc_kv2.load([start, 0])
        score = tl.dot(rope_q, tl.trans(rope_k), score)
        score *= sm_scale
        # k_idx = tl.where(off_k < stop_k, k_idx, seqlen)
        # score = tl.where(q_idx >= k_idx[None, :], score * sm_scale, float('-inf'))
        new_m_i = tl.maximum(m_i, tl.max(score, 1))
        alpha = tl.exp2(m_i - new_m_i)
        exp_score = tl.exp2(score - new_m_i[:, None])
        l_i = l_i * alpha + tl.sum(exp_score, 1)
        acc *= alpha[:, None]
        acc += tl.dot(exp_score.to(kv.dtype), kv)
        m_i = new_m_i
    acc /= l_i[:, None]
    m_i += tl.log2(l_i)
    
    tl.store(Lse + (bos + q_idx) * QH + tl.arange(0, BLOCK_H), m_i, tl.arange(0, BLOCK_H) < QH)
    desc_o.store([0, 0], acc.to(desc_o.dtype))
        
    
def mqa_sparser_fwd(q, kv, index, VD, cu_seqlens, maxlen, sm_scale):
    T, QH, D = q.shape
    B = len(cu_seqlens) - 1
    D1 = VD
    D2 = D - D1
    topk = index.size(-1)
    assert T == kv.size(0) and D == kv.size(2) and kv.size(1) == 1
    assert index.size(0) == T
    assert triton.next_power_of_2(D1) == D1 and triton.next_power_of_2(D2) == D2
    assert sm_scale is not None
    BLOCK_H = max(triton.next_power_of_2(QH), 16)
    
    o = torch.empty(T, QH, VD, device=q.device, dtype=q.dtype)
    lse = torch.empty(T, QH, device=q.device, dtype=torch.float32)
    kwargs = {"BLOCK_M": 64, "num_warps": 4, "num_stages": 2}
    grid = lambda meta: (maxlen, B)
    _fwd_kernel[grid](
        q,
        kv,
        o,
        lse,
        index,
        cu_seqlens,
        *q.stride(),
        *kv.stride(),
        *o.stride(),
        topk,
        sm_scale,
        QH,
        D1,
        D2,
        BLOCK_H,
        # **kwargs
    )
    return o, lse


def torch_mqa_sparser_ref(q, kv, index, VD, sm_scale=1.):
    T, QH, D = q.shape
    D1 = VD
    D2 = D - D1
    index = index[..., :T]
    q = q.transpose(0, 1)
    kv = kv.transpose(0, 1)
    v = kv[..., :VD]
    s = q @ kv.permute(0, 2, 1)
    idx = torch.arange(T, device=q.device)
    causal_mask = idx[:, None] >= idx[None, :]
    slc_mask = torch.zeros_like(causal_mask)
    slc_mask.scatter_(1, index, 1)
    mask = causal_mask & slc_mask
    s = torch.where(mask[None], s * sm_scale, float('-inf'))
    p = s.softmax(-1, dtype=torch.float32)
    o = p.to(v.dtype) @ v
    return o.transpose(0, 1).contiguous(), p


# ruff: noqa
import torch
import tilelang
from tilelang import language as T


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert (topk %
            block_I == 0), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert (
            kv_group == 1
        ), "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(
                seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
                    bx,
                    by,
                    bz,
                ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            Lse_shared = T.alloc_shared([H_per_block], accum_dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):

                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i,
                                              d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i,
                                                  D + d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, O_shared)
            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse_shared)
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    # assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    kernel = sparse_mla_fwd(heads, dim, tail_dim, topk, kv_group, sm_scale, is_casual)
    out, lse = kernel(q, kv, indices)
    return out, lse
