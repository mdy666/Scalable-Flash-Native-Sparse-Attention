# Copyright (c) 2025 Duyue Ma

import torch
import triton
import triton.language as tl

# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                   for ns in [1, 2, 3, 4]
#                   for nw in [2, 4, 8]
#                   ],
#                   key=['BLOCK_QH','BLOCK_KH',"BLOCK_D"])
@triton.jit
def _fused_apply_rope_kernel(
    Q, 
    K, 
    QE,
    KE,
    COS, 
    SIN,
    q_stride_b, q_stride_n, q_stride_h, q_stride_d,
    k_stride_b, k_stride_n, k_stride_h, k_stride_d,
    qe_stride_b, qe_stride_n, qe_stride_h, qe_stride_d,
    ke_stride_b, ke_stride_n, ke_stride_h, ke_stride_d,
    cos_stride_b, cos_stride_n, cos_stride_d,
    QH,
    KH,
    D,
    HALF_D,
    BLOCK_QH:tl.constexpr, 
    BLOCK_KH:tl.constexpr, 
    BLOCK_D:tl.constexpr,
    BACKWARD: tl.constexpr=False,
):
    off_b = tl.program_id(1)
    off_n = tl.program_id(0)


    Q += off_b * q_stride_b + off_n * q_stride_n
    K += off_b * k_stride_b + off_n * k_stride_n
    QE += off_b * qe_stride_b + off_n * qe_stride_n
    KE += off_b * ke_stride_b + off_n * ke_stride_n
    COS += off_b * cos_stride_b + off_n * cos_stride_n
    SIN += off_b * cos_stride_b + off_n * cos_stride_n


    k1_ptrs = tl.make_block_ptr(K,
                                (KH, HALF_D),
                                (k_stride_h, k_stride_d),
                                (0, 0),
                                (BLOCK_KH, BLOCK_D),
                                (1, 0))

    k2_ptrs = tl.make_block_ptr(K + HALF_D,
                                (KH, HALF_D),
                                (k_stride_h, k_stride_d),
                                (0, 0),
                                (BLOCK_KH, BLOCK_D),
                                (1, 0))

    q1_ptrs = tl.make_block_ptr(Q,
                                (QH, HALF_D),
                                (q_stride_h, q_stride_d),
                                (0, 0),
                                (BLOCK_QH, BLOCK_D),
                                (1, 0))

    q2_ptrs = tl.make_block_ptr(Q + HALF_D,
                                (QH, HALF_D),
                                (q_stride_h, q_stride_d),
                                (0, 0),
                                (BLOCK_QH, BLOCK_D),
                                (1, 0))

    k1 = tl.load(k1_ptrs, boundary_check=(0,1)).to(tl.float32)
    k2 = tl.load(k2_ptrs, boundary_check=(0,1)).to(tl.float32)
    q1 = tl.load(q1_ptrs, boundary_check=(0,1)).to(tl.float32)
    q2 = tl.load(q2_ptrs, boundary_check=(0,1)).to(tl.float32)
    cols = tl.arange(0, BLOCK_D)
    cos1 = tl.load(COS + cols, mask=cols<HALF_D, other=0.).to(tl.float32)[None, :]
    cos2 = tl.load(COS + cols + HALF_D, mask=(cols + HALF_D)<D, other=0.).to(tl.float32)[None, :]
    if not BACKWARD:
        sin1 = tl.load(SIN + cols, mask=cols<HALF_D, other=0.).to(tl.float32)[None, :]
        sin2 = tl.load(SIN + cols + HALF_D, mask=(cols + HALF_D)<D, other=0.).to(tl.float32)[None, :]
    else:
        sin2 = -tl.load(SIN + cols, mask=cols<HALF_D, other=0.).to(tl.float32)[None, :]
        sin1 = -tl.load(SIN + cols + HALF_D, mask=(cols + HALF_D)<D, other=0.).to(tl.float32)[None, :]


    ke1 = k1 * cos1 - k2 * sin1
    ke2 = k2 * cos2 + k1 * sin2
    qe1 = q1 * cos1 - q2 * sin1
    qe2 = q2 * cos2 + q1 * sin2

    ke1_ptrs = tl.make_block_ptr(KE,
                                (KH, HALF_D),
                                (ke_stride_h, ke_stride_d),
                                (0, 0),
                                (BLOCK_KH, BLOCK_D),
                                (1, 0))

    ke2_ptrs = tl.make_block_ptr(KE + HALF_D,
                                (KH, HALF_D),
                                (ke_stride_h, ke_stride_d),
                                (0, 0),
                                (BLOCK_KH, BLOCK_D),
                                (1, 0))

    qe1_ptrs = tl.make_block_ptr(QE,
                                (QH, HALF_D),
                                (qe_stride_h, qe_stride_d),
                                (0, 0),
                                (BLOCK_QH, BLOCK_D),
                                (1, 0))

    qe2_ptrs = tl.make_block_ptr(QE + HALF_D,
                                (QH, HALF_D),
                                (qe_stride_h, qe_stride_d),
                                (0, 0),
                                (BLOCK_QH, BLOCK_D),
                                (1, 0))

    tl.store(ke1_ptrs, ke1.to(Q.dtype.element_ty), boundary_check=(0,1))
    tl.store(ke2_ptrs, ke2.to(Q.dtype.element_ty), boundary_check=(0,1))
    tl.store(qe1_ptrs, qe1.to(Q.dtype.element_ty), boundary_check=(0,1))
    tl.store(qe2_ptrs, qe2.to(Q.dtype.element_ty), boundary_check=(0,1))

class _FusedApplyRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, inplace):
        assert q.size(1) == cos.size(1)

        B, N, QH, D = q.size()
        KH = k.size(-2)
        HALF_D = D // 2
        BLOCK_D = triton.next_power_of_2(HALF_D)
        BLOCK_QH = triton.next_power_of_2(QH)
        BLOCK_KH = triton.next_power_of_2(KH)

        if not inplace:
            qe = torch.empty(B, N, QH, D, device=q.device, dtype=k.dtype)
            ke = torch.empty(B, N, KH, D, device=q.device, dtype=k.dtype)
        else:
            qe, ke = q, k

        grid = lambda meta: (N, B)
        _fused_apply_rope_kernel[(grid)](
            q,
            k,
            qe,
            ke,
            cos, 
            sin,
            *q.stride(),
            *k.stride(),
            *qe.stride(),
            *ke.stride(),
            *cos.stride(),
            QH,
            KH,
            D,
            HALF_D,
            BLOCK_QH,
            BLOCK_KH,
            BLOCK_D,
        )
        ctx.save_for_backward(q, k, cos, sin)
        ctx.infos = (B, N, QH, KH, D, HALF_D, BLOCK_QH, BLOCK_KH, BLOCK_D)
        ctx.inplace = inplace
        return qe, ke

    @staticmethod
    def backward(ctx, dqe, dke):
        B, N, QH, KH, D, HALF_D, BLOCK_QH, BLOCK_KH, BLOCK_D = ctx.infos
        q, k, cos,sin = ctx.saved_tensors

        # dq, dk = torch.empty_like(q), torch.empty_like(k)
        dq, dk = q, k

        grid = lambda meta: (N, B)
        _fused_apply_rope_kernel[(grid)](
            dqe,
            dke,
            dq,
            dk,
            cos, 
            sin,
            *dqe.stride(),
            *dke.stride(),
            *dq.stride(),
            *dk.stride(),
            *cos.stride(),
            QH,
            KH,
            D,
            HALF_D,
            BLOCK_QH,
            BLOCK_KH,
            BLOCK_D,
            BACKWARD=True
        )
        return dq, dk, None, None, None

def fused_apply_rope(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    inplace: bool = False,
)-> tuple[torch.Tensor, torch.Tensor]:
    '''
    rope kernel

    Args:
        q (torch.Tensor): [b, s, qh, d]
        k (torch.Tensor): [b, s, kh, d]
        cos (torch.Tensor): [b, s, d]
        sin (torch.Tensor): [b, s, d]
        inplace (bool): Whether do inplace op on q and k

    Returns:
        qe (torch.Tensor): [b, s, qh, d]
        ke (torch.Tensor): [b, s, kh, d]
    '''
    if len(cos.shape) == 4:
        cos = cos.squeeze(-2)
    if len(sin.shape) == 4:
        sin = sin.squeeze(-2)
    if cos.size(0) != q.size(0):
        b = q.size(0)
        _, s, d = cos.shape
        cos = cos.expand(b, s, d)
        sin = sin.expand(b, s, d)
    return _FusedApplyRope.apply(q, k, cos, sin, inplace)