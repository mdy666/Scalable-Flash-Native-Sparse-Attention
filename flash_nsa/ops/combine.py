# Copyright (c) 2025 Duyue Ma

import math

import torch
import triton
import triton.language as tl


# @triton.autotune([triton.Config({'BLOCK_N': bsn}, num_warps=nw, num_stages=ns)
#                  for bsn in [16, 32, 64, 128]
#                  for nw in [4, 8, 16]
#                  for ns in [1, 2, 3 ,4]
#                  ], key=["D"])
@triton.jit
def _fused_sigmoid_combine_fwd_kernel(
    A,
    B,
    C,
    W,
    Y,
    san, sah, sad,
    sbn, sbh, sbd,
    scn, sch, scd,
    swn, swh, swk,
    syn, syh, syd,
    N:tl.constexpr,
    D:tl.constexpr,
    BLOCK_N:tl.constexpr=32,
):
    off_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    off_h = tl.program_id(1)
    off_d = tl.arange(0, D)
    mask = off_n < N

    # a = tl.load(A + off_n[:, None] * san + off_h * sah + off_d[None, :] * sad, mask=mask[:, None]).to(tl.float32)
    # b = tl.load(B + off_n[:, None] * sbn + off_h * sbh + off_d[None, :] * sbd, mask=mask[:, None]).to(tl.float32)
    # c = tl.load(C + off_n[:, None] * scn + off_h * sch + off_d[None, :] * scd, mask=mask[:, None]).to(tl.float32)
    a = tl.load(A + off_n[:, None] * san + off_h * sah + off_d[None, :] * sad, mask=mask[:, None])
    b = tl.load(B + off_n[:, None] * sbn + off_h * sbh + off_d[None, :] * sbd, mask=mask[:, None])
    c = tl.load(C + off_n[:, None] * scn + off_h * sch + off_d[None, :] * scd, mask=mask[:, None])
    w1 = tl.load(W + off_n * swn + off_h * swh + 0 * swk, mask=mask).to(tl.float32)
    w2 = tl.load(W + off_n * swn + off_h * swh + 1 * swk, mask=mask).to(tl.float32)
    w3 = tl.load(W + off_n * swn + off_h * swh + 2 * swk, mask=mask).to(tl.float32)
    y = a * tl.sigmoid(w1[:, None]) + b * tl.sigmoid(w2[:, None]) + c * tl.sigmoid(w3[:, None])
    tl.store(Y + off_n[:, None] * syn + off_h * syh + off_d[None, :] * syd, y, mask=mask[:, None])


# @triton.autotune([triton.Config({'BLOCK_N': bsn}, num_warps=nw, num_stages=ns)
#                  for bsn in [16, 32, 64, 128]
#                  for nw in [4, 8, 16]
#                  for ns in [1, 2, 3 ,4]
#                  ], key=["D"])
@triton.jit
def _fused_sigmoid_combine_bwd_kernel(
    A,
    B,
    C,
    W,
    DY,
    DA,
    DB,
    DC,
    DW,
    san, sah, sad,
    sbn, sbh, sbd,
    scn, sch, scd,
    swn, swh, swk,
    sdyn, sdyh, sdyd,
    sdan, sdah, sdad,
    sdbn, sdbh, sdbd,
    sdcn, sdch, sdcd,
    sdwn, sdwh, sdwk,
    N:tl.constexpr,
    D:tl.constexpr,
    BLOCK_N:tl.constexpr=32,
):
    off_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    off_h = tl.program_id(1)
    off_d = tl.arange(0, D)
    mask = off_n < N

    dy = tl.load(DY + off_n[:, None] * sdyn + off_h * sdyh + off_d[None, :] * sdyd, mask=mask[:, None])
    w1 = tl.load(W + off_n * swn + off_h * swh + 0 * swk, mask=mask).to(tl.float32)
    w2 = tl.load(W + off_n * swn + off_h * swh + 1 * swk, mask=mask).to(tl.float32)
    w3 = tl.load(W + off_n * swn + off_h * swh + 2 * swk, mask=mask).to(tl.float32)
    a = tl.load(A + off_n[:, None] * san + off_h * sah + off_d[None, :] * sad, mask=mask[:, None])
    b = tl.load(B + off_n[:, None] * sbn + off_h * sbh + off_d[None, :] * sbd, mask=mask[:, None])
    c = tl.load(C + off_n[:, None] * scn + off_h * sch + off_d[None, :] * scd, mask=mask[:, None])

    sig_w1 = tl.sigmoid(w1)
    sig_w2 = tl.sigmoid(w2)
    sig_w3 = tl.sigmoid(w3)

    dsig_w1 = tl.sum(dy * a, 1)
    dsig_w2 = tl.sum(dy * b, 1)
    dsig_w3 = tl.sum(dy * c, 1)

    dw1 = sig_w1 * (1 - sig_w1) * dsig_w1
    dw2 = sig_w2 * (1 - sig_w2) * dsig_w2
    dw3 = sig_w3 * (1 - sig_w3) * dsig_w3
    tl.store(DW + off_n * sdwn + off_h * sdwh + 0 * sdwk, dw1, mask=mask)
    tl.store(DW + off_n * sdwn + off_h * sdwh + 1 * sdwk, dw2, mask=mask)
    tl.store(DW + off_n * sdwn + off_h * sdwh + 2 * sdwk, dw3, mask=mask)   

    da = dy * sig_w1[:, None]
    db = dy * sig_w2[:, None]
    dc = dy * sig_w3[:, None]
    tl.store(DA + off_n[:, None] * sdan + off_h * sdah + off_d[None, :] * sdad, da, mask=mask[:, None])
    tl.store(DB + off_n[:, None] * sdbn + off_h * sdbh + off_d[None, :] * sdbd, db, mask=mask[:, None])
    tl.store(DC + off_n[:, None] * sdcn + off_h * sdch + off_d[None, :] * sdcd, dc, mask=mask[:, None])


def combine_fwd(a, b, c, w, out=None):
    T, H, D = a.shape
    assert a.shape == b.shape and a.shape == c.shape
    # assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous()
    assert list(w.shape) == [T, H, 3], f"w.shape: {w.shape}, expected: [{T}, {H}, 3]"
    assert math.log2(D).is_integer()

    if out is None:
        out = torch.empty(T, H, D, device=a.device, dtype=a.dtype)

    kwargs = {"BLOCK_N":16, "num_warps":8, "num_stages":3}
    grid = lambda meta: (triton.cdiv(T, meta['BLOCK_N']), H)
    _fused_sigmoid_combine_fwd_kernel[(grid)](
        a, 
        b, 
        c,
        w,
        out,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        *w.stride(),
        *out.stride(),
        T,
        D,
        **kwargs
    )

    return out

def combine_bwd(a, b, c, w, dout, dw=None):
    T, H, D = a.shape
    # dabc = torch.empty(3, T, H, D, device=a.device, dtype=a.dtype)
    # da, db, dc = dabc[0], dabc[1], dabc[2]
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    dc = torch.empty_like(c)
    dw = torch.empty_like(w) if dw is None else dw
    kwargs = {'BLOCK_N':16, "num_warps": 8, "num_stages":2}
    grid = lambda meta: (triton.cdiv(T, meta['BLOCK_N']), H)
    _fused_sigmoid_combine_bwd_kernel[(grid)](
        a, 
        b, 
        c,
        w,
        dout,
        da,
        db,
        dc,
        dw,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        *w.stride(),
        *dout.stride(),
        *da.stride(),
        *db.stride(),
        *dc.stride(),
        *dw.stride(),
        T,
        D,
        **kwargs
    )
    return da, db, dc, dw

class _FusedSigmoidCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c, w):
        out = combine_fwd(a, b, c, w)
        ctx.save_for_backward(a, b, c, w)
        return out

    @staticmethod
    def backward(ctx, dout):
        a, b, c, w = ctx.saved_tensors
        da, db, dc, dw = combine_bwd(a, b, c, w, dout)
        return da, db, dc, dw


def fused_sigmoid_combine(
    a: torch.Tensor, 
    b: torch.Tensor, 
    c: torch.Tensor, 
    w: torch.Tensor,
) -> torch.Tensor:
    '''
    combine cmp_o, slc_o and swa_o

    Args: 
        a (torch.tensor): [t, h, d]
        b (torch.tensor): [t, h, d]
        c (torch.tensor): [t, h, d]
        weight (torch.tensor): [t, h, 3]
    Returns:
        out (torch.tensor): [t, h, d]
    '''
    return _FusedSigmoidCombine.apply(a, b, c, w)

# def torch_sigmoid_combine(a, b, c, w):
#     return a * torch.nn.functional.sigmoid(w[..., 0].unsqueeze(-1)) + \
#         b * torch.nn.functional.sigmoid(w[..., 1].unsqueeze(-1)) + \
#         c * torch.nn.functional.sigmoid(w[..., 2].unsqueeze(-1))

