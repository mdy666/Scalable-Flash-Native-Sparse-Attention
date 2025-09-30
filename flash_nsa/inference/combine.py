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
def _prefill_kernel(
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
    
# @triton.autotune([triton.Config({}, num_warps=nw, num_stages=ns)
#                  for nw in [1, 2, 4, 8]
#                  for ns in [1, 2, 3 ,4]
#                  ], key=["D"])
@triton.jit
def _decode_kernel(
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
    D:tl.constexpr,
):
    off_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_d = tl.arange(0, D)

    a = tl.load(A + off_n * san + off_h * sah + off_d * sad)
    b = tl.load(B + off_n * sbn + off_h * sbh + off_d * sbd)
    c = tl.load(C + off_n * scn + off_h * sch + off_d * scd)
    w1 = tl.load(W + off_n * swn + off_h * swh + 0 * swk).to(tl.float32)
    w2 = tl.load(W + off_n * swn + off_h * swh + 1 * swk).to(tl.float32)
    w3 = tl.load(W + off_n * swn + off_h * swh + 2 * swk).to(tl.float32)
    y = a * tl.sigmoid(w1) + b * tl.sigmoid(w2) + c * tl.sigmoid(w3)
    tl.store(Y + off_n * syn + off_h * syh + off_d * syd, y)

def combine_prefill(a, b, c, w, out=None):
    T, H, D = a.shape

    if out is None:
        out = torch.empty(T, H, D, device=a.device, dtype=a.dtype)

    kwargs = {"BLOCK_N":16, "num_warps":8, "num_stages":3}
    grid = lambda meta: (triton.cdiv(T, meta['BLOCK_N']), H)
    _prefill_kernel[(grid)](
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

def combine_decode(a, b, c, w, out=None):
    B, H, D = a.shape

    if out is None:
        out = torch.empty(B, H, D, device=a.device, dtype=a.dtype)

    kwargs = {"num_warps":2, "num_stages":2}
    grid = lambda meta: (B, H)
    _decode_kernel[(grid)](
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
        D,
        **kwargs
    )
    return out