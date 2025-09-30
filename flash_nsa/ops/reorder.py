# Copyright (c) 2025 Duyue Ma

import torch
import triton
import triton.language as tl

# @triton.autotune([triton.Config({"BLOCK_N": BLOCK_N}, num_stages=ns, num_warps=nw)
#                   for BLOCK_N in [128, 256, 512, 1024]
#                   for ns in [1, 2, 3, 4]
#                   for nw in [2, 4, 8]
#                   ],
#                   key=["BLOCK_M", "N"])
@triton.jit
def _kernel(
    X,
    Y,
    SHUFFLE: tl.constexpr,
    INPLACE: tl.constexpr,
    N:tl.constexpr,
    BLOCK_M:tl.constexpr,
    BLOCK_N:tl.constexpr,
):
    off_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    off_m = tl.arange(0, BLOCK_M)

    mask = off_n[None, :] < N

    if SHUFFLE:
        x1 = tl.load(X + off_m[:, None] * N + off_n[None, :], mask=mask)
        x2 = tl.load(X + (off_m[:, None] + BLOCK_M) * N + off_n[None, :], mask=mask)
        if INPLACE:
            tl.debug_barrier()
        tl.store(Y + 2 * off_m[:, None] * N + off_n[None, :], x1)
        tl.store(Y + (2 * (BLOCK_M - off_m[:, None]) - 1) * N + off_n[None, :], x2)
    else:
        x1 = tl.load(X + 2 * off_m[:, None] * N + off_n[None, :], mask=mask)
        x2 = tl.load(X + (2 * (BLOCK_M - off_m[:, None]) - 1) * N + off_n[None, :], mask=mask)
        if INPLACE:
            tl.debug_barrier()
        tl.store(Y + off_m[:, None] * N + off_n[None, :], x1)
        tl.store(Y + (off_m[:, None] + BLOCK_M) * N + off_n[None, :], x2)

def reorder(
    x: torch.Tensor, 
    world_size: int, 
    shuffle_mode: str = "shuffle", 
    inplace: bool = True
) -> torch.Tensor:

    '''
    Only for context parallel and cp_mode=2.

    When after kv all_gather, the order is not sequential, we need reorder the kv.
    [0, 7, 1, 6, 2, 5, 3, 4] -> [0, 1, 2, 3, 4, 5, 6, 7]

    When before dkdv reduce_scatter, the order is sequential, but chunk_kv it is not. we need shuffle the dkdv.
    [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 7, 1, 6, 2, 5, 3, 4]

    Args:
        x (torch.Tensor): [t, h, d]
        world_size (int): context paraller size, must be power of 2
        shuffle_mode (str): must be shuffle or unshuffle, shuffle for reduce_scatter and unshuffle for all_gather
        inplace (bool): Whether do inplace op in x.
    Returns:
        y (torch.Tensor): [t, h, d]
    '''

    assert len(x.shape) == 3
    assert x.is_contiguous()
    assert world_size == triton.next_power_of_2(world_size)
    assert shuffle_mode in ["shuffle", "unshuffle"]
    T, H, D = x.shape
    assert T % (world_size * 2) == 0

    shuffle = shuffle_mode == "shuffle"
    chunk_size = T // (world_size * 2)
    N = chunk_size * H * D
    BLOCK_M = world_size

    y = x if inplace else torch.empty_like(x) 

    kwargs = {"BLOCK_N": 8192//BLOCK_M, "num_warps":2, "num_stages":4}
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), )
    _kernel[grid](
        x,
        y,
        shuffle,
        inplace,
        N,
        BLOCK_M,
        **kwargs
    )
    return y

