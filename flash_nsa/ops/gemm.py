# Copyright (c) 2025 Duyue Ma

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from ..utils import use_tma


def hook(config):
    BLOCK_M, BLOCK_N, BLOCK_K = config['BLOCK_M'], config['BLOCK_N'], config['BLOCK_K']
    if not config["TRANS_A"]:
        config['desc_a'].block_shape = [BLOCK_M, BLOCK_K]
    else:
        config['desc_a'].block_shape = [BLOCK_K, BLOCK_M]
    if config["TRANS_B"]:
        config['desc_b'].block_shape = [BLOCK_N, BLOCK_K]
    else:
        config['desc_b'].block_shape = [BLOCK_K, BLOCK_N]
    config['desc_c'].block_shape = [BLOCK_M, BLOCK_N]

# @triton.autotune(configs=[triton.Config({"BLOCK_M": BM, "BLOCK_N":BN, "BLOCK_K": BK}, num_stages=ns, num_warps=nw, pre_hook=hook)
#                           for BM in [128, 256]
#                           for BN in [128, 256]
#                           for BK in [64, 128]
#                           for ns in [2, 3, 4]
#                           for nw in [4, 8]], key=["M", "N", "K", "ACC"])
@triton.jit
def _gemm_kernel(
    desc_a, 
    desc_b,
    desc_c,
    ACC: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr=8,
):

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in tl.range(0, K, BLOCK_K):

        if not TRANS_A:
            a = desc_a.load([start_m, start_k])
        else:
            a = desc_a.load([start_k, start_m])
            a = tl.trans(a)
        if TRANS_B:
            b = desc_b.load([start_n, start_k])
            b = tl.trans(b)
        else:
            b = desc_b.load([start_k, start_n])

        acc = tl.dot(a, b, acc)
    if not ACC:
        desc_c.store([start_m, start_n], acc.to(desc_c.dtype))
    else:
        desc_c.atomic_add([start_m, start_n], acc)

# @triton.autotune(configs=[triton.Config({"BLOCK_M": BM, "BLOCK_N":BN, "BLOCK_K": BK}, num_stages=ns, num_warps=nw, pre_hook=hook)
#                           for BM in [128, 256]
#                           for BN in [128, 256]
#                           for BK in [64, 128]
#                           for ns in [2, 3, 4]
#                           for nw in [4, 8]], key=["M", "N", "K", "ACC"])
@triton.jit
def _persistent_gemm_kernel(desc_a, 
                 desc_b,
                 desc_c,
                 ACC: tl.constexpr,
                 M: tl.constexpr,
                 N: tl.constexpr,
                 K: tl.constexpr,
                 TRANS_A: tl.constexpr,
                 TRANS_B: tl.constexpr,
                 BLOCK_M: tl.constexpr,
                 BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr,
                 GROUP_SIZE_M: tl.constexpr=8,
                 ):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_tiles = num_pid_m * num_pid_n
    start_tile_id = tl.program_id(0)

    for pid in range(start_tile_id, num_tiles, tl.num_programs(0)):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        start_m = pid_m * BLOCK_M
        start_n = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start_k in tl.range(0, K, BLOCK_K):
            if not TRANS_A:
                a = desc_a.load([start_m, start_k])
            else:
                a = desc_a.load([start_k, start_m])
                a = tl.trans(a)
            if TRANS_B:
                b = desc_b.load([start_n, start_k])
                b = tl.trans(b)
            else:
                b = desc_b.load([start_k, start_n])
            acc = tl.dot(a, b, acc)
        if not ACC:
            desc_c.store([start_m, start_n], acc.to(desc_c.dtype))
        else:
            desc_c.atomic_add([start_m, start_n], acc)
@use_tma    
def gemm(
    a: torch.Tensor, 
    b: torch.Tensor, 
    out: torch.Tensor=None, 
    acc: bool = False, 
    transpose_a: bool = False,
    transpose_b: bool = True,
    persistent: bool = False
) -> torch.Tensor:
    '''
    Triton GEMM kernel
    It equals:
        a = a if not transpose_a else a.transpose(0, 1)
        b = b if not transpose_b else b.transpose(0, 1)
        out = torch.matmul(a, b)

    Args:
        a (torch.Tensor): [M, K] after whether tranpose
        b (torch.Tensor): [N, K] after whether tranpose
        out (torch.Tensor): [M, N]
        acc: (bool): if out is provided, use atomic add on out
        transpose_a (bool): whether tranpose tensor a
        transpose_b (bool): whether tranpose tensor b
        persistent (bool): whether use persistent kernel
    Return:
        out (torch.Tensor): [M, N]

    '''
    assert a.stride(-1) == 1 and b.stride(-1) == 1

    if not transpose_a:
        M, K = a.shape
    else:
        K, M = a.shape

    if transpose_b:
        N, K2 = b.shape
    else:
        K2, N = b.shape

    assert K == K2

    if acc:
        assert out is not None and out.dtype == torch.float32


    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)


    kwargs = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps":8, "num_stages":3}

    if N < K or ((not transpose_a) and (not transpose_b)):
        kwargs = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps":4, "num_stages":4}
    elif M < K:
        kwargs = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps":4, "num_stages":4}

    if not transpose_a:
        desc_a = TensorDescriptor.from_tensor(a, (kwargs["BLOCK_M"], kwargs["BLOCK_K"]))
    else:
        desc_a = TensorDescriptor.from_tensor(a, (kwargs["BLOCK_K"], kwargs["BLOCK_M"]))
    if transpose_b:
        desc_b = TensorDescriptor.from_tensor(b, (kwargs["BLOCK_N"], kwargs["BLOCK_K"]))
    else:
        desc_b = TensorDescriptor.from_tensor(b, (kwargs["BLOCK_K"], kwargs["BLOCK_N"]))

    desc_c = TensorDescriptor.from_tensor(out, (kwargs["BLOCK_M"], kwargs["BLOCK_N"]))
    if not persistent:  
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )
        _gemm_kernel[grid](desc_a,
                        desc_b,
                        desc_c,
                        acc,
                        M,
                        N,
                        K,
                        transpose_a,
                        transpose_b,
                        **kwargs
                        )
    else:
        _persistent_gemm_kernel[(132, )](desc_a,
                        desc_b,
                        desc_c,
                        acc,
                        M,
                        N,
                        K,
                        transpose_a,
                        transpose_b,
                        **kwargs
                        )
    return out
