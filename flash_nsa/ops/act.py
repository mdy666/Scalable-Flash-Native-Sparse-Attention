# Copyright (c) 2025 Duyue Ma

import torch
import triton
import triton.language as tl

# @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_stages=ns, num_warps=nw)
#                           for BLOCK_SIZE in [1024, 2048, 4096, 8192]
#                           for ns in [1, 2, 4]
#                           for nw in [4, 8, 16]],
#                           key=['N'])
@triton.jit
def _swiglu_fwd(
    X, 
    Y,
    M: tl.constexpr,
    N: tl.constexpr, 
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):

    off = tl.program_id(0).cast(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    gate = tl.load(X + off % K + (off // K) * N, mask=off<M*K).to(tl.float32)
    up = tl.load(X + off % K + (off // K) * N + K, mask=off<M*K).to(tl.float32)
    act = gate * tl.sigmoid(gate)
    y = act * up
    tl.store(Y + off, y, mask=off<M*K)


# @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_stages=ns, num_warps=nw)
#                           for BLOCK_SIZE in [1024, 2048, 4096, 8192]
#                           for ns in [1, 2, 4]
#                           for nw in [4, 8, 16]],
#                           key=['N'])
@triton.jit
def _swiglu_bwd(
    X, 
    DY,
    M: tl.constexpr,
    N: tl.constexpr, 
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):

    off = tl.program_id(0).cast(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    gate = tl.load(X + off % K + (off // K) * N, mask=off<M*K).to(tl.float32)
    up = tl.load(X + off % K + (off // K) * N + K, mask=off<M*K).to(tl.float32)
    dy = tl.load(DY + off, mask=off<M*K, other=0.).to(tl.float32)
    gate_sigmoid = tl.sigmoid(gate)
    act = gate_sigmoid * gate
    dup = act * dy
    dact = up * dy
    dgate = (gate_sigmoid + act * (1-gate_sigmoid)) * dact
    tl.store(X + off % K + (off // K) * N + K, dup, mask=off<M*K)
    tl.store(X + off % K + (off // K) * N, dgate, mask=off<M*K)


def swiglu_fwd(x):
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, N = x.shape
    assert N % 2 == 0
    K = N // 2
    y = torch.empty(M, K, device=x.device, dtype=x.dtype)
    kwargs = {"BLOCK_SIZE":1024, "num_warps":8, "num_stages":2}
    grid = lambda meta: (triton.cdiv(M * K, meta['BLOCK_SIZE']), )
    _swiglu_fwd[(grid)](
        x, 
        y, 
        M,
        N,
        K,
        **kwargs
    )
    return y.view(*input_shape[:-1], K)

def swiglu_bwd(x, dy):
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, N = x.shape
    K = N // 2
    kwargs = {"BLOCK_SIZE":2048, "num_warps":16, "num_stages":2}
    grid = lambda meta: (triton.cdiv(M * K, meta['BLOCK_SIZE']), )
    _swiglu_bwd[grid](
        x,
        dy, 
        M,
        N,
        K,
        **kwargs
    )
    return x.view(*input_shape)

class _SwigluNoSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        y = swiglu_fwd(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dx = swiglu_bwd(x, dy)
        return dx


def swiglu_impl(x: torch.Tensor) -> torch.Tensor:
    '''
    swiglu, x = [gate, up]
    out = F.silu(gate) * up

    Args: 
        x (torch.tensor): [*, 2*D]
    Returns:
        out (torch.tensor): shape: [*, D]
    '''
    return _SwigluNoSplit.apply(x)


# @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_warps=nw)
#                           for BLOCK_SIZE in [512, 1024, 2048, 4096, 8192]
#                           for nw in [4, 8, 16]],
#                           key=['N'])
@triton.jit
def _sigmoid_mul_fwd(
    X, 
    Y,
    M: tl.constexpr,
    N: tl.constexpr, 
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):

    off = tl.program_id(0).cast(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    gate = tl.load(X + off % K + (off // K) * N, mask=off<M*K).to(tl.float32)
    up = tl.load(X + off % K + (off // K) * N + K, mask=off<M*K).to(tl.float32)
    act = tl.sigmoid(gate)
    y = act * up
    tl.store(Y + off, y, mask=off<M*K)


# @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_stages=ns, num_warps=nw)
#                           for BLOCK_SIZE in [1024, 2048, 4096, 8192]
#                           for ns in [1, 2, 4]
#                           for nw in [4, 8, 16]],
#                           key=['N'])
@triton.jit
def _sigmoid_mul_bwd(
    X, 
    DY,
    M: tl.constexpr,
    N: tl.constexpr, 
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):

    off = tl.program_id(0).cast(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    gate = tl.load(X + off % K + (off // K) * N, mask=off<M*K).to(tl.float32)
    up = tl.load(X + off % K + (off // K) * N + K, mask=off<M*K).to(tl.float32)
    dy = tl.load(DY + off, mask=off<M*K, other=0.).to(tl.float32)
    act = tl.sigmoid(gate)
    dup = act * dy
    dact = up * dy
    dgate = act * (1 - act) * dact
    tl.store(X + off % K + (off // K) * N + K, dup, mask=off<M*K)
    tl.store(X + off % K + (off // K) * N, dgate, mask=off<M*K)

def sigmoid_mul_fwd(x):
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, N = x.shape
    assert N % 2 == 0
    K = N // 2
    y = torch.empty(M, K, device=x.device, dtype=x.dtype)
    kwargs = {"BLOCK_SIZE":1024, "num_warps":8, "num_stages":2}
    grid = lambda meta: (triton.cdiv(M * K, meta['BLOCK_SIZE']), )
    _sigmoid_mul_fwd[(grid)](
        x, 
        y, 
        M,
        N,
        K,
        **kwargs
    )
    return y.view(*input_shape[:-1], K)

def sigmoid_mul_bwd(x, dy):
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, N = x.shape
    K = N // 2
    kwargs = {"BLOCK_SIZE":2048, "num_warps":16, "num_stages":2}
    grid = lambda meta: (triton.cdiv(M * K, meta['BLOCK_SIZE']), )
    _sigmoid_mul_bwd[grid](
        x,
        dy, 
        M,
        N,
        K,
        **kwargs
    )
    return x.view(*input_shape)

class _SigmoidMulNoSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        y = sigmoid_mul_fwd(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert dy.is_contiguous()
        x, = ctx.saved_tensors
        dx = sigmoid_mul_bwd(x, dy)
        return dx


def sigmoid_mul_impl(x):
    '''
    sigmoid-mul, x = [gate, up]
    out = F.sigmoid(gate) * up

    Args: 
        x (torch.tensor): [*, 2*D]
    Returns:
        out (torch.tensor): shape: [*, D]
    '''
    return _SigmoidMulNoSplit.apply(x)











