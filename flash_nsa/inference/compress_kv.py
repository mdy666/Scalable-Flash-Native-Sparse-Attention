import torch
import triton
import triton.language as tl

from .utils import split_d

# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                   for ns in [1, 2, 3, 4]
#                   for nw in [1, 2, 4, 8]
#                   ],
#                   key=['D1','D2',"stride"])
@triton.jit
def _prefill_kernel(
    X, 
    SLOT_MAPPING,
    TABELS,
    Y_CU_SEQLENS,
    sxk, sxn, sxh, sxd,
    table_stride,
    kernel_size: tl.constexpr, 
    stride: tl.constexpr, 
    PAGE_SIZE: tl.constexpr,
    D1: tl.constexpr, 
    D2: tl.constexpr,
):
    idx = tl.program_id(0)
    start_n = idx * stride
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    y_bos, y_eos = tl.load(Y_CU_SEQLENS + off_b), tl.load(Y_CU_SEQLENS + off_b + 1)
    y_len = y_eos - y_bos

    if idx >= y_len:
        return

    sxk = tl.cast(sxk, tl.int64)
    sxn = tl.cast(sxn, tl.int64)

    slot = tl.load(SLOT_MAPPING + y_bos + idx)
    if slot == -1:
        return

    y = tl.zeros((D1,), dtype=tl.float32)
    if D2 > 0:
        y2 = tl.zeros((D2,), dtype=tl.float32)
        
    for i in range(kernel_size//stride):
        tabel_idx = start_n // PAGE_SIZE
        cache_idx = tl.load(TABELS + off_b * table_stride + tabel_idx)
        off_n = (start_n + tl.arange(0, stride)) % PAGE_SIZE
        x = tl.load(X + cache_idx * sxk + off_n[:, None] * sxn + off_h * sxh + tl.arange(0, D1)[None, :]).to(tl.float32)
        y += tl.sum(x, 0)
        if D2 > 0:
            x2 = tl.load(X + cache_idx * sxk + off_n[:, None] * sxn + off_h * sxh + D1 + tl.arange(0, D2)[None, :]).to(tl.float32)
            y2 += tl.sum(x2, 0)
        start_n += stride
        
    tl.store(X + slot * sxn + off_h * sxh + tl.arange(0, D1), y / kernel_size)
    if D2 > 0:
        tl.store(X + slot * sxn + off_h * sxh + D1 + tl.arange(0, D2), y2 / kernel_size)
        
        
# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                   for ns in [1, 2, 3, 4]
#                   for nw in [1, 2, 4, 8]
#                   ],
#                   key=['D1','D2',"stride"])
@triton.jit
def _decode_kernel(
    X, 
    SLOT_MAPPING,
    TABELS,
    CONTEXT_LENS,
    sxk, sxn, sxh, sxd,
    table_stride,
    kernel_size: tl.constexpr, 
    stride: tl.constexpr, 
    PAGE_SIZE: tl.constexpr,
    D1: tl.constexpr, 
    D2: tl.constexpr,
):

    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    
    slot = tl.load(SLOT_MAPPING + off_b)
    if slot == -1:
        return
    
    sxk = tl.cast(sxk, tl.int64)
    sxn = tl.cast(sxn, tl.int64)
    
    x_len = tl.load(CONTEXT_LENS + off_b)
    idx = (x_len - kernel_size) // stride
    start_n = idx * stride

    y = tl.zeros((D1,), dtype=tl.float32)
    if D2 > 0:
        y2 = tl.zeros((D2,), dtype=tl.float32)
        
    for i in range(kernel_size//stride):
        tabel_idx = start_n // PAGE_SIZE
        cache_idx = tl.load(TABELS + off_b * table_stride + tabel_idx)
        off_n = (start_n + tl.arange(0, stride)) % PAGE_SIZE
        x = tl.load(X + cache_idx * sxk + off_n[:, None] * sxn + off_h * sxh + tl.arange(0, D1)[None, :]).to(tl.float32)
        y += tl.sum(x, 0)
        if D2 > 0:
            x2 = tl.load(X + cache_idx * sxk + off_n[:, None] * sxn + off_h * sxh + D1 + tl.arange(0, D2)[None, :])
            y2 += tl.sum(x2, 0)
        start_n += stride
        
    tl.store(X + slot * sxn + off_h * sxh + tl.arange(0, D1), y / kernel_size)
    if D2 > 0:
        tl.store(X + slot * sxn + off_h * sxh + D1 + tl.arange(0, D2), y2 / kernel_size)
        
# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                   for ns in [1, 2, 3, 4]
#                   for nw in [1, 2, 4, 8]
#                   ],
#                   key=['D1','D2',"stride"])
@triton.jit
def _kv_decode_kernel(
    K,
    V, 
    SLOT_MAPPING,
    TABELS,
    CONTEXT_LENS,
    skk, skn, skh, skd,
    svk, svn, svh, svd,
    table_stride,
    kernel_size: tl.constexpr, 
    stride: tl.constexpr, 
    PAGE_SIZE: tl.constexpr,
    D1: tl.constexpr, 
    D2: tl.constexpr,
    VD: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    pid = tl.program_id(2)
    
    slot = tl.load(SLOT_MAPPING + off_b)
    if slot == -1:
        return
    
    x_len = tl.load(CONTEXT_LENS + off_b)
    if x_len <= kernel_size - 1:
        return
    idx = (x_len - kernel_size) // stride
    start_n = idx * stride
        
    skk = tl.cast(skk, tl.int64)
    skn = tl.cast(skn, tl.int64)
    svk = tl.cast(svk, tl.int64)
    svn = tl.cast(svn, tl.int64)
    
    if pid == 0:
        y = tl.zeros((D1,), dtype=tl.float32)
        if D2 > 0:
            y2 = tl.zeros((D2,), dtype=tl.float32)
            
        for i in range(kernel_size//stride):
            tabel_idx = start_n // PAGE_SIZE
            cache_idx = tl.load(TABELS + off_b * table_stride + tabel_idx)
            off_n = (start_n + tl.arange(0, stride)) % PAGE_SIZE
            x = tl.load(K + cache_idx * skk + off_n[:, None] * skn + off_h * skh + tl.arange(0, D1)[None, :]).to(tl.float32)
            y += tl.sum(x, 0)
            if D2 > 0:
                x2 = tl.load(K + cache_idx * skk + off_n[:, None] * skn + off_h * skh + D1 + tl.arange(0, D2)[None, :])
                y2 += tl.sum(x2, 0)
            start_n += stride
            
        tl.store(K + slot * skn + off_h * skh + tl.arange(0, D1), y / kernel_size)
        if D2 > 0:
            tl.store(K + slot * skn + off_h * skh + D1 + tl.arange(0, D2), y2 / kernel_size)
    else:
        y = tl.zeros((VD,), dtype=tl.float32)
            
        for i in range(kernel_size//stride):
            tabel_idx = start_n // PAGE_SIZE
            cache_idx = tl.load(TABELS + off_b * table_stride + tabel_idx)
            off_n = (start_n + tl.arange(0, stride)) % PAGE_SIZE
            x = tl.load(V + cache_idx * svk + off_n[:, None] * svn + off_h * svh + tl.arange(0, VD)[None, :]).to(tl.float32)
            y += tl.sum(x, 0)
            start_n += stride
            
        tl.store(V + slot * svn + off_h * svh + tl.arange(0, VD), y / kernel_size)
        
        
def mean_pooling_prefill(
    x: torch.Tensor, 
    y_cu_seqlens: torch.Tensor, 
    y_maxlen: torch.Tensor,
    slot_mapping: torch.Tensor, 
    block_table: torch.Tensor, 
    kernel_size: int = 32,
    stride: int = 16,
) -> torch.Tensor:
    
    _, PAGE_SIZE, H, D = x.shape
    D1, D2 = split_d(D)
    B = len(y_cu_seqlens) - 1
    kwargs = {'num_warps':1, 'num_stages': 1}
    grids = (y_maxlen, B, H)
    _prefill_kernel[grids](
        x, 
        slot_mapping,
        block_table,
        y_cu_seqlens,
        *x.stride(),
        block_table.stride(0),
        kernel_size, 
        stride, 
        PAGE_SIZE,
        D1, 
        D2, 
        **kwargs
    )
    return x

def mean_pooling_decode(
    x: torch.Tensor, 
    slot_mapping: torch.Tensor, 
    block_table: torch.Tensor, 
    context_lens: torch.Tensor = None, 
    kernel_size: int = 32,
    stride: int = 16,
) -> torch.Tensor:
    
    _, PAGE_SIZE, H, D = x.shape
    D1, D2 = split_d(D)

    B = len(context_lens)
    kwargs = {'num_warps':1, 'num_stages': 1}
    grids = (B, H)
    _decode_kernel[grids](
        x, 
        slot_mapping,
        block_table,
        context_lens,
        *x.stride(),
        block_table.stride(0),
        kernel_size, 
        stride, 
        PAGE_SIZE,
        D1, 
        D2, 
        **kwargs
    )
    return x

def kv_mean_pooling_decode(
    k: torch.Tensor, 
    v: torch.Tensor,
    slot_mapping: torch.Tensor, 
    block_table: torch.Tensor, 
    context_lens: torch.Tensor = None, 
    kernel_size: int = 32,
    stride: int = 16,
) -> torch.Tensor:
    
    _, PAGE_SIZE, H, D = k.shape
    VD = v.size(-1)
    D1, D2 = split_d(D)

    B = len(context_lens)
    kwargs = {'num_warps':1, 'num_stages': 1}
    grids = (B, H, 2)
    _kv_decode_kernel[grids](
        k, 
        v,
        slot_mapping,
        block_table,
        context_lens,
        *k.stride(),
        *v.stride(),
        block_table.stride(0),
        kernel_size, 
        stride, 
        PAGE_SIZE,
        D1, 
        D2, 
        VD,
        **kwargs
    )
    return k, v