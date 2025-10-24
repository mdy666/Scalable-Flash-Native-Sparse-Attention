# Copyright (c) 2025 Xingkai Yu

import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context, get_nsa_constant_context, get_nsa_dynamic_context

try:
    from flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_v3
    HAVE_FA3 = True
except:
    HAVE_FA3 = False

try:
    from flash_attn import flash_attn_with_kvcache as flash_attn_with_kvcache_v2, flash_attn_varlen_func as flash_attn_varlen_func_v2
    HAVE_FA2 = True
except:
    HAVE_FA2 = False


from flash_nsa import nsa_prefill, nsa_decode



def nano_vllm_nsa_prefill(q, k, v, w, sm_scale):
    context = get_context()
    nsa_dynamic_context = get_nsa_dynamic_context()
    nsa_constant_context = get_nsa_constant_context()
    x_cu_seqlens = context.cu_seqlens_q
    y_cu_seqlens = nsa_dynamic_context.cmp_cu_seqlens
    x_maxlen = context.max_seqlen_q
    y_maxlen = nsa_dynamic_context.cmp_max_seqlen
    context_lens = context.context_lens
    block_tables = context.block_tables
    cmp_slot_mapping = nsa_dynamic_context.cmp_slot_mapping
    cmp_block_tables = nsa_dynamic_context.cmp_block_tables
    kernel_size = nsa_constant_context.kernel_size
    stride = nsa_constant_context.stride
    block_size = nsa_constant_context.block_size
    top_n = nsa_constant_context.top_n
    num_init_blocks = nsa_constant_context.num_init_blocks
    num_local_blocks = nsa_constant_context.num_local_blocks
    window_size = nsa_constant_context.window_size
    fixed_num_slc_blocks = nsa_constant_context.fixed_num_slc_blocks
    fixed_y_maxlen = nsa_constant_context.fixed_y_maxlen

    o = nsa_prefill(
        q,
        k,
        v,
        w,
        x_cu_seqlens=x_cu_seqlens,
        y_cu_seqlens=y_cu_seqlens,
        x_maxlen=x_maxlen,
        y_maxlen=y_maxlen,
        context_lens=context_lens,
        block_tables=block_tables,
        cmp_slot_mapping=cmp_slot_mapping,
        cmp_block_tables=cmp_block_tables,
        fixed_num_slc_blocks=fixed_num_slc_blocks,
        fixed_y_maxlen=fixed_y_maxlen,
        kernel_size=kernel_size,
        stride=stride,
        block_size=block_size,
        top_n=top_n,
        num_init_blocks=num_local_blocks,
        num_local_blocks=num_init_blocks,
        window_size=window_size,
        sm_scale=sm_scale
    )

    return o

def nano_vllm_nsa_decode(q, k, v, w, sm_scale):
    context = get_context()
    nsa_dynamic_context = get_nsa_dynamic_context()
    nsa_constant_context = get_nsa_constant_context()
    context_lens = context.context_lens
    block_tables = context.block_tables
    cmp_slot_mapping = nsa_dynamic_context.cmp_slot_mapping
    cmp_block_tables = nsa_dynamic_context.cmp_block_tables
    kernel_size = nsa_constant_context.kernel_size
    stride = nsa_constant_context.stride
    block_size = nsa_constant_context.block_size
    top_n = nsa_constant_context.top_n
    num_init_blocks = nsa_constant_context.num_init_blocks
    num_local_blocks = nsa_constant_context.num_local_blocks
    window_size = nsa_constant_context.window_size
    fixed_num_slc_blocks = nsa_constant_context.fixed_num_slc_blocks
    fixed_y_maxlen = nsa_constant_context.fixed_y_maxlen
    cmp_num_splits = nsa_constant_context.cmp_num_splits
    slc_num_splits= nsa_constant_context.slc_num_splits
    
    o = nsa_decode(
        q,
        k,
        v,
        w,
        context_lens=context_lens,
        block_tables=block_tables,
        cmp_slot_mapping=cmp_slot_mapping,
        cmp_block_tables=cmp_block_tables,
        fixed_y_maxlen=fixed_y_maxlen,
        fixed_num_slc_blocks=fixed_num_slc_blocks,
        cmp_num_splits=cmp_num_splits,
        slc_num_splits=slc_num_splits,
        kernel_size=kernel_size,
        stride=stride,
        block_size=block_size,
        top_n=top_n,
        num_init_blocks=num_local_blocks,
        num_local_blocks=num_init_blocks,
        window_size=window_size,
        sm_scale=sm_scale
    )

    return o



@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        nsa,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.nsa = nsa

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor=None):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if not self.nsa:
            if not k_cache.numel() and not v_cache.numel():
                t, h, d = q.shape
                vd = v.size(-1)
                o = torch.randn(t, h, vd, device=q.device, dtype=q.dtype)
            elif context.is_prefill:
                if HAVE_FA3:
                    o = flash_attn_with_kvcache_v3(q, k_cache, v_cache, cu_seqlens_q=context.cu_seqlens_q, max_seqlen_q=context.max_seqlen_q,
                                                cache_seqlens=context.context_lens, page_table=context.block_tables, 
                                                softmax_scale=self.scale, causal=True)
                else:
                    if context.block_tables is not None:    # prefix cache
                        k, v = k_cache, v_cache
                    o = flash_attn_varlen_func_v2(q, k, v,
                                                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                                softmax_scale=self.scale, causal=True, block_table=context.block_tables)

            else:    # decode
                if HAVE_FA3:
                    o = flash_attn_with_kvcache_v3(q.unsqueeze(1), k_cache, v_cache,
                                                cache_seqlens=context.context_lens, page_table=context.block_tables, 
                                                softmax_scale=self.scale, causal=True)
                else:
                    o = flash_attn_with_kvcache_v2(q.unsqueeze(1), k_cache, v_cache,
                                                cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                                softmax_scale=self.scale, causal=True)

        else:
            if not k_cache.numel() and not v_cache.numel():
                # now must provide block_kv_cache. so we init the tensor directly
                t, h, d = q.shape
                kh, vd = v.shape[1:]
                p = torch.zeros(kh, t, 4096, device=q.device, dtype=q.dtype)
                o3 = torch.randn(3, t, h, vd, device=q.device, dtype=q.dtype)
                o = o3.sum(0)
            elif context.is_prefill:
                o = nano_vllm_nsa_prefill(q, k_cache, v_cache, w, sm_scale=self.scale)
            else:
                o = nano_vllm_nsa_decode(q, k_cache, v_cache, w, sm_scale=self.scale)
        return o
