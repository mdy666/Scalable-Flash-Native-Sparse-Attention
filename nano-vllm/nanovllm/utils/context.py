from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

@dataclass
class NSAConstantContext:
    kernel_size: int = -1
    stride: int = -1
    block_size: int = -1
    top_n: int = -1
    num_init_blocks: int = -1
    num_local_blocks: int = -1
    window_size: int = -1
    fixed_y_maxlen: int = -1
    fixed_num_slc_blocks: int = -1
    cmp_num_splits: int = -1
    slc_num_splits: int = -1

_NSA_CONSTANT_CONTEXT = NSAConstantContext()

def get_nsa_constant_context():
    return _NSA_CONSTANT_CONTEXT

def set_nsa_constant_context(kernel_size, stride, block_size, top_n, num_init_blocks, num_local_blocks, window_size,
                             fixed_y_maxlen, fixed_num_slc_blocks, cmp_num_splits, slc_num_splits):
    global _NSA_CONSTANT_CONTEXT
    _NSA_CONSTANT_CONTEXT = NSAConstantContext(kernel_size, stride, block_size, top_n, num_init_blocks, num_local_blocks, window_size,
                                               fixed_y_maxlen, fixed_num_slc_blocks, cmp_num_splits, slc_num_splits)

def reset_nsa_constant_context():
    global _NSA_CONSTANT_CONTEXT
    _NSA_CONSTANT_CONTEXT = NSAConstantContext()

@dataclass
class NSADynamicContext:
    cmp_cu_seqlens: int = None
    cmp_max_seqlen: int = 0
    cmp_slot_mapping: torch.Tensor | None = None
    cmp_block_tables: torch.Tensor | None = None

_NSA_DYNAMIC_CONTEXT = NSADynamicContext()

def get_nsa_dynamic_context():
    return _NSA_DYNAMIC_CONTEXT

def set_nsa_dynamic_context(cmp_cu_seqlens=None, cmp_max_seqlen=None, cmp_slot_mapping=None, cmp_block_tables=None):
    global _NSA_DYNAMIC_CONTEXT
    _NSA_DYNAMIC_CONTEXT = NSADynamicContext(cmp_cu_seqlens, cmp_max_seqlen, cmp_slot_mapping, cmp_block_tables)

def reset_nsa_dynamic_context():
    global _NSA_DYNAMIC_CONTEXT
    _NSA_DYNAMIC_CONTEXT = NSADynamicContext()
