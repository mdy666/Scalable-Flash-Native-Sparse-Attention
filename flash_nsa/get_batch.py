import torch

from typing import Dict, Any
from megatron.core import parallel_state
from megatron.training import get_args

'''
A demo for split batch when use cp.
'''

def get_cp_mode_from_cu_seqlens(cu_seqlens: torch.Tensor, threshold: int = None) -> int:
    '''
    How to choose cp_mode=1 or cp_mode=2.
    Please rewrite the condition if you have a better choice.
    '''
    cp_size = parallel_state.get_context_parallel_world_size()
    S = cu_seqlens[-1].item()
    if threshold is None:
        threshold = S // cp_size * 1.5
        # threshold = S // cp_size * 2
        # threshold = S // 1.5
        # threshold = S // 2
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_len = seqlens.max().item()
    if max_len >= threshold:
        cp_mode = 2
    else:
        cp_mode = 1
    return cp_mode

def get_batch_on_this_cp_rank_for_nsa(batch: Dict[str, Any]):
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size > 1:
        args = get_args()
        cp_rank = parallel_state.get_context_parallel_rank()
        if args.use_auto_cp_mode:
            cp_mode = get_cp_mode_from_cu_seqlens(batch["cu_seqlens"])
        else:
            cp_mode = args.cp_mode
        batch["cp_mode"] = cp_mode
        keys = ["input_ids", "labels", "position_ids"]
        for key in keys:
            val = batch[key]
            b, s = val.shape
            val = val
            val = val.chunk(cp_size * cp_mode)
            if cp_mode == 1:
                val = val[cp_rank]
            else:
                val = torch.cat([val[cp_rank], val[2 * cp_size - 1 - cp_rank]])
            val = val.view(b, -1)
            batch[key] = val
    return batch