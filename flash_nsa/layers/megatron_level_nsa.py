import torch
import torch.nn as nn
import triton

from ..ops.rope import fused_apply_rope
from ..modules.megatron_level_nsa import NSACore
from ..config import NSAConfig

class NSA(nn.Module):
    '''
    It is just a demo, please replace related class or function in your need
    '''
    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim

        try:
            from megatron.core import parallel_state
            self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
            cp_process_group = parallel_state.get_context_parallel_group()
            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            assert self.num_heads % self.tp_size == 0 and self.num_kv_heads % self.tp_size == 0
        except:
            self.tp_size = 1
            cp_process_group = None
            pp_size = 1

        self.nsa = NSACore(config, process_group=cp_process_group, pp_size=pp_size)

        self.num_heads_per_rank = self.num_heads // self.tp_size
        self.num_kv_heads_per_rank = self.num_kv_heads // self.tp_size

        # if (num_head / num_kv_head) % 8 == 0, it will not pad
        output_dim = self.num_heads * (self.qk_head_dim + 3) + self.num_kv_heads * (self.qk_head_dim + self.v_head_dim)
        output_dim_per_rank = triton.cdiv(output_dim, self.tp_size)
        align_factor = 8
        output_dim_per_rank_align = triton.cdiv(output_dim_per_rank, align_factor) * align_factor
        output_dim_align = output_dim_per_rank_align * self.tp_size
        output_dim, output_dim_per_rank = output_dim_align, output_dim_per_rank_align
        split_size = [
            self.num_heads_per_rank * self.qk_head_dim, 
            self.num_kv_heads_per_rank * self.num_kv_heads,
            self.num_kv_heads_per_rank * self.v_head_dim,
            self.num_heads_per_rank * 3
        ]
        split_size = split_size + [output_dim_per_rank - sum(split_size)]
        self.split_size = split_size

        try:
            from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
            self.qkvw_proj = ColumnParallelLinear(
                self.hidden_size, 
                output_dim, 
                config=config, 
                init_method=config.init_method, 
                bias=False
            )
            self.o_proj = RowParallelLinear(
                self.v_head_dim * self.num_heads, 
                self.hidden_size, 
                config=config, 
                init_method=config.init_method, 
                bias=False
            )
        except:
            self.qkv_proj = nn.Linear(self.hidden_size, output_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor], 
        extra_kwargs: dict = {},
    ) -> torch.Tensor:
        # sequence first, [s/tp, b, d]
        s, b, d = hidden_states.shape

        qkvw = self.qkvw_proj(hidden_states)
        if not isinstance(qkvw, torch.Tensor):
            qkvw = qkvw[0]
        q, k, v, w, _ = qkvw.split(self.split_size, -1)
        q = q.view(*q.shape[:2], self.num_heads_per_rank, self.qk_head_dim)
        k = k.view(*k.shape[:2], self.num_kv_heads_per_rank, self.qk_head_dim)
        v = v.view(*v.shape[:2], self.num_kv_heads_per_rank, self.v_head_dim)
        combine_weight = w.view(*w.shape[:2], self.num_heads_per_rank, 3)

        cos, sin = rotary_pos_emb
        if cos.size(0) != q.size(0):
            # batch first
            cos = cos.transpose(0, 1)
            sin = sin.transpose(0, 1)
        # if batch>1, we create new contiguous q and k, then flatten op just a view op
        q, k = fused_apply_rope(q.transpose(0, 1), k.transpose(0, 1), cos, sin, inplace=(b==1))

        q = q.flatten(0, 1)
        k = k.flatten(0, 1)
        v = v.transpose(0, 1).flatten(0, 1)
        combine_weight = combine_weight.transpose(0, 1).flatten(0, 1)

        cu_seqlens = extra_kwargs.get("cu_seqlens", None)
        assert cu_seqlens is not None
        cu_seqlens_np = extra_kwargs.get("cu_seqlens_np", None)
        cp_mode = extra_kwargs.get("cp_mode", None)
        o = self.nsa(q, k, v, combine_weight, cu_seqlens=cu_seqlens, cu_seqlens_np=cu_seqlens_np, cp_mode=cp_mode)
        o = o.view(b, -1, self.num_heads_per_rank * self.v_head_dim).transpose(0, 1).contiguous()
        output = self.o_proj(o)
        return output


