import torch
import torch.nn as nn

from ..ops.rope import fused_apply_rope
from ..modules.hf_level_nsa import HFNSACore
from ..config import NSAConfig

class HFNSA(nn.Module):
    '''
    just a demo
    '''
    def __init__(self, config: NSAConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.qk_head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.combine_proj = nn.Linear(self.hidden_size, self.num_heads * 3, bias=False)

        self.nsa = HFNSACore(config)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        position_embeddings: tuple[torch.Tensor, torch.Tensor], 
        cu_seqlens: torch.Tensor = None
    ) -> torch.Tensor:

        b, s, d = hidden_states.shape

        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.qk_head_dim)
        k = self.k_proj(hidden_states).view(b, s, self.num_kv_heads, self.qk_head_dim)
        v = self.v_proj(hidden_states).view(b, s, self.num_kv_heads, self.v_head_dim)
        combine_weight = self.combine_proj(hidden_states).view(b, s, self.num_heads, 3)

        cos, sin = position_embeddings
        q, k = fused_apply_rope(q, k, cos, sin, inplace=True)

        q = q.flatten(0, 1)
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)
        combine_weight = combine_weight.flatten(0, 1)

        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0] + [s] * b, device=q.device).cumsum(-1).to(torch.int32)

        o = self.nsa(q, k, v, combine_weight, cu_seqlens)

        output = self.o_proj(o.view(b, s, -1))
        return output


