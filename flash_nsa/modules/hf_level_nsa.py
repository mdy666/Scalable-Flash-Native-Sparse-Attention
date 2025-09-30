import math
import torch
import triton
import pandas as pd

from ..ops.compress_kv import construct_block, mean_pooling
from ..ops.compress_attn import compress_attn
from ..ops.select_attn import select_attn
from ..ops.topk import slc_topk_indices
from ..ops.act import swiglu_impl, sigmoid_mul_impl
from ..ops.combine import fused_sigmoid_combine
from ..ops.sliding_window_attention import swa_varlen_func

from ..utils import NSAHelper
from ..config import NSAConfig

class CompressKV(torch.nn.Module):
    def __init__(self, config: NSAConfig, head_dim=128, version=None):
        super().__init__()
        self.head_dim = head_dim
        self.kernel_size = getattr(config, "kernel_size", 32)
        self.stride =getattr(config, "stride", 32)
        assert version in ["mean", "sigmoid-mul", "swiglu", "linear"]
        self.version = version

        if self.version != "mean":
            if self.version in ["swiglu", "sigmoid-mul"]:
                out_features = 2 * head_dim
            elif self.version == "linear":
                out_features = head_dim
            self.fc = torch.nn.Linear(self.kernel_size * self.head_dim, out_features, bias=False)

    def forward(self, x):
        '''
        If you want to change the compress method, witre your code base on the block_x
        '''

        # x.shape: [t, h, d]
        if self.version=="mean":
            # [y_cu_seqlens[-1], h, d]
            return mean_pooling(x)

        # block_x.shape: [y_cu_seqlens[-1], h, kernel_size, d]
        # mean_pooling(x) <==> block_x.sum(-2)
        block_x = construct_block(x)

        # flatten_block_x.shape: [y_cu_seqlens[-1], h, kernel_size * d]
        hidden_state = self.fc(torch.flatten(block_x, start_dim=-2, end_dim=-1))

        if self.version == "linear":
            # [y_cu_seqlens[-1], h, d]
            out = hidden_state
        elif self.version == "sigmoid-mul":
            # [y_cu_seqlens[-1], h, 2 * d]
            out = sigmoid_mul_impl(hidden_state)
            # [y_cu_seqlens[-1], h, d]
        elif self.version == "swiglu":
            out = swiglu_impl(hidden_state)

        return out

class HFNSACore(torch.nn.Module):
    """
    native sparse attention module for huggingface level users.
    You can change the compress method and combine method according to your design.
    """
    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config
        self.qk_head_dim = config.qk_head_dim if config.qk_head_dim is not None else config.head_dim
        self.v_head_dim = config.v_head_dim if config.v_head_dim is not None else config.head_dim
        self.kernel_size = getattr(config, "kernel_size", 32)
        self.stride = getattr(config, "stride", 16)
        self.block_size = getattr(config, "block_size", 64)
        self.top_n = getattr(config, "top_n", 16)
        self.num_local_blocks = getattr(config, "num_local_blocks", 1)
        self.num_init_blocks = getattr(config, "num_init_blocks", 2)
        self.window_size = getattr(config, "window_size", 512)

        self.sm_scale = self.qk_head_dim ** -0.5

        assert math.log2(self.stride).is_integer(), "stride must be a power of 2, like 8, 16 etc."
        assert self.kernel_size % self.stride == 0
        assert self.block_size % self.stride == 0

        NSAHelper.set_hyperparameters(
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            block_size=self.block_size, 
            window_size=self.window_size, 
            top_n=self.top_n, 
            num_init_blocks=self.num_init_blocks, 
            num_local_blocks=self.num_local_blocks
        )

        self.compress_k = CompressKV(config, head_dim=self.qk_head_dim, version=config.cmp_k_method)
        self.compress_v = CompressKV(config, head_dim=self.v_head_dim, version=config.cmp_v_method)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        combine_weight: torch.Tensor, 
        cu_seqlens: torch.Tensor, 
        bench: bool = False,
        print_result: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for the NSA Attention module.

        Args:
            q (torch.Tensor): [t, num_q_head, qk_head_dim]
            k (torch.Tensor): [t, num_kv_head, qk_head_dim]
            v (torch.Tensor): [t, num_kv_head, v_head_dim]
            combine_weight (torch.Tensor): [t, num_q_head, 3]
            cu_seqlens (Int32 tensor): [batch+1].
            bench (bool): Whether to benchmark the operations.

        Return:
            o (torch.Tensor): [t, num_q_head, v_head_dim]
        """

        # init nsa helper if need
        assert cu_seqlens is not None
        NSAHelper.init_cu_seqlens(cu_seqlens,)

        # compress_kv
        cmp_k = self.compress_k(k)
        cmp_v = self.compress_v(v)
        # compute compress attention
        cmp_o, cmp_lse = compress_attn(q, cmp_k, cmp_v, sm_scale=self.sm_scale)
        # compute topk indices
        topk, _ = slc_topk_indices(q, cmp_k, cmp_lse, sm_scale=self.sm_scale, maybe_efficient_version=True, scale_slc_p=True)
        # compute select attention
        slc_o = select_attn(q, k, v, topk, sm_scale=self.sm_scale)
        # compute sliding window attention
        swa_o = swa_varlen_func(q, k, v, self.sm_scale)
        # combine the 3 attention outputs
        combine_o = fused_sigmoid_combine(cmp_o, slc_o, swa_o, combine_weight)

        if not bench:
            return combine_o

        t1 = triton.testing.do_bench(lambda: self.compress_k(k))
        t2 = triton.testing.do_bench(lambda: self.compress_v(v))
        t3 = triton.testing.do_bench(lambda: compress_attn(q, cmp_k, cmp_v, sm_scale=self.sm_scale))
        t4 = triton.testing.do_bench(lambda: slc_topk_indices(q, cmp_k, cmp_lse, sm_scale=self.sm_scale, maybe_efficient_version=True))
        t5 = triton.testing.do_bench(lambda: select_attn(q, k, v, topk, sm_scale=self.sm_scale))
        t6 = triton.testing.do_bench(lambda: swa_varlen_func(q, k, v, self.sm_scale))
        t7 = triton.testing.do_bench(lambda: fused_sigmoid_combine(cmp_o, slc_o, swa_o, combine_weight))

        result = {"cmp_k": t1, "cmp_v":t2, "cmp_o":t3, "top_k": t4, "slc_o": t5, "swa_o": t6, "combine":t7}
        df = pd.Series(result)
        if print_result:
            print("forward time (ms) s: ")
            print(df)
        return combine_o, df


