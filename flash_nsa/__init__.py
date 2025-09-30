from .utils import NSAHelper
from .config import NSAConfig

from .modules.hf_level_nsa import (
    HFNSACore,
    select_attn,
    slc_topk_indices,
    compress_attn,
    mean_pooling,
    construct_block,
    swa_varlen_func,
    )

from .modules.megatron_level_nsa import (
    NSACore,
    flash_nsa_varlen_func,
    cp_flash_nsa_varlen_func,
    )

from .layers.megatron_level_nsa import NSA
from .layers.hf_level_nsa import HFNSA

from .inference import nsa_prefill, nsa_decode

