import torch
import triton
from functools import wraps
from functools import lru_cache
from ..utils import NSAHelper, use_tma

@lru_cache(5)
def split_d(d):
    d_power_2 = triton.next_power_of_2(d)
    if d == d_power_2:
        return d, 0
    else:
        d1 = d_power_2 // 2
        d2 = d - d1
        assert d2 == triton.next_power_of_2(d2)
        return d1, d2
    

