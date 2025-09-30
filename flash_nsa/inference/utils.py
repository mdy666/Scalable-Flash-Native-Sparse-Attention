import torch
import triton
from functools import wraps
from functools import lru_cache

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
    
def set_allocator():
    device = torch.cuda.current_device()
    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device=device)
    triton.set_allocator(alloc_fn)

def use_tma(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        set_allocator()
        
        return func(*args, **kwargs)
    return wrapper