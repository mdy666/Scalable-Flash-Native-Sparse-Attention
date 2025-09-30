import torch
import torch.distributed as dist
from .ops.reorder import reorder

from .utils import NSAHelper

def maybe_unshuffle(x, cp_mode):
    if cp_mode != 1:
        reorder(x, NSAHelper.world_size, shuffle_mode="unshuffle", inplace=True)

def maybe_shuffle(x, cp_mode):
    if cp_mode != 1:
        reorder(x, NSAHelper.world_size, shuffle_mode="shuffle", inplace=True)

class Comm:
    KEY='k'
    VALUE="v"
    GRAD_KEY="dk"
    GRAD_VALUE="dv"

    def __init__(self, process_group, buffer={}, cp_mode=1) -> None:
        assert cp_mode in [1, 2]
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.cp_mode = cp_mode
        self.buffer = buffer
        self.handles = {}

        for key in self.buffer:
            self.handles[key] = []

    @torch.no_grad()
    def all_gather(self, x, key):
        assert self.buffer[key] is not None
        x = x.contiguous()
        y = self.buffer[key]
        handle = dist.all_gather_into_tensor(y, x, group=self.process_group, async_op=True)
        self.handles[key].append(handle)
        return y

    @torch.no_grad()
    def reduce_scatter(self, x, key):
        assert self.buffer[key] is not None
        y = self.buffer[key].to(x.dtype)
        maybe_shuffle(y, self.cp_mode)
        handle = dist.reduce_scatter_tensor(x, y, group=self.process_group, async_op=True)
        self.handles[key].append(handle)
        return x

    def get_tensor(self, key):
        assert key in self.buffer
        return self.buffer[key]

    def wait(self, key):
        assert key in self.handles

        if len(self.handles[key]) == 0:
            return

        for handle in self.handles[key]:
            handle.wait()
        if key in [Comm.KEY, Comm.VALUE]:
            maybe_unshuffle(self.buffer[key], self.cp_mode)
        self.handles[key] = []

    def wait_all(self):
        for key in self.handles:
            self.wait(key)














