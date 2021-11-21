import torch
import torch.nn as nn

from .Flow import ConditionalFlow


class Shuffle(ConditionalFlow):
    def __init__(self, p):
        super(Shuffle, self).__init__()
        self.register_buffer("p", p)
        inv = torch.empty_like(p)
        inv[p] = torch.arange(len(p), device=p.device, dtype=p.dtype)
        self.register_buffer("inv_p", inv)

    def forward_flow(self, x, condition):
        return x[:, self.p],  0.

    def inverse_flow(self, x, condition):
        return x[:, self.inv_p], 0.


class Reverse(ConditionalFlow):
    def forward_flow(self, x, condition):
        return torch.flip(x, [len(x.shape) - 1]), 0.

    def inverse_flow(self, x, condition):
        return torch.flip(x, [len(x.shape) - 1]), 0.



def __inverse_permutation(p):
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.shape, device=p.device, dtype=p.dtype)
    return inv
