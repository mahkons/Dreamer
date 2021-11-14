import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaskedLinear(nn.Module):
    def __init__(self, in_sz, out_sz, mask):
        super(MaskedLinear, self).__init__()
        self.register_buffer("mask", mask)
        assert(mask.shape == (out_sz, in_sz))
        self.W = nn.Parameter(torch.randn((out_sz, in_sz)) * math.sqrt(2. / (in_sz + out_sz)))
        self.bias = nn.Parameter(torch.zeros(out_sz))

    def forward(self, x):
        return F.linear(x, self.W * self.mask, self.bias)


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask,
            stride=1, padding=0, dilation=1, groups=1, bias=True,
            padding_mode="zeros"):
        super(MaskedConv2d, self).__init__()
        # TODO
        self.register_buffer("mask", mask)
        assert(len(mask.shape) == 3)
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
