import torch
import torch.nn as nn

from .common import MLP


class RewardNetwork(MLP):
    def __init__(self, state_dim):
        super(RewardNetwork, self).__init__(state_dim, 1, [300, 300], nn.GELU)

    def forward(self, x):
        x = super().forward(x)
        return x.squeeze(len(x.shape) - 1)



class DiscountNetwork(MLP):
    def __init__(self, state_dim):
        super(DiscountNetwork, self).__init__(state_dim, 1, [300, 300], nn.GELU)

    def forward(self, x):
        assert(False)

    def predict_log(self, x):
        x = super().forward(x)
        return x.squeeze(len(x.shape) - 1)
