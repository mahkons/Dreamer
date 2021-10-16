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

    def predict_logit(self, x):
        x = super().forward(x)
        return x.squeeze(len(x.shape) - 1)


class ObservationEncoder(nn.Module):
    def __init__(self, obs_dim, out_dim, from_pixels):
        super(ObservationEncoder, self).__init__()
        assert(not from_pixels)
        
        self.model = MLP(obs_dim, out_dim, [300, 300], nn.GELU)

    def forward(self, x):
        return self.model(x)

class ObservationDecoder(nn.Module):
    def __init__(self, in_dim, obs_dim, from_pixels):
        super(ObservationDecoder, self).__init__()
        assert(not from_pixels)
        
        self.model = MLP(in_dim, obs_dim, [300, 300], nn.GELU)

    def forward(self, x):
        return self.model(x)
