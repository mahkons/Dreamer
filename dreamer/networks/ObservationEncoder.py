import torch
import torch.nn as nn

class ObservationEncoder(nn.Module):
    def __init__(self, obs_shape, state_size):
        super(ObservationEncoder, self)
        assert(len(obs_shape == 3))
