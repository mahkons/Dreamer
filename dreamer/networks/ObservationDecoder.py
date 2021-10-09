import torch
import torch.nn

class ObservationDecoder(nn.Module):
    def __init__(self, state_size, obs_shape):
        super(ObservationDecoder, self)
