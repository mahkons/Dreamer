import torch
import torch.nn as nn

from .Flow import ConditionalFlow

class CouplingLayer(ConditionalFlow):
    def __init__(self, input_dim, condition_dim, hidden_shape, num_hidden, mask):
        super(CouplingLayer, self).__init__()
        assert(mask.shape == torch.Size([input_dim]))
        assert(len(mask.shape) == 1)

        self.register_buffer("mask", mask)
        self.register_parameter("log_scale_scale", nn.Parameter(torch.tensor(0., dtype=torch.float)))

        modules = [nn.utils.weight_norm(nn.Linear(input_dim + condition_dim, hidden_shape)), nn.ReLU()] \
            + sum([[nn.utils.weight_norm(nn.Linear(hidden_shape, hidden_shape)), nn.ReLU()] for _ in range(num_hidden)], []) \
            + [nn.utils.weight_norm(nn.Linear(hidden_shape, 2 * input_dim))]

        self.model = nn.Sequential(*modules)


    def forward_flow(self, x, condition):
        masked_x = x * self.mask
        input = torch.cat([masked_x, condition], dim=-1)
        log_s, t = self.model(input).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * self.log_scale_scale
        return masked_x + (1 - self.mask) * (x * torch.exp(log_s) + t), (log_s * (1 - self.mask)).sum(dim=1)

    def inverse_flow(self, x, condition):
        masked_x = x * self.mask
        input = torch.cat([masked_x, condition], dim=-1)
        log_s, t = self.model(input).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * self.log_scale_scale
        return masked_x + (1 - self.mask) * ((x - t) * torch.exp(-log_s)), -(log_s * (1 - self.mask)).sum(dim=1)
