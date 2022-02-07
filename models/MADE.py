import torch
import torch.nn as nn

from .Flow import ConditionalFlow
from .MaskedLayers import MaskedLinear

class MADE(ConditionalFlow):
    def __init__(self, dim, condition_dim, hidden_dim):
        super(MADE, self).__init__()

        assert(hidden_dim >= dim > 2)

        order_input = torch.arange(dim)
        order_hidden = torch.arange(hidden_dim) % (dim - 2) + 1
        order_out = torch.arange(2 * dim) % dim

        self.model = nn.Sequential(
            MaskedLinear(dim + condition_dim, hidden_dim, 
                torch.cat([order_hidden[:, None] >= order_input[None],
                    torch.ones((hidden_dim, condition_dim), dtype=torch.bool, device=self.device)], dim=1)),
            nn.ELU(),
            MaskedLinear(hidden_dim, hidden_dim, order_hidden[:, None] >= order_hidden[None]),
            nn.ELU(),
            MaskedLinear(hidden_dim, 2 * dim, order_out[:, None] > order_hidden[None])
        )
        #self.log_scale_scale = nn.Parameter(torch.tensor(0., dtype=torch.float))

    def forward_flow(self, x, condition):
        log_s, t = self.model(torch.cat([x, condition], dim=1)).chunk(2, dim=1)
        log_s = torch.tanh(log_s)
        return x * torch.exp(log_s) + t, log_s.sum(dim=1)

    def inverse_flow(self, u, condition):
        x = torch.zeros_like(u)
        for i in range(0, x.shape[1]):
            log_s, t = self.model(torch.cat([x, condition], dim=1)).chunk(2, dim=1)
            log_s = torch.tanh(log_s)
            x[:, i] = (u[:,i] - t[:,i]) * torch.exp(-log_s[:,i])
        return x, -log_s.sum(dim=1)

