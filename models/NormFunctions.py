import torch
import torch.nn as nn

from .Flow import ConditionalFlow

class ActNormImage(ConditionalFlow):
    def __init__(self, dim):
        super(ActNormImage, self).__init__()
        self.mean = nn.Parameter(torch.zeros((dim, 1, 1), dtype=torch.float))
        self.log_s = nn.Parameter(torch.zeros((dim, 1, 1), dtype=torch.float))

    def forward_flow(self, x, condition):
        return (x - self.mean) * torch.exp(-self.log_s), -self.log_s.sum().repeat(x.shape[0]) * x.shape[2] * x.shape[3]

    def inverse_flow(self, x, condition):
        return x * torch.exp(self.log_s) + self.mean, self.log_s.sum().repeat(x.shape[0]) * x.shape[2] * x.shape[3]

    def data_init(self, x, condition):
        self.mean.data.copy_(x.mean(dim=(0, 2, 3))[:, None, None])
        d = ((x - self.mean) ** 2).mean(dim=(0, 2, 3))[:, None, None]
        self.log_s.data.copy_(torch.log(torch.sqrt(d) + 0.1))

        return self.forward_flow(x, condition)[0]


class ActNorm(ConditionalFlow):
    def __init__(self, dim):
        super(ActNorm, self).__init__()
        self.mean = nn.Parameter(torch.zeros((dim,), dtype=torch.float))
        self.log_s = nn.Parameter(torch.zeros((dim,), dtype=torch.float))

    def forward_flow(self, x, condition):
        return (x - self.mean) * torch.exp(-self.log_s), -self.log_s.sum().repeat(x.shape[0])

    def inverse_flow(self, x, condition):
        return x * torch.exp(self.log_s) + self.mean, self.log_s.sum().repeat(x.shape[0])

    def data_init(self, x, condition):
        self.mean.data.copy_(x.mean(dim=(0,)))
        d = ((x - self.mean) ** 2).mean(dim=(0,))
        self.log_s.data.copy_(torch.log(torch.sqrt(d) + 0.1))

        return self.forward_flow(x, condition)[0]



