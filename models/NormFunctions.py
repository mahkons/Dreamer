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
        d = torch.var(x, dim=(0, 2, 3))[:, None, None]
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
        d = torch.var(x, dim=0)
        self.log_s.data.copy_(torch.log(torch.sqrt(d) + 0.1))

        return self.forward_flow(x, condition)[0]


class RunningBatchNorm1d(ConditionalFlow):
    def __init__(self, dim, tau=0.1, eps=1e-5): #TODO works poorly with tau != 1 /shrug
        super(RunningBatchNorm1d, self).__init__()
        self.register_buffer("m", torch.zeros((dim,), dtype=torch.float))
        self.register_buffer("s", torch.ones((dim,), dtype=torch.float))
        self.tau = tau
        self.eps = eps

    def forward_flow(self, x, condition):
        if not self.training:
            return (x - self.m) / torch.sqrt(self.s + self.eps), -0.5 * torch.log(self.s + self.eps).sum().repeat(x.shape[0])
        nm = x.mean(dim=0)
        ns = torch.var(x, dim=0)

        with torch.no_grad():
            self.m = nm * self.tau + (1 - self.tau) * self.m
            self.s = ns * self.tau + (1 - self.tau) * self.s

        return (x - nm) / torch.sqrt(ns + 1e-5), -0.5 * torch.log(ns + 1e-5).sum().repeat(x.shape[0])

    def inverse_flow(self, x, condition):
        assert(not self.training)
        return x * torch.sqrt(self.s + 1e-5) + self.m, 0.5 * torch.log(self.s + 1e-5).sum().repeat(x.shape[0])

    def data_init_(self, x, condition):
        return self.forward_flow(x, condition)[0]


