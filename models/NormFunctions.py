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
    def __init__(self, dim, tau=0.9):
        super(RunningBatchNorm1d, self).__init__()
        self.register_buffer("m", torch.zeros((dim,), dtype=torch.float))
        self.register_buffer("s", torch.ones((dim,), dtype=torch.float))
        self.tau = tau

    def forward_flow(self, x, condition):
        cur_m = x.mean(dim=0)
        cur_s = torch.var(x, dim=0)

        # backprop through cur_m, cur_s as in RealNVP
        nm = cur_m * self.tau + (1 - self.tau) * self.m
        ns = cur_s * self.tau + (1 - self.tau) * self.s

        # TODO add train/eval or smth
        if x.shape[0] == 1:
            nm, ns = self.m, self.s

        with torch.no_grad():
            self.m.copy_(nm)
            self.s.copy_(ns)

        return (x - nm) / torch.sqrt(ns + 1e-5), -0.5 * torch.log(ns + 1e-5).sum().repeat(x.shape[0])

    def inverse_flow(self, x, condition):
        return x * torch.sqrt(self.s + 1e-5) + self.m, 0.5 * torch.log(self.s + 1e-5).sum().repeat(x.shape[0])

    def data_init_(self, x, condition):
        return self.forward_flow(x, condition)[0]


