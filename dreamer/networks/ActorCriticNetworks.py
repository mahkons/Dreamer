import torch
import torch.nn as nn

from .common import MLP

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.GELU(),
            nn.Linear(400, 400),
            nn.GELU(),
            nn.Linear(400, 300),
            nn.GELU(),
        )

        self.fcmu = nn.Linear(300, action_dim)
        self.fclogs = nn.Linear(300, action_dim)

    def forward(self, state):
        x = self.model(state)
        mu, logs = self.fcmu(x), self.fclogs(x)
        eps = torch.randn_like(mu)

        return torch.tanh(mu + torch.exp(logs) * eps)
        


class CriticNetwork(MLP):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__(state_dim, 1, [400, 400, 300], nn.GELU)

    def forward(self, x):
        x = super().forward(x)
        return x.squeeze(len(x.shape) - 1)