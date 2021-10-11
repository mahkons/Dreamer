import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, activation_func_module):
        super(MLP, self).__init__()
        layers = [nn.Linear(in_dim, hidden_dims[0]), activation_func_module()]
        layers += sum([[nn.Linear(ind, outd), activation_func_module()]
            for ind, outd in zip(hidden_dims, hidden_dims[1:])], [])
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
