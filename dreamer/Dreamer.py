import numpy as np
import torch
import math

from WorldModel import WorldModel
from ActorCritic import ActorCritic

class Dreamer():
    def __init__(self, state_dim, action_dim, device):
        # yet without images
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        HIDDEN_DIM = 64
        self.world_model = WorldModel(state_dim, action_dim, device)
        self.agent = ActorCritic(HIDDEN_DIM, action_dim, device)

    def __call__(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float).to(self.device)
            state = self.world_model.encoder(state)
            action, mu = self.agent.act(state, isTrain=False)
            
            mu = mu.cpu()
            return torch.tanh(mu + torch.randn_like(mu) * math.sqrt(0.3)) # superb exploration

    def optimize(self, batch_seq):
        state, action, reward, done = batch_seq

        self.world_model.optimize(state, action, reward, done)
        self.agent.optimize(self.world_model, state[:-1].reshape(-1, self.state_dim))



