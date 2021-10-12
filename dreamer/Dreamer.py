import numpy as np
import torch

from WorldModel import WorldModel
from ActorCritic import ActorCritic

class Dreamer():
    def __init__(self, state_dim, action_dim, device):
        # yet without images
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.world_model = WorldModel(state_dim, action_dim, device)
        self.agent = ActorCritic(state_dim, action_dim, device)

    def __call__(self, state):
        with torch.no_grad():
            return self.agent.act(
                    torch.as_tensor(state, dtype=torch.float).to(self.device), 
                    isTrain=False).cpu()

    def optimize(self, batch_seq):
        state, next_state, action, reward, done = batch_seq
        self.world_model.optimize(state, next_state, action, reward, done)
        self.agent.optimize(self.world_model, state.reshape(-1, self.state_dim))



