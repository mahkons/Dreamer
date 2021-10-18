import numpy as np
import torch
import math

from WorldModel import WorldModel
from ActorCritic import ActorCritic

STOCH_DIM = 32
DETER_DIM = 256
EMBED_DIM = 256

class Dreamer():
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.world_model = WorldModel(state_dim, action_dim, device)
        self.agent = ActorCritic(STOCH_DIM + DETER_DIM, action_dim, device)

    def __call__(self, obs, hidden, prev_action):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0)
            prev_action = torch.as_tensor(prev_action, dtype=torch.float, device=self.device).unsqueeze(0)
            embed = self.world_model.encoder(obs)
            next_hidden, _, _ = self.world_model.transition_model.obs_step(prev_action, hidden, embed)
            action, mu = self.agent.act(torch.cat(next_hidden, dim=-1), isTrain=False)
            
            mu = mu[0].cpu()
            action = torch.tanh(mu) + torch.randn_like(mu) * math.sqrt(0.3) # superb exploration
            return action.clip_(-1, 1), next_hidden

    def optimize(self, batch_seq):
        obs, action, reward, done = batch_seq

        hidden = self.world_model.optimize(obs, action, reward, done)
        hidden = hidden.detach_()[:-1].view(-1, STOCH_DIM + DETER_DIM)
        self.agent.optimize(self.world_model, hidden)



