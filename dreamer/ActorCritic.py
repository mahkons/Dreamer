import torch
import numpy as np
import itertools

from networks import ActorNetwork, CriticNetwork

ACTOR_LR = 8e-5
CRITIC_LR = 8e-5
GAMMA = 0.99
LAMBDA = 0.95


class ActorCritic():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)

    def act(self, state):
        return np.random.uniform(-1, 1, size=self.action_dim)

    def optimize(self, differentiable_env, init_state):
        pass

