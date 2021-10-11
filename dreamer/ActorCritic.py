import torch
import torch.nn.functional as F
import numpy as np
import itertools

from networks import ActorNetwork, CriticNetwork

AGENT_LR = 5e-8
GAMMA = 0.99
LAMBDA = 0.95
HORIZON = 15


class ActorCritic():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        self.optimizer = torch.optim.Adam(itertools.chain(
            self.actor.parameters(),
            self.critic.parameters(),
        ), lr=AGENT_LR)


    def act(self, state):
        return self.actor(state)

    def optimize(self, env, init_state):
        """
            :param env: differentiable environment
        """

        state, action, reward, discount = env.imagine(self, init_state, HORIZON)
        values = self.critic(state)
        values_lr = self._compute_value_estimates(values, reward, discount)

        actor_loss = values_lr.mean()
        critic_loss = F.mse_loss(values[:, :-1], values_lr[:, :-1])

        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        self.optimizer.step()
    

    def _compute_value_estimates(self, values, reward, discount):
        assert(len(values.shape) == 2)

        values_lr = values.clone()
        for i in reversed(range(reward.shape[1])):
            values_lr[:, i] = reward[:, i] + discount[:, i] * \
                    ((1 - LAMBDA) * values[:, i + 1] + LAMBDA * values_lr[:, i + 1])
        return values_lr


        


