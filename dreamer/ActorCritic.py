import torch
import torch.nn.functional as F
import numpy as np
import itertools

from networks import ActorNetwork, CriticNetwork

ACTOR_LR = 8e-5
CRITIC_LR = 8e-5
GAMMA = 0.99
LAMBDA = 0.95
HORIZON = 15


class ActorCritic():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)


    def act(self, state):
        return self.actor(state)

    def optimize(self, env, init_state):
        """
            :param env: differentiable environment
        """

        # TODO avoid duplicate calculations
        state, action, reward, discount = env.imagine(self, init_state, HORIZON)
        values = self.critic(state)
        values_lr = self._compute_value_estimates(values, reward, discount)

        values2 = self.critic(state.detach())
        values_lr2 = self._compute_value_estimates(values2, reward, discount)

        actor_loss = -values_lr.mean()
        critic_loss = F.mse_loss(values2[:, :-1], values_lr2[:, :-1])
        print(actor_loss.item(), critic_loss.item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    

    def _compute_value_estimates(self, values, reward, discount):
        assert(len(values.shape) == 2)

        values_lr = values.clone()
        for i in reversed(range(reward.shape[1])):
            values_lr[:, i] = reward[:, i] + discount[:, i] * \
                    ((1 - LAMBDA) * values[:, i + 1] + LAMBDA * values_lr[:, i + 1])
        return values_lr


        


