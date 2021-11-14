import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from utils.logger import log
from networks import ActorNetwork, CriticNetwork
from params import ACTOR_LR, CRITIC_LR, GAMMA, LAMBDA, HORIZON, MAX_GRAD_NORM

class ActorCritic():
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        log().add_plot("agent_loss", ["actor_loss", "critic_loss"])

    def act(self, state, isTrain):
        return self.actor.act(state, isTrain)

    def optimize(self, env, init_state):
        """
            :param env: differentiable environment
        """

        state, action, reward, discount = env.imagine(self, init_state, HORIZON)
        seq_discount = torch.cat([torch.ones_like(discount[0:1]), torch.cumprod(discount, dim=0)], dim=0)

        values = self.critic(state)
        values_lr = self._compute_value_estimates(values, reward, discount)

        actor_loss = -(values_lr * seq_discount).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
        self.actor_optimizer.step()

        critic_loss = (F.mse_loss(values, values_lr.detach(), reduction="none") * seq_discount).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward(inputs=list(self.critic.parameters()))
        nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
        self.critic_optimizer.step()

        log().add_plot_point("agent_loss", [actor_loss.item(), critic_loss.item()])
    

    def _compute_value_estimates(self, values, reward, discount):
        assert(len(values.shape) == 2)

        values_lr = torch.empty_like(values)
        values_lr[-1] = values[-1]
        for i in reversed(range(reward.shape[0])):
            values_lr[i] = reward[i] + discount[i] * \
                    ((1 - LAMBDA) * values[i + 1] + LAMBDA * values_lr[i + 1])
        return values_lr


        


