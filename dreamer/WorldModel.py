import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from networks import RewardNetwork, DiscountNetwork

HIDDEN_DIM = 128

MODEL_LR = 6e-4
GAMMA = 0.99

class WorldModel():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.reward_model = RewardNetwork(self.state_dim)
        self.discount_model = DiscountNetwork(self.state_dim)

        self.transition_model = nn.GRU(self.state_dim + self.action_dim, HIDDEN_DIM, batch_first=True)
        self.transition_fc = nn.Sequential(nn.GELU(), nn.Linear(HIDDEN_DIM, self.state_dim))

        self.optimizer = torch.optim.Adam(itertools.chain(
            self.reward_model.parameters(),
            self.discount_model.parameters(),
            self.transition_model.parameters(),
        ), lr=MODEL_LR)


    def optimize(self, state, next_state, action, reward, done):
        state_action = torch.cat([state, action], dim=2)
        predicted_state = self.transition_fc(self.transition_model(state_action)[0])

        predicted_reward = self.reward_model(state)
        predicted_done_log = self.discount_model(state)

        state_loss = F.mse_loss(state, predicted_state)
        reward_loss = F.mse_loss(reward, predicted_reward)
        done_loss = F.binary_cross_entropy_with_logits(predicted_done_log, GAMMA * done)

        self.optimizer.zero_grad()
        (state_loss + reward_loss + done_loss).backward()
        self.optimizer.step()

        
    def imagine(self, agent, state, horizon):
        state_list, reward_list, discount_list, action_list = [state], [], [], []
        hidden = torch.zeros((1, state.shape[0], HIDDEN_DIM), dtype=torch.float)
        for _ in range(horizon):
            action = agent.act(state)
            reward = torch.exp(self.reward_model(state))
            discount = self.discount_model(state)
            _, next_hidden = self.transition_model(torch.cat([state, action], dim=1).unsqueeze(1), hidden)
            next_state = self.transition_fc(next_hidden.squeeze(0))
            state, hidden = next_state, next_hidden

            state_list.append(state)
            reward_list.append(reward)
            action_list.append(action)
            discount_list.append(discount)

        return torch.stack(state_list).transpose(0, 1), \
                torch.stack(action_list).transpose(0, 1), \
                torch.stack(reward_list).transpose(0, 1), \
                torch.stack(discount_list).transpose(0, 1)

