import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from networks import RewardNetwork, DiscountNetwork

HIDDEN_DIM = 64

MODEL_LR = 6e-4
GAMMA = 0.99

class WorldModel():
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.reward_model = RewardNetwork(HIDDEN_DIM).to(device)
        self.discount_model = DiscountNetwork(HIDDEN_DIM).to(device)

        self.transition_model = nn.GRU(HIDDEN_DIM + self.action_dim, HIDDEN_DIM).to(device)

        self.encoder = nn.Sequential(nn.Linear(self.state_dim, 300), nn.GELU(), nn.Linear(300, HIDDEN_DIM)).to(device)
        self.decoder = nn.Sequential(nn.Linear(HIDDEN_DIM, 300), nn.GELU(), nn.Linear(300, self.state_dim)).to(device)

        self.optimizer = torch.optim.Adam(itertools.chain(
            self.reward_model.parameters(),
            self.discount_model.parameters(),
            self.transition_model.parameters(),
            self.encoder.parameters(),
            self.decoder.parameters(),
        ), lr=MODEL_LR)


    def optimize(self, state, action, reward, done):
        encoded_state = self.encoder(state)

        state_action = torch.cat([encoded_state[:-1], action], dim=2)
        predicted_state = self.decoder(self.transition_model(state_action)[0])
        predicted_reward = self.reward_model(encoded_state[1:])
        predicted_discount_log = self.discount_model.predict_log(encoded_state[1:])

        state_loss = F.mse_loss(state[1:], predicted_state)
        reward_loss = F.mse_loss(reward, predicted_reward)
        discount_loss = F.binary_cross_entropy_with_logits(predicted_discount_log, (1 - done) * GAMMA)

        self.optimizer.zero_grad()
        (state_loss + reward_loss + discount_loss).backward()
        self.optimizer.step()

        print(state_loss.item(), reward_loss.item(), discount_loss.item())

        
    def imagine(self, agent, state, horizon):
        state_list, reward_list, discount_list, action_list = [state], [], [], []
        hidden = torch.zeros((1, state.shape[0], HIDDEN_DIM), dtype=torch.float, device=self.device)
        for _ in range(horizon):
            action = agent.act(state, isTrain=True)
            _, next_hidden = self.transition_model(torch.cat([state, action], dim=1).unsqueeze(0), hidden)
            next_state = next_hidden.squeeze(0)
            state, hidden = next_state, next_hidden

            state_list.append(state)
            action_list.append(action)

        state = torch.stack(state_list)
        action = torch.stack(action_list)
        reward = self.reward_model(state[1:])
        discount = self.discount_model(state[1:])
        return state, action, reward, discount

