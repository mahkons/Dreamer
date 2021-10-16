import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from networks import RewardNetwork, DiscountNetwork, ObservationEncoder, ObservationDecoder

HIDDEN_DIM = 64

MODEL_LR = 6e-4
GAMMA = 0.99
MAX_GRAD_NORM = 100

class WorldModel():
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.reward_model = RewardNetwork(HIDDEN_DIM).to(device)
        self.discount_model = DiscountNetwork(HIDDEN_DIM).to(device)

        self.transition_model = nn.GRU(self.action_dim, HIDDEN_DIM).to(device)

        self.encoder = ObservationEncoder(self.state_dim, HIDDEN_DIM, from_pixels=False).to(device)
        self.decoder = ObservationDecoder(HIDDEN_DIM, self.state_dim, from_pixels=False).to(device)

        self.parameters = itertools.chain(
            self.reward_model.parameters(),
            self.discount_model.parameters(),
            self.transition_model.parameters(),
            self.encoder.parameters(),
            self.decoder.parameters(),
        )

        self.optimizer = torch.optim.Adam(self.parameters, lr=MODEL_LR)


    def optimize(self, state, action, reward, done):
        encoded_state = self.encoder(state)

        hidden = encoded_state[0].unsqueeze(0)
        predicted_state = self.decoder(self.transition_model(action, hidden)[0])
        predicted_reward = self.reward_model(encoded_state[1:])
        predicted_discount_logit = self.discount_model.predict_logit(encoded_state[1:])

        state_loss = F.mse_loss(state[1:], predicted_state)
        reward_loss = F.mse_loss(reward, predicted_reward)
        discount_loss = F.binary_cross_entropy_with_logits(predicted_discount_logit, (1 - done) * GAMMA)

        self.optimizer.zero_grad()
        (state_loss + reward_loss + discount_loss).backward()
        nn.utils.clip_grad_norm_(self.parameters, MAX_GRAD_NORM)
        self.optimizer.step()

        print(state_loss.item(), reward_loss.item(), discount_loss.item())

        
    def imagine(self, agent, state, horizon):
        state_list, reward_list, discount_list, action_list = [state], [], [], []
        for _ in range(horizon):
            action = agent.act(state, isTrain=True)
            _, next_state = self.transition_model(action.unsqueeze(0), state.unsqueeze(0))
            state = next_state.squeeze(0)
            state_list.append(state)
            action_list.append(action)

        state = torch.stack(state_list)
        action = torch.stack(action_list)
        reward = self.reward_model(state[1:])
        discount = torch.sigmoid(self.discount_model.predict_logit(state[1:]))
        return state, action, reward, discount

