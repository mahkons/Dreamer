import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from utils.logger import log

from networks import RewardNetwork, DiscountNetwork, ObservationEncoder, ObservationDecoder
from models.RSSM import RSSM

MODEL_LR = 6e-4
GAMMA = 0.99
MAX_GRAD_NORM = 100
FROM_PIXELS = False

STOCH_DIM = 32
DETER_DIM = 256
EMBED_DIM = 256
MAX_KL = 3.

class WorldModel():
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.reward_model = RewardNetwork(STOCH_DIM + DETER_DIM).to(device)
        self.discount_model = DiscountNetwork(STOCH_DIM + DETER_DIM).to(device)
        self.transition_model = RSSM(STOCH_DIM, DETER_DIM, EMBED_DIM, self.action_dim).to(device)
        self.encoder = ObservationEncoder(self.state_dim, EMBED_DIM, from_pixels=FROM_PIXELS).to(device)
        self.decoder = ObservationDecoder(STOCH_DIM + DETER_DIM, self.state_dim, from_pixels=FROM_PIXELS).to(device)

        self.parameters = itertools.chain(
            self.reward_model.parameters(),
            self.discount_model.parameters(),
            self.transition_model.parameters(),
            self.encoder.parameters(),
            self.decoder.parameters(),
        )

        self.optimizer = torch.optim.Adam(self.parameters, lr=MODEL_LR)

        log().add_plot("model_loss", ["reconstruction_loss", "kl_divergence_loss", "reward_loss", "discount_loss"])


    def optimize(self, obs, action, reward, done):
        embed = self.encoder(obs)
        hidden, prior, post = self.transition_model.observe(embed, action)

        predicted_obs = self.decoder(hidden)
        predicted_reward = self.reward_model(hidden[1:])
        predicted_discount_logit = self.discount_model.predict_logit(hidden[1:])

        div = _kl_div(post, prior).clip(max=MAX_KL)
        obs_loss = F.mse_loss(obs, predicted_obs) + div
        reward_loss = F.mse_loss(reward, predicted_reward)
        discount_loss = F.binary_cross_entropy_with_logits(predicted_discount_logit, (1 - done) * GAMMA)

        self.optimizer.zero_grad()
        (obs_loss + reward_loss + discount_loss).backward()
        nn.utils.clip_grad_norm_(self.parameters, MAX_GRAD_NORM)
        self.optimizer.step()

        log().add_plot_point("model_loss", [
            obs_loss.item() - div.item(),
            div.item(),
            reward_loss.item(),
            discount_loss.item()
        ])
        return hidden

        
    def imagine(self, agent, state, horizon):
        state_list, reward_list, discount_list, action_list = [state], [], [], []
        for _ in range(horizon):
            action = agent.act(state, isTrain=True)
            state, _ = self.transition_model.imagine_step(action, torch.split(state.detach(), [STOCH_DIM, DETER_DIM], dim=1))
            state = torch.cat(state, dim=-1)

            state_list.append(state)
            action_list.append(action)

        state = torch.stack(state_list)
        action = torch.stack(action_list)
        reward = self.reward_model(state[1:])
        discount = torch.sigmoid(self.discount_model.predict_logit(state[1:]))
        return state, action, reward, discount



def _kl_div(p, q):
    pmu, plogs = p
    qmu, qlogs = q
    d = plogs.shape[2]
    div = (qlogs.sum(dim=2) - plogs.sum(dim=2)) + 0.5 * (-d + torch.exp(2 * (plogs - qlogs)).sum(dim=2) 
            + torch.einsum("lbi,lbi->lb", (pmu - qmu) * torch.exp(-2 * qlogs), pmu - qmu) )
    return torch.mean(div)

