import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from utils.logger import log

from networks import RewardNetwork, DiscountNetwork, ObservationEncoder, ObservationDecoder
from models.RSSM import RSSM
from models.MAF import MAF
from params import STOCH_DIM, DETER_DIM, EMBED_DIM, MAX_KL, \
    MODEL_LR, GAMMA, MAX_GRAD_NORM, FROM_PIXELS, PREDICT_DONE, \
    FLOW_GRU_DIM, FLOW_HIDDEN_DIM, FLOW_NUM_BLOCKS, FLOW_LOSS_COEFF, REC_L2_REG

class WorldModel():
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.reward_model = RewardNetwork(FLOW_GRU_DIM).to(device)
        self.discount_model = DiscountNetwork.create(FLOW_GRU_DIM, PREDICT_DONE, GAMMA).to(device)
        self.encoder = ObservationEncoder(self.state_dim, EMBED_DIM, from_pixels=FROM_PIXELS).to(device)
        self.decoder = ObservationDecoder(EMBED_DIM, self.state_dim, from_pixels=FROM_PIXELS).to(device)


        self.transition_model = nn.GRUCell(EMBED_DIM + action_dim, FLOW_GRU_DIM).to(device)
        self.flow_model = MAF(EMBED_DIM, FLOW_GRU_DIM + action_dim, FLOW_HIDDEN_DIM, FLOW_NUM_BLOCKS, device).to(device)

        self.parameters = itertools.chain(
            self.reward_model.parameters(),
            self.discount_model.parameters(),
            self.transition_model.parameters(),
            self.flow_model.parameters(),
            self.encoder.parameters(),
            self.decoder.parameters(),
        )

        self.optimizer = torch.optim.Adam(self.parameters, lr=MODEL_LR)

        log().add_plot("model_loss", ["reconstruction_loss", "flow_loss", "reward_loss", "discount_loss", "l2_reg_loss"])


    def optimize(self, obs, action, reward, done):
        batch_size = action.shape[1]
        embed = self.encoder(obs)
        reconstruction = self.decoder(embed)
        rec_loss = F.mse_loss(obs, reconstruction)
        l2_reg_loss = REC_L2_REG * (embed ** 2).sum(dim=(1, 2)).mean(dim=0)
        embed = embed.detach()

        init_hidden, prev_action = self.initial_state(batch_size)
        action = torch.cat([prev_action.unsqueeze(0), action], dim=0)
        hidden = self.observe(embed, action, init_hidden)
        flow_loss = self.flow_model.calc_loss(embed, torch.cat([hidden[:-1], action], dim=-1)) 
        
        predicted_reward = self.reward_model(hidden[2:])
        reward_loss = F.mse_loss(reward, predicted_reward)

        discount_loss = torch.tensor(0.)
        if PREDICT_DONE:
            predicted_discount_logit = self.discount_model.predict_logit(hidden[1:])
            discount_loss = F.binary_cross_entropy_with_logits(predicted_discount_logit, (1 - done) * GAMMA)

        self.optimizer.zero_grad()
        (rec_loss + flow_loss * FLOW_LOSS_COEFF + reward_loss + discount_loss + l2_reg_loss).backward()
        nn.utils.clip_grad_norm_(self.parameters, MAX_GRAD_NORM)
        self.optimizer.step()

        log().add_plot_point("model_loss", [
            rec_loss.item(),
            flow_loss.item(),
            reward_loss.item(),
            discount_loss.item(),
            l2_reg_loss.item()
        ])
        return hidden


    def observe(self, embed_seq, action_seq, init_hidden):
        seq_len, batch_size, embed_size = embed_seq.shape

        hidden = init_hidden
        hidden_list = torch.empty(seq_len + 1, batch_size, FLOW_GRU_DIM, dtype=torch.float, device=self.device)
        hidden_list[0] = hidden

        for i, (embed, action) in enumerate(zip(embed_seq, action_seq)):
            hidden = self.obs_step(embed, action, hidden)
            hidden_list[i + 1] = hidden

        return hidden_list

    def obs_step(self, embed, action, hidden):
        embed_flow, _ = self.flow_model.forward_flow(embed, torch.cat([hidden, action], dim=-1))
        hidden = self.transition_model(torch.cat([embed_flow, action], dim=-1), hidden)
        return hidden
        
        
    def imagine(self, agent, state, horizon):
        batch_size = state.shape[0]
        state_list, reward_list, discount_list, action_list = [state], [], [], []
        for _ in range(horizon):
            action = agent.act(state, isTrain=True)
            noise = self.flow_model.prior.sample([batch_size, EMBED_DIM])
            state = self.transition_model(torch.cat([noise, action], dim=-1), state)

            state_list.append(state)
            action_list.append(action)

        state = torch.stack(state_list)
        action = torch.stack(action_list)
        reward = self.reward_model(state[1:])
        discount = torch.sigmoid(self.discount_model.predict_logit(state[1:]))
        return state, action, reward, discount

    def initial_state(self, batch_size):
        hidden = torch.zeros((batch_size, FLOW_GRU_DIM), dtype=torch.float, device=self.device)
        prev_action = torch.zeros((batch_size, self.action_dim), dtype=torch.float, device=self.device)
        return hidden, prev_action



def _kl_div(p, q):
    pmu, pstd = p
    qmu, qstd = q
    plogs, qlogs = torch.log(pstd), torch.log(qstd)

    d = plogs.shape[2]
    div = (qlogs.sum(dim=2) - plogs.sum(dim=2)) + 0.5 * (-d + torch.exp(2 * (plogs - qlogs)).sum(dim=2) 
            + torch.einsum("lbi,lbi->lb", (pmu - qmu) * torch.exp(-2 * qlogs), pmu - qmu) )
    return div

