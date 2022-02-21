import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from copy import deepcopy

from utils.logger import log

from networks import RewardNetwork, DiscountNetwork, ObservationEncoder, ObservationDecoder, ResnetEncoder
from models import RealNVP, MAF
from params import STOCH_DIM, DETER_DIM, EMBED_DIM, MAX_KL, \
    MODEL_LR, GAMMA, MAX_GRAD_NORM, FROM_PIXELS, PREDICT_DONE, \
    FLOW_GRU_DIM, FLOW_HIDDEN_DIM, FLOW_NUM_BLOCKS, REC_L2_REG, MODEL_WEIGHT_DECAY, \
    FLOW_LOSS_IN_GRU_MULTIPLIER, ED_MODEL_LR, WITH_PRIOR_MODEL, FLOW_LOSS_COEFF, WITH_RESNET_ENCODER


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(WorldModel, self).__init__()

        assert(FROM_PIXELS)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.reward_model = RewardNetwork(FLOW_GRU_DIM).to(device)
        self.discount_model = DiscountNetwork.create(FLOW_GRU_DIM, PREDICT_DONE, GAMMA).to(device)
        self.encoder = ObservationEncoder(self.state_dim, EMBED_DIM, from_pixels=FROM_PIXELS).to(device)
        self.decoder = ObservationDecoder(EMBED_DIM, self.state_dim, from_pixels=FROM_PIXELS).to(device)
        if WITH_RESNET_ENCODER:
            self.encoder = ResnetEncoder().to(device)

        
        # TODO clean up this mess
        self.transition_model = TransitionModel(EMBED_DIM + action_dim, FLOW_GRU_DIM).to(device)
        #self.flow_model = RealNVP(EMBED_DIM, FLOW_GRU_DIM + action_dim, FLOW_HIDDEN_DIM, FLOW_NUM_BLOCKS, 1, device).to(device)
        self.flow_model = MAF(EMBED_DIM, FLOW_GRU_DIM + action_dim, FLOW_HIDDEN_DIM, FLOW_NUM_BLOCKS, device).to(device)
        self.prior_model = PriorModel(FLOW_GRU_DIM + action_dim, EMBED_DIM, device).to(device)

        self.model_params = itertools.chain(
            self.reward_model.parameters(),
            self.discount_model.parameters(),
            self.transition_model.parameters(),
            self.prior_model.parameters(),
            self.flow_model.parameters(),
        )
        self.ed_parameters = itertools.chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
        )

        self.model_optimizer = torch.optim.Adam(self.model_params, lr=MODEL_LR)
        self.ed_optimizer = torch.optim.Adam(self.ed_parameters, lr=ED_MODEL_LR)

        log().add_plot("model_loss", ["reconstruction_loss", "flow_loss", "reward_loss", "discount_loss", "l2_reg_loss"])
        log().add_plot("test_model_loss", ["reconstruction_loss", "flow_loss", "reward_loss", "discount_loss", "l2_reg_loss"])
        self.data_initialized = False

    def test(self, obs, action, reward, discount):
        with torch.no_grad():
            embed = self.encoder(embed)
            l2_reg_loss, rec_loss = self.calc_encoder_loss(obs, embed)
            hidden, flow_list, reward_loss, discount_loss, flow_loss = \
                    self.calc_flow_model_loss(embed, action, reward, discount)

            log().add_plot_point("test_model_loss", [
                rec_loss.item(),
                flow_loss.item(),
                reward_loss.item(),
                discount_loss.item(),
                l2_reg_loss.item()
            ])

    def calc_encoder_loss(self, obs, embed):
        l2_reg_loss = REC_L2_REG * (embed ** 2).sum(dim=2).mean(dim=(0, 1))
        simple_reconstruction = self.decoder(embed)
        rec_loss = ((obs - simple_reconstruction) ** 2).sum(dim=(2, 3, 4)).mean(dim=(0, 1))
        return l2_reg_loss, rec_loss

    def optimize_encoder(self, obs):
        if WITH_RESNET_ENCODER:
            with torch.no_grad():
                return self.encoder(obs), torch.zeros(1), torch.zeros(1)

        embed = self.encoder(obs)

        l2_reg_loss, rec_loss = self.calc_encoder_loss(obs, embed)

        self.ed_optimizer.zero_grad()
        (rec_loss + l2_reg_loss).backward()
        nn.utils.clip_grad_norm_(self.ed_parameters, MAX_GRAD_NORM)
        self.ed_optimizer.step()
        embed = embed.detach()

        return embed, rec_loss, l2_reg_loss

    def calc_flow_model_loss(self, embed, action, reward, discount):
        batch_size = action.shape[1]
        init_hidden, prev_action = self.initial_state(batch_size)
        action = torch.cat([prev_action.unsqueeze(0), action], dim=0)
        hidden, flow_list, jac_list = self.observe(embed, action, init_hidden)
        condition = torch.cat([hidden[:-1], action], -1)

        prior = self.prior_model(condition)
        
        predicted_reward = self.reward_model(hidden[2:])
        reward_loss = F.mse_loss(reward, predicted_reward)

        discount_loss = torch.tensor(0.)
        if PREDICT_DONE:
            predicted_discount_logit = self.discount_model.predict_logit(hidden[2:])
            discount_loss = F.binary_cross_entropy_with_logits(predicted_discount_logit, discount * GAMMA)

        flow_loss = -(prior.log_prob(flow_list).sum(dim=2) + jac_list).mean()
        return hidden, flow_list, reward_loss, discount_loss, flow_loss

    def optimize(self, obs, action, reward, discount):
        embed, rec_loss, l2_reg_loss = self.optimize_encoder(obs)

        hidden, flow_list, reward_loss, discount_loss, flow_loss = \
                self.calc_flow_model_loss(embed, action, reward, discount)

        self.model_optimizer.zero_grad()
        (reward_loss + discount_loss + flow_loss * FLOW_LOSS_COEFF).backward()
        nn.utils.clip_grad_norm_(self.model_params, MAX_GRAD_NORM)
        self.model_optimizer.step()

        log().add_plot_point("model_loss", [
            rec_loss.item(),
            flow_loss.item(),
            reward_loss.item(),
            discount_loss.item(),
            l2_reg_loss.item()
        ])

        return torch.cat([hidden[:-1], flow_list], dim=-1)


    def observe(self, embed_seq, action_seq, init_hidden):
        seq_len, batch_size, embed_size = embed_seq.shape

        hidden = init_hidden
        hidden_list = torch.empty(seq_len + 1, batch_size, FLOW_GRU_DIM, dtype=torch.float, device=self.device)
        flow_list = torch.empty(seq_len, batch_size, EMBED_DIM, dtype=torch.float, device=self.device)
        jac_list = torch.empty(seq_len, batch_size, dtype=torch.float, device=self.device)
        hidden_list[0] = hidden

        for i, (embed, action) in enumerate(zip(embed_seq, action_seq)):
            hidden, embed_flow, logjac = self.obs_step(embed, action, hidden)
            hidden_list[i + 1] = hidden
            flow_list[i], jac_list[i] = embed_flow, logjac

        return hidden_list, flow_list, jac_list

    def obs_step(self, embed, action, hidden):
        condition = torch.cat([hidden, action], dim=-1)
        condition = GradMultiplier.apply(condition, FLOW_LOSS_IN_GRU_MULTIPLIER)

        embed_flow, logjac = self.flow_model.forward_flow(embed, condition)
        hidden = self.transition_model(torch.cat([embed_flow, action], dim=-1), hidden)
        return hidden, embed_flow, logjac
        
        
    def imagine(self, agent, state, horizon):
        batch_size = state.shape[0]
        state_list, reward_list, discount_list, action_list = [state], [], [], []
        hidden, noise = state[:, :FLOW_GRU_DIM], state[:, FLOW_GRU_DIM:]
        for _ in range(horizon + 1):
            action = agent.act(state, isTrain=True)

            hidden = self.transition_model(torch.cat([noise, action], dim=-1), hidden)
            prior = self.prior_model(torch.cat([hidden, action], -1))
            noise = prior.rsample()
            state = torch.cat([hidden, noise], dim=-1)

            state_list.append(state)
            action_list.append(action)

        state = torch.stack(state_list)
        action = torch.stack(action_list)
        reward = self.reward_model(state[2:, :, :FLOW_GRU_DIM])
        discount = torch.sigmoid(self.discount_model.predict_logit(state[2:, :, :FLOW_GRU_DIM]))
        return state[:-1], action[:-1], reward, discount

    def initial_state(self, batch_size):
        hidden = torch.zeros((batch_size, FLOW_GRU_DIM), dtype=torch.float, device=self.device)
        prev_action = torch.zeros((batch_size, self.action_dim), dtype=torch.float, device=self.device)
        return hidden, prev_action


def _soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def _kl_div(p, q):
    pmu, pstd = p
    qmu, qstd = q
    plogs, qlogs = torch.log(pstd), torch.log(qstd)

    d = plogs.shape[2]
    div = (qlogs.sum(dim=2) - plogs.sum(dim=2)) + 0.5 * (-d + torch.exp(2 * (plogs - qlogs)).sum(dim=2) 
            + torch.einsum("lbi,lbi->lb", (pmu - qmu) * torch.exp(-2 * qlogs), pmu - qmu) )
    return div


class GradMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.coeff, None


# simple
class TransitionModel(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super(TransitionModel, self).__init__()
        self.before_rnn = nn.Sequential(
            nn.Linear(input_sz, hidden_sz),
            nn.ELU(),
            nn.Linear(hidden_sz, hidden_sz),
            nn.ELU()
        )
        self.rnn = nn.GRUCell(hidden_sz, hidden_sz)

    def forward(self, input, hidden):
        input = self.before_rnn(input)
        hidden = self.rnn(input, hidden)
        return hidden


class PriorModel(nn.Module):
    def __init__(self, hidden_sz, flow_dim, device):
        super(PriorModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_sz, hidden_sz),
            nn.ELU(),
            nn.Linear(hidden_sz, 2 * flow_dim)
        )
        self.device = device


    def forward(self, hidden):
        mu, log_std = self.model(hidden).chunk(2, dim=-1) 
        if WITH_PRIOR_MODEL:
            return torch.distributions.Normal(mu, torch.exp(log_std))
        return torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(log_std))

