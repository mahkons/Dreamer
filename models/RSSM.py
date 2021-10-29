import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

from params import RSSM_HIDDEN_DIM, MIN_STD

class RSSM(nn.Module):
    def __init__(self, stoch_dim, deter_dim, embed_dim, action_dim):
        super(RSSM, self).__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.rnn_cell = nn.GRUCell(RSSM_HIDDEN_DIM, deter_dim)

        self.input_encode = nn.Sequential(
            nn.Linear(self.stoch_dim + self.action_dim, RSSM_HIDDEN_DIM),
            nn.GELU()
        )

        self.imagine_encode = nn.Sequential(
            nn.Linear(self.deter_dim, RSSM_HIDDEN_DIM),
            nn.GELU(),
        )
        self.imagine_mu = nn.Linear(RSSM_HIDDEN_DIM, self.stoch_dim)
        self.imagine_std = nn.Linear(RSSM_HIDDEN_DIM, self.stoch_dim)


        self.obs_encode = nn.Sequential(
            nn.Linear(self.deter_dim + self.embed_dim, RSSM_HIDDEN_DIM),
            nn.GELU(),
        )
        self.obs_mu = nn.Linear(RSSM_HIDDEN_DIM, self.stoch_dim)
        self.obs_std = nn.Linear(RSSM_HIDDEN_DIM, self.stoch_dim)

        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return super(RSSM, self).to(device)

    def forward(self, *args):
        raise NotImplementedError

    def initial_state(self, batch_size):
        stoch = torch.zeros((batch_size, self.stoch_dim), dtype=torch.float, device=self.device)
        deter = torch.zeros((batch_size, self.deter_dim), dtype=torch.float, device=self.device)
        prev_action = torch.zeros((batch_size, self.action_dim), dtype=torch.float, device=self.device)
        return (stoch, deter), prev_action

    def observe(self, embed_seq, action_seq):
        seq_len, batch_size, embed_size = embed_seq.shape
        hidden, prev_action = self.initial_state(batch_size)

        hidden_list = torch.empty(seq_len, batch_size, self.stoch_dim + self.deter_dim, dtype=torch.float, device=self.device)
        prior_mu = torch.empty(seq_len, batch_size, self.stoch_dim, dtype=torch.float, device=self.device)
        prior_std = torch.empty_like(prior_mu)
        post_mu, post_std = torch.empty_like(prior_mu), torch.empty_like(prior_std)

        for i, (embed, action) in enumerate(zip(embed_seq, itertools.chain([prev_action], action_seq))):
            hidden, (prior_mu[i], prior_std[i]), (post_mu[i], post_std[i]) = self.obs_step(action, hidden, embed)
            hidden_list[i] = torch.cat(hidden, dim=-1)

        return hidden_list, (prior_mu, prior_std), (post_mu, post_std)

    def obs_step(self, prev_action, hidden, embed):
        deter = self._deterministic_step(prev_action, hidden)
        prior_mu, prior_std = self._get_prior(deter)
        mu, std = self._get_post(deter, embed)
        stoch = self._reparametrization_trick(mu, std)
        return (stoch, deter), (prior_mu, prior_std), (mu, std)
        
    def imagine_step(self, prev_action, hidden):
        deter = self._deterministic_step(prev_action, hidden)
        mu, std = self._get_prior(deter)
        stoch = self._reparametrization_trick(mu, std)
        return (stoch, deter), (mu, std)


    def _get_post(self, deter, embed):
        x = self.obs_encode(torch.cat([deter, embed], dim=-1))
        mu, std = self.obs_mu(x), self.obs_std(x)
        std = F.softplus(std) + MIN_STD
        return mu, std

    def _get_prior(self, deter):
        x = self.imagine_encode(deter)
        mu, std = self.imagine_mu(x), self.imagine_std(x)
        std = F.softplus(std) + MIN_STD
        return mu, std

    def _reparametrization_trick(self, mu, std):
        eps = torch.randn_like(mu)
        return mu + std * eps

    def _deterministic_step(self, prev_action, hidden):
        stoch, deter = hidden
        x = self.input_encode(torch.cat([stoch, prev_action], dim=-1))
        deter = self.rnn_cell(x, deter)
        return deter
