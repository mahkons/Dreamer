import torch
import numpy as np
import random
from collections import namedtuple, deque


class Episode():
    def __init__(self, init_state):
        init_obs, init_cond = init_state
        self.states = [torch.as_tensor(init_obs, dtype=torch.float)]
        self.conds = [torch.as_tensor(init_cond, dtype=torch.float)]
        self.rewards = []
        self.actions = []
        self.discounts = []

        self.done = False

    def add_transition(self, action, reward, next_state, discount, done):
        assert(not self.done)
        self.states.append(torch.as_tensor(next_state[0], dtype=torch.float))
        self.conds.append(torch.as_tensor(next_state[1], dtype=torch.float))
        self.actions.append(torch.as_tensor(action, dtype=torch.float))
        self.rewards.append(torch.as_tensor(reward, dtype=torch.float))
        self.discounts.append(torch.as_tensor(discount, dtype=torch.float))
        if done:
            self.done = True
            self.discounts = torch.stack(self.discounts)
            self.states = torch.stack(self.states)
            self.conds = torch.stack(self.conds)
            self.actions = torch.stack(self.actions)
            self.rewards = torch.stack(self.rewards)

    def sample(self, seq_len):
        assert(self.done)
        pos = np.random.choice(len(self))
        if pos >= len(self) - seq_len:
            pos = len(self) - seq_len

        return self.states[pos : pos + seq_len + 1], \
                self.conds[pos : pos + seq_len + 1], \
                self.actions[pos : pos + seq_len], \
                self.rewards[pos : pos + seq_len], \
                self.discounts[pos : pos + seq_len]


    def __len__(self):
        return len(self.actions)



class ReplayBuffer():
    def __init__(self, capacity_steps):
        self.capacity_steps = capacity_steps
        self.memory = deque()
        self.step_to_episode = deque()

    def push(self, episode):
        self.memory.append(episode)
        for i in range(len(episode)):
            self.step_to_episode.append(episode)
        while len(self.step_to_episode) > self.capacity_steps:
            self.pop()

    def pop(self):
        for i in range(len(self.memory[0])):
            self.step_to_episode.popleft()
        self.memory.popleft()

    def sample_seq(self, seq_len, batch_size, device=torch.device("cpu")):
        """
            Samples episodes weighed by their length
            Samples starting position in episode uniformly
                and shifts it to left so that seq_len fits into episode
        """
        episodes = [self.step_to_episode[step] for step in np.random.choice(len(self.step_to_episode), size=batch_size)]
        trajectories = [episode.sample(seq_len) for episode in episodes]        
        state, cond, action, reward, done = zip(*trajectories)
        return torch.stack(state).transpose(0, 1).contiguous().to(device), \
            torch.stack(cond).transpose(0, 1).to(device), \
            torch.stack(action).transpose(0, 1).to(device), \
            torch.stack(reward).transpose(0, 1).to(device), \
            torch.stack(done).transpose(0, 1).to(device)

    def num_episodes(self):
        return len(self.memory)

    def num_steps(self):
        return len(self.step_to_episode)

    def clear(self):
        self.memory.clear()
        self.step_to_episode().clear()
