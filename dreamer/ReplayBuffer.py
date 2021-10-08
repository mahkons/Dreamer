import torch
import random
from collections import namedtuple, deque


class Episode():
    def __init__(self, init_state):
        self.states = [torch.as_tensor(init_state)]
        self.rewards = []
        self.actions = []

        self.done = False

    def add_transition(self, action, reward, next_state, done):
        assert(not self.done)
        self.states.append(torch.as_tensor(next_state, dtype=torch.float))
        self.actions.append(torch.as_tensor(action, dtype=torch.float))
        self.rewards.append(torch.as_tensor(reward, dtype=torch.float))
        if done:
            self.done = True
            self.states = torch.stack(self.states)
            self.actions = torch.stack(self.actions)
            self.rewards = torch.stack(self.rewards)

    def sample(self, seq_len):
        assert(self.done)
        pos = np.random.choice(len(self))
        done = torch.zeros(seq_len, dtype=torch.float)
        if pos >= len(self) - seq_len + 1:
            pos = len(self) - seq_len + 1
            done[-1] = 1

        return self.states[pos : pos + seq_len], \
                self.states[pos + 1 : pos + seq_len + 1], \
                self.actions[pos : pos + seq_len], \
                self.rewards[pos : pos + seq_len], \
                done


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
        state, next_state, action, reward, done = zip(*trajectories)
        return torch.as_tensor(state, device=device), \
            torch.as_tensor(next_state, device=device), \
            torch.as_tensor(action, device=device), \
            torch.as_tensor(reward, device=device), \
            torch.as_tensor(done, device=device)


    def clear(self):
        self.memory.clear()
        self.step_to_episode().clear()
