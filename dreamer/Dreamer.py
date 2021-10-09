import numpy as np
import torch


class Dreamer():
    def __init__(self, action_size):
        self.action_size = action_size

    def __call__(self, state):
        return self.act(state)

    def act(self, state):
        return np.random.uniform(-1, 1, size=self.action_size)

    def train(self, batch_seq):
        pass
        
