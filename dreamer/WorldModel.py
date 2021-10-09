import torch
import torch.nn as nn


class WorldModel():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def optimize(self, state, next_state, action, reward, done):
        pass

    def imagine(self, state, horizon):
        pass
