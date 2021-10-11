import numpy as np

class ActionRepeatWrapper():
    def __init__(self, env, action_repeat):
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.action_repeat = action_repeat

    def reset(self):
        return self.env.reset()

    def step(self, action):
        reward_sum = 0
        for _ in range(self.action_repeat):
            obs, reward, done = self.env.step(action)
            reward_sum += reward
            if done:
                break
        return obs, reward_sum, done
