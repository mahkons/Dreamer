import numpy as np
from dm_control import suite

class DMControlWrapper():
    def __init__(self):
        self.env = suite.load(domain_name="cartpole", task_name="balance")
        print(self.env.action_spec()) # TODO all kind of assertions
        self.action_size = self.env.action_spec().shape[0]
        #TODO learn from pixels
        self.from_pixels = False

    def reset(self):
        time_step = self.env.reset()
        return self._get_obs(time_step)

    def step(self, action):
        time_step = self.env.step(action)
        reward = time_step.reward or 0
        done = time_step.last()
        return self._get_obs(time_step), reward, done

    def _get_obs(self, time_step):
        if self.from_pixels:
            assert(False)
        else:
            return np.concatenate(list(time_step.observation.values()))

