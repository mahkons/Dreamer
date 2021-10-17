import numpy as np
from dm_control import suite

class DMControlWrapper():
    def __init__(self, domain_name, task_name, from_pixels, random_seed):
        self.env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": random_seed},
        )
        action_spec = self.env.action_spec()
        assert(all(action_spec.minimum == -1))
        assert(all(action_spec.maximum == 1))
        assert(len(action_spec.shape) == 1)

        self.action_dim = action_spec.shape[0]

        self.state_dim = sum(map(lambda v: v.shape[0], self.env.observation_spec().values()))
        self.from_pixels = from_pixels
        assert not self.from_pixels

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

