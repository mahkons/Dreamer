import numpy as np
from dm_control import suite


class DMControlWrapper():
    def __init__(self, domain_name, task_name, from_pixels, random_seed):
        self.env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": random_seed},
        )
        self.action_spec = self.env.action_spec()
        assert(len(self.action_spec.shape) == 1)
        self.action_dim = self.action_spec.shape[0]

        self.from_pixels = from_pixels
        if self.from_pixels:
            self._size = (64, 64)
            self.state_dim = (3, 64, 64)
            self._camera = dict(quadruped=2).get(domain_name, 0) # TODO from dreamer why?
        else:
            self.state_dim = sum(map(lambda v: v.shape[0], self.env.observation_spec().values()))

    def reset(self):
        time_step = self.env.reset()
        return self._get_obs(time_step)

    def step(self, action):
        action = ((action + 1.) / 2. * (self.action_spec.maximum - self.action_spec.minimum)) \
                + self.action_spec.minimum
        time_step = self.env.step(action)
        reward = time_step.reward or 0
        done = time_step.last()
        return self._get_obs(time_step), reward, done

    def _get_obs(self, time_step):
        if self.from_pixels:
            obs = self.env.physics.render(*self._size, camera_id=self._camera)
            return (np.ascontiguousarray(obs.transpose(2, 0, 1)) / 255.) - 0.5
        else:
            return np.concatenate(list(time_step.observation.values()))



# as in dreamer paper
def dm_suite_benchmark():
    return [
        ("acrobot", "swingup"),
        #  ("acrobot", "swingup_sparse"),
        ("ball_in_cup", "catch"),
        ("cartpole", "balance"),
        ("cartpole", "balance_sparse"),
        ("cartpole", "swingup"),
        ("cartpole", "swingup_sparse"),
        ("cheetah", "run"),
        ("finger", "spin"),
        ("finger", "turn_easy"),
        ("finger", "turn_hard"),
        #  ("fish", "upright"),
        #  ("fish", "swim"),
        ("hopper", "stand"),
        ("hopper", "hop"),
        #  ("humanoid", "stand"),
        #  ("humanoid", "walk"),
        #  ("humanoid", "run"),
        #  ("manipulator", "bring_ball"),
        ("pendulum", "swingup"),
        #  ("point_mass", "easy"),
        ("reacher", "easy"),
        ("reacher", "hard"),
        #  ("swimmer", "swimmer6"),
        #  ("swimmer", "swimmer15"),
        ("quadruped", "walk"),
        ("quadruped", "run"),
        ("walker", "stand"),
        ("walker", "walk"),
        ("walker", "run"),
    ]
