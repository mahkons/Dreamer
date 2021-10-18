import sys
sys.path.append("..")

import itertools
import math
import torch
import numpy as np

from envs import DMControlWrapper, ActionRepeatWrapper
from utils.logger import init_logger, log
from utils.random import init_random_seeds

from Dreamer import Dreamer
from ReplayBuffer import Episode, ReplayBuffer

# TODO
# put all hyperparameters in some sort of config
RANDOM_SEED = 239
INIT_STEPS = 10**4
MEMORY_SIZE = 10**6
TOTAL_STEPS = 10**6
SEQ_LEN = 50
BATCH_SIZE = 30 # TODO 50 does not fit on my small local gpu =(
FROM_PIXELS = False
device = torch.device("cpu")

def sample_episode(env, agent):
    obs = env.reset()
    episode = Episode(obs)
    hidden, action = agent.world_model.transition_model.initial_state(batch_size=1) 
    action.squeeze_(0)
    for steps in itertools.count(1):
        action, hidden = agent(obs, hidden, action)
        obs, reward, done = env.step(action)
        episode.add_transition(action, reward, obs, done)
        if done:
            break
    return episode


def train(env, agent):
    log().add_plot("eval_reward", ["episode", "steps", "reward"])

    memory = ReplayBuffer(MEMORY_SIZE)

    step_count, episode_count = 0, 0
    while step_count < TOTAL_STEPS:
        episode = sample_episode(env, agent)
        memory.push(episode)
        step_count += len(episode)
        episode_count += 1
        log().add_plot_point("eval_reward", [episode_count, step_count, episode.rewards.sum()])
        print(episode.rewards.sum())

        if memory.num_steps() >= INIT_STEPS:
            for i in range(10):
                batch_seq = memory.sample_seq(SEQ_LEN, BATCH_SIZE, device)
                agent.optimize(batch_seq)

    log().save_logs() # TODO logger context


if __name__ == "__main__":
    init_logger("logdir", "tmplol")
    init_random_seeds(RANDOM_SEED, cuda_determenistic=False)

    env = ActionRepeatWrapper(
        DMControlWrapper("cartpole", "balance", from_pixels=FROM_PIXELS, random_seed=RANDOM_SEED),
        action_repeat=2
    )
    agent = Dreamer(env.state_dim, env.action_dim, device)
    train(env, agent)

