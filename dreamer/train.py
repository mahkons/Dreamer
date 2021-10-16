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
BATCH_SIZE = 50
device = torch.device("cuda")

def sample_episode(env, agent):
    state = env.reset()
    episode = Episode(state)
    for steps in itertools.count(1):
        action = agent(state)
        next_state, reward, done = env.step(action)
        episode.add_transition(action, reward, next_state, done)
        state = next_state
        if done:
            break
    return episode


def train(env, agent):
    memory = ReplayBuffer(MEMORY_SIZE)

    steps_count = 0
    while steps_count < TOTAL_STEPS:
        episode = sample_episode(env, agent)
        memory.push(episode)
        steps_count += len(episode)
        print(episode.rewards.sum())

        if memory.num_steps() >= INIT_STEPS:
            for i in range(10):
                batch_seq = memory.sample_seq(SEQ_LEN, BATCH_SIZE, device)
                agent.optimize(batch_seq)

if __name__ == "__main__":
    init_logger("logdir", "tmplol")
    init_random_seeds(RANDOM_SEED, cuda_determenistic=False)

    env = ActionRepeatWrapper(DMControlWrapper(RANDOM_SEED), action_repeat=2)
    agent = Dreamer(env.state_dim, env.action_dim, device)
    train(env, agent)

