import sys
sys.path.append("..")

import itertools
import torch
import numpy as np

from envs.DMControlWrapper import DMControlWrapper
from utils.logger import init_logger, log
from utils.random import init_random_seeds

from Dreamer import Dreamer
from ReplayBuffer import Episode, ReplayBuffer


RANDOM_SEED = 239
INIT_STEPS = 2500
MEMORY_SIZE = 10**4
TOTAL_STEPS = 10**4
SEQ_LEN = 50
BATCH_SIZE = 50
device = torch.device("cpu")

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
            batch_seq = memory.sample_seq(SEQ_LEN, BATCH_SIZE, device)
            agent.train(batch_seq)

if __name__ == "__main__":
    init_logger("logdir", "tmplol")
    init_random_seeds(RANDOM_SEED, cuda_determenistic=False)

    env = DMControlWrapper(RANDOM_SEED)
    agent = Dreamer(env.action_size)
    train(env, agent)
