import sys
sys.path.append("..")

import itertools
import math
import torch
import numpy as np
import multiprocessing
import os

from envs import DMControlWrapper, ActionRepeatWrapper, dm_suite_benchmark
from utils.logger import init_logger, log
from utils.random import init_random_seeds

from Dreamer import Dreamer
from ReplayBuffer import Episode, ReplayBuffer

from params import RANDOM_SEED, INIT_STEPS, MEMORY_SIZE, TOTAL_STEPS, \
        SEQ_LEN, BATCH_SIZE, FROM_PIXELS, TRAIN_ITERS_PER_EPISODE, \
        ACTION_REPEAT

device = torch.device("cuda")

def sample_episode(env, agent):
    obs = env.reset()
    episode = Episode(obs)
    hidden, action = agent.world_model.initial_state(batch_size=1) 
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
        step_count += len(episode) * ACTION_REPEAT
        episode_count += 1
        log().add_plot_point("eval_reward", [episode_count, step_count, episode.rewards.sum().item()])

        if memory.num_steps() >= INIT_STEPS:
            for i in range(TRAIN_ITERS_PER_EPISODE):
                batch_seq = memory.sample_seq(SEQ_LEN, BATCH_SIZE, device)
                agent.optimize(batch_seq)

        log().save_logs() # TODO logger context?



def launch_single(logname, env_domain, env_task_name, description=None):
    init_logger("logdir", logname, description)
    init_random_seeds(RANDOM_SEED, cuda_determenistic=False)

    env = ActionRepeatWrapper(
        DMControlWrapper(env_domain, env_task_name, from_pixels=FROM_PIXELS, random_seed=RANDOM_SEED),
        action_repeat=ACTION_REPEAT
    )
    agent = Dreamer(env.state_dim, env.action_dim, device)
    train(env, agent)


def launch_single_pool(args):
    launch_single(*args)

def launch_set(suite_logname):
    os.mkdir(os.path.join("logdir", suite_logname))
    env_set = [("walker", "walk"), ("cartpole", "swingup")]

    launch_args = list(map(lambda it: (os.path.join(suite_logname, it[0] + "_" + it[1]), it[0], it[1]), env_set))
    with multiprocessing.Pool(len(env_set)) as p:
        p.map(launch_single_pool, launch_args)

if __name__ == "__main__":
    description = "tmp"
    launch_single("tmp", "cartpole", "balance", description)
    #  launch_set("tmplol_suite")
