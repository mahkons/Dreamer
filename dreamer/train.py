import sys
sys.path.append("..")

import itertools


from envs.DMControlWrapper import DMControlWrapper
from utils.logger import init_logger, log

from Dreamer import Dreamer
from ReplayBuffer import Episode, ReplayBuffer





if __name__ == "__main__":
    init_logger("tmplol")
    #TODO init random seeds

    env = DMControlWrapper()
    agent = Dreamer(env.action_size)
    memory = ReplayBuffer(1e4)

    for i in range(10):
        state = env.reset()
        episode = Episode(state)
        for steps in itertools.count(1):
            action = agent(state)
            next_state, reward, done = env.step(action)
            episode.add_transition(action, reward, next_state, done)
            state = next_state
            if done:
                break
        memory.push(episode)
        print(episode.rewards.sum())



