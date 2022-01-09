import sys
sys.path.append("..")

import torch
import torchvision
import matplotlib.pyplot as plt
import random
import itertools

from utils.logger import init_logger, log
from utils.random import init_random_seeds

from envs import DMControlWrapper, ActionRepeatWrapper, dm_suite_benchmark
from Dreamer import Dreamer

device=torch.device("cpu")

def sample_episode(env, agent):
    obs = env.reset()
    hidden, action = agent.world_model.initial_state(batch_size=1) 
    hidden_list, action_list = [hidden], [action]
    obs_list = [torch.from_numpy(obs) + 0.5]
    action.squeeze_(0)

    for steps in itertools.count(1):
        action, hidden = agent(obs, hidden, action)
        obs, reward, discount, done = env.step(action)

        hidden_list.append(hidden)
        action_list.append(torch.from_numpy(action).unsqueeze(0))
        obs_list.append(torch.from_numpy(obs) + 0.5)
        if done:
            break
    return list(zip(hidden_list, action_list)), obs_list

def play_random(env, agent, horizon):
    states, obs = sample_episode(env, agent)
    init_state = random.choice(states)
    
    return play(agent, horizon, init_state)


def play_both(env, agent, horizon):
    states, obs = sample_episode(env, agent)
    init_state = states[10]
    
    images = play(agent, horizon, init_state)

    images_obs = torch.stack(obs[10:10+horizon+1])
    return torch.cat([images, images_obs], dim=0)


def play(agent, horizon, init_state=None):
    with torch.no_grad():
        if init_state is None:
            init_state = agent.world_model.initial_state(batch_size=1)
        hidden, prev_action = init_state

        state, action, reward, discount = agent.world_model.imagine(agent.agent, hidden, horizon)
        action = torch.cat([prev_action, action.squeeze(1)])
        condition = torch.cat([state.squeeze(1), action], dim=-1)

        embeds = agent.world_model.flow_model.sample(condition)
        images = agent.world_model.decoder(embeds) + 0.5
    return images



if __name__ == "__main__":
    RANDOM_SEED = 17923957
    ACTION_REPEAT = 2
    init_logger("logdir", "tmp", "play.py")
    init_random_seeds(RANDOM_SEED, cuda_determenistic=False)

    env = ActionRepeatWrapper(
        DMControlWrapper("walker", "walk", from_pixels=True, random_seed=RANDOM_SEED),
        action_repeat=ACTION_REPEAT
    )
    agent = Dreamer(env.state_dim, env.action_dim, device)
    agent.load_state_dict(torch.load("logdir/maf5.0/dreamer.torch"))


    HORIZON = 44
    images = play_both(env, agent, HORIZON)[::5]
    grid_image = torchvision.utils.make_grid(images, nrow=(HORIZON+1)//5)
    plt.imshow(grid_image.permute(1, 2, 0))
    plt.show()
