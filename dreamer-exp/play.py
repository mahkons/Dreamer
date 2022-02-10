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


def test_decoder(env, agent):
    with torch.no_grad():
        _, obs = sample_episode(env, agent)
        obs = torch.stack(obs[::5]).float() # only every 5th
        embed = agent.world_model.encoder(obs)
        rec = agent.world_model.decoder(embed) + 0.5
    return rec


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
    pretrained = torch.load("logdir/dreamer.torch")

    #  encoder_dict = agent.world_model.encoder.state_dict()
    #  decoder_dict = agent.world_model.decoder.state_dict()
    #  encoder_dict.update({k: pretrained["world_model.encoder." + k] for k in encoder_dict.keys()})
    #  decoder_dict.update({k: pretrained["world_model.decoder." + k] for k in decoder_dict.keys()})
    #  agent.world_model.encoder.load_state_dict(encoder_dict)
    #  agent.world_model.decoder.load_state_dict(decoder_dict)


    agent.eval()


    images = test_decoder(env, agent)[:10]
    grid_image = torchvision.utils.make_grid(images, nrow=5)
    plt.imshow(grid_image.permute(1, 2, 0))
    plt.show()
