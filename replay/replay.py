import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

import sys
sys.path.append('a2c_ppo_acktr')
from .a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from .a2c_ppo_acktr.utils import get_render_func, get_vec_normalize



class Trained_model_wrapper():
    def __init__(self, env_name, load_dir, seed):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.env = make_vec_envs(
            args.env_name,
            args.seed,
            1,
            None,
            None,
            device="cuda:0",
            allow_early_resets=False)

        # We need to use the same statistics for normalization as used in training
        self.actor_critic, ob_rms = \
                    torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

        vec_norm = get_vec_normalize(env)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms

    
    def select_action(self, state, variance): #variance is a dummy
        normalized = self.env._obfilt(state)
        recurrent_hidden_states = torch.zeros(1,
                        self.actor_critic.recurrent_hidden_state_size).to(self.device)
        value, action, _, _ = self.actor_critic.act(
                    normalized, recurrent_hidden_states, masks, deterministic=True)
        return action


    def play(self):

        recurrent_hidden_states = torch.zeros(1,
                                actor_critic.recurrent_hidden_state_size).to(self.device)
        masks = torch.zeros(1, 1)

        obs = env.reset()

        rew = 0
        while True:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = self.actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=True)

            # Obser reward and next obs
            obs, reward, done, _ = self.env.step(action)
            rew += reward.numpy()
            masks.fill_(0.0 if done else 1.0)

            if done:
                print(rew)
                rew = 0


