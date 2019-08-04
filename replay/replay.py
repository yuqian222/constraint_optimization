import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from .a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from .a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

class Trained_model_wrapper():
    def __init__(self, env_name, load_dir, seed):
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        self.env = make_vec_envs(
            env_name,
            seed,
            1,
            None,
            None,
            device=device_name,
            allow_early_resets=False)

        # We need to use the same statistics for normalization as used in training
        self.actor_critic, ob_rms = \
                    torch.load(os.path.join(load_dir, env_name + ".pt"), map_location=device_name)

        vec_norm = get_vec_normalize(self.env)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms

    
    def select_action(self, state, variance): #variance is a dummy
        normalized = torch.Tensor(self.env.get_normalized([state])).to(self.device)
        recurrent_hidden_states = torch.zeros(1,self.actor_critic.recurrent_hidden_state_size).to(self.device)
        value, action, _, _ = self.actor_critic.act(normalized, recurrent_hidden_states, torch.zeros(1, 1), deterministic=True)
        return action.detach().cpu().float().numpy()

    def play(self, n=10):
        recurrent_hidden_states = torch.zeros(1,
                                self.actor_critic.recurrent_hidden_state_size).to(self.device)
        masks = torch.zeros(1, 1)

        obs = self.env.reset()
        rew = 0
        traj = 0
        while traj < n:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = self.actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=True)

            # Obser reward and next obs
            obs, reward, done, _ = self.env.step(action)
            rew += reward.numpy()
            masks.fill_(0.0 if done else 1.0)

            if done:
                print(rew)
                traj += 1
                rew = 0


if __name__ == '__main__':
    Trained_model_wrapper("Hopper-v2", "./trained_models/", 567).play()

