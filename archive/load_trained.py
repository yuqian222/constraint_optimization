import os
import time
import numpy as np
import os.path as os
import tensorflow as tf
from baselines import logger

from collections import deque

import cv2

import matplotlib.pyplot as plt



from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner
from baselines.ppo2.model import Model
from baselines.ppo2.runner import Runner
from baselines.run import train

def get_model_env(env):

    env_id = env
    env_type = 'mujoco'

    total_timesteps = 0
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )
    return model, env

def load_model(path, render=False):

    model, env = train(args, extra_args)
    model.load(path)

    if render: #play
        obs = env.reset()
        score = 0
        done = False

        while done == False:
            # Get the action
            actions, values, _ = model.step(obs)

            # Take actions in env and look the results
            obs, rewards, done, info = env.step(actions)

            score += rewards

            env.render()

        print("Score ", score)
        env.close()

    return model, env