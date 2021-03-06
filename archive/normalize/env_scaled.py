import argparse, gym, copy, math, pickle, torch, random, json, os
import numpy as np
from itertools import count
from heapq import nlargest
from time import gmtime, strftime
from operator import itemgetter
import torch.nn as nn
import torch as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.distributions import Categorical, Bernoulli
from copy import deepcopy
from policies import *
from collections import OrderedDict, Counter

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from Normalizer import *

from args import get_args

#GLOBAL VARIABLES
INIT_WEIGHT = False
CUMULATIVE = True
TOP_N_CONSTRIANTS = 30
N_SAMPLES = 25
STEP_SIZE = 0.01
LOW_REW_SET = 20
BAD_STATE_VAR = 0.3

# number of trajectories for evaluation
SAMPLE_TRAJ = 20
EVAL_TRAJ = 20


def make_env(env, seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    def _thunk():
        env_ = gym.make(env)
        env_.seed(seed)
        return env_
    envs = DummyVecEnv([_thunk])
    envs = VecNormalize(envs, ret=False)
    envs = VecPyTorch(envs, device)
    return envs


def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dir_name = "results/%s/%s-%s"%(args.env, "basic", strftime("%m_%d_%H_%M", gmtime()))
    os.makedirs(dir_name, exist_ok=True)
    logfile = open(dir_name+"/log.txt", "w")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env = make_env(args.env, args.seed, device)

    num_hidden = args.hidden_size
    VARIANCE = args.var

    # just to make more robust for differnet envs
    if args.policy == "linear":
        device = torch.device("cpu")
        N_SAMPLES = env.observation_space.shape[0]
        LOW_REW_SET = N_SAMPLES*2
        TOP_N_CONSTRIANTS = int(N_SAMPLES*1.5)
        def make_policy():
            return Policy_lin(env.observation_space.shape[0],
                            env.action_space.shape[0]).to(device)
    elif args.policy == "nn": #assume it's 2 layer here
        N_SAMPLES = int(env.observation_space.shape[0]*2) #(num_hidden) a little underdetermined
        LOW_REW_SET = N_SAMPLES*2
        TOP_N_CONSTRIANTS = int(N_SAMPLES * 2)
        def make_policy():
            return Policy_quad(env.observation_space.shape[0],
                                env.action_space.shape[0],
                                num_hidden=num_hidden).to(device)

    print('Using device:', device)

    sample_policy, sample_eval = make_policy(), -1700

    replay_buffer = Replay_buffer(args.gamma)

    ep_no_improvement = 0


    for i_episode in count(1):

        # hack
        if ep_no_improvement > 3:
            N_SAMPLES = int(N_SAMPLES * 1.5)
            TOP_N_CONSTRIANTS = int(N_SAMPLES*1.5)
            VARIANCE = VARIANCE/1.2
            print("Updated Var to: %.3f"%(VARIANCE))
            ep_no_improvement = 0


        # Exploration
        num_steps = 0
        explore_episodes = 0
        explore_rew =0

        while num_steps < args.iter_steps:
            state = env.reset()

            state_action_rew_env = []
            lowest_rew = []

            for t in range(1000): 
                action = sample_policy.select_action(state, VARIANCE)
                name_str = "expl_var" #explore
                if args.correct:
                    if num_steps < 200:
                        copied_env = deepcopy(env)
                next_state, reward, done, _ = env.step(action)
                reward = reward.detach().numpy().item()
                explore_rew += reward

                replay_buffer.push((state,next_state,action, reward, done, (name_str, explore_episodes, t))) 
                
                if args.correct:
                    if (ENV == "Hopper-v2" or ENV == "Walker2d-v2") and done:
                        reward = float('-inf')
                    if len(lowest_rew) < LOW_REW_SET or (ENV == "Hopper-v2" or ENV == "Walker2d-v2" and done):
                        state_action_rew_env.append([state,action,reward,copied_env])
                        lowest_rew.append(reward)
                    elif reward < max(lowest_rew):
                        state_action_rew_env = sorted(state_action_rew_env, key=lambda l: l[2]) #sort by reward
                        state_action_rew_env[-1] = [state,action,reward,copied_env]
                        lowest_rew.remove(max(lowest_rew))
                        lowest_rew.append(reward)

                if done:
                    break
                state = next_state

            num_steps += (t-1)
            explore_episodes += 1

        explore_rew /= explore_episodes

        print('\nEpisode {}\tExplore reward: {:.2f}\tAverage ep len: {:.1f}\n'.format(i_episode, explore_rew, num_steps/explore_episodes))

        if args.correct:
            print("exploring better actions")
            low_rew_constraints_set = []

            #sample possible corrections
            for s, a, r, saved_env in state_action_rew_env:
                max_r, max_a = r, a
                for i in range(20): #sample 20 different actions
                    step_env = deepcopy(saved_env)
                    action_explore = sample_policy.select_action(s, BAD_STATE_VAR)
                    _, reward, done, _ = step_env.step(action_explore)
                    if reward > max_r and not done:
                        max_r, max_a = reward, action_explore
                if max_r - r >= 0.1:
                    low_rew_constraints_set.append((s, max_a,"bad_states", max_r, 0))
                    print("improved bad state from %.3f to %.3f" %(r, max_r))
                if len(low_rew_constraints_set) > N_SAMPLES/3:
                    break #enough bad correction constraints
        else:
            low_rew_constraints_set = []

            
        best_tuples = replay_buffer.best_state_actions(top_n_constraints=TOP_N_CONSTRIANTS, by='rewards', discard = True)

        # sample and solve
        max_policy, max_eval, max_set = sample_policy, sample_eval, best_tuples

        for branch in range(args.branches):

            branch_policy = make_policy()
            branch_buffer = Replay_buffer(args.gamma)

            constraints = random.sample(best_tuples+low_rew_constraints_set, N_SAMPLES)
            print(all_l2_norm(constraints)[:5])

            # Get metadata of constraints
            states, actions, info, rewards, _ = zip(*constraints)
            print("ep %d b %d: %d constraints mean: %.3f  std: %.3f  max: %.3f" % ( i_episode, branch, len(constraints), np.mean(rewards), np.std(rewards), max(rewards)))
            print(info)

            if isinstance(states[0], torch.Tensor):
                states = torch.cat(states)
            else:
                states = torch.tensor(states).float()
            
            if isinstance(actions[0], torch.Tensor):
                actions = torch.cat(actions)
            else: 
                actions = torch.tensor(actions).float()
           
            
            branch_policy.train(states.to(device), actions.to(device), epoch=args.training_epoch)
           
            # Evaluate
            eval_rew = 0
            for i in range(EVAL_TRAJ):
                state, done = env.reset(), False
                step = 0
                while not done: # Don't infinite loop while learning
                    action = branch_policy.select_action(state,0)
                    next_state, reward, done, _ = env.step(action)
                    reward = reward.detach().numpy().item()
                    eval_rew += reward
                    branch_buffer.push((state, next_state, action, reward, done, ("eval", i, step))) 
                    state = next_state
                    step += 1
                    if args.render:
                        env.render()
                    if done:
                        break

            eval_rew /= EVAL_TRAJ

            #log
            print('Episode {}\tBranch: {}\tEval reward: {:.2f}\tExplore reward: {:.2f}'.format(
                i_episode, branch, eval_rew, explore_rew))
            logfile.write('Episode {}\tBranch: {}\tEval reward: {:.2f}\n'.format(i_episode, branch, eval_rew))

            if eval_rew > max_eval:
                print("updated to this policy")
                max_eval, max_policy, max_set = eval_rew, branch_policy, constraints
                replay_buffer = branch_buffer

        # the end of branching
        if max_eval > sample_eval:
            with open("%s/%d_constraints.p"%(dir_name,i_episode), "wb") as f:
                pickle.dump({"all": best_tuples, "constraints": max_set}, f)

            with open("%s/%d_policy.p"%(dir_name,i_episode), 'wb') as out:
                policy_state_dict = OrderedDict({k:v.to('cpu') for k, v in max_policy.state_dict().items()})
                pickle.dump(policy_state_dict, out)

            sample_policy, sample_eval = max_policy, max_eval
            ep_no_improvement = 0
        else:
            ep_no_improvement +=1


def all_l2_norm(constraints):
    states, _, _, _ ,_ = zip(*constraints)
    states = [s.cpu().numpy() for s in states]
    all_dist = []
    for i, x1 in enumerate(states):
        for x2 in states[i+1:]:
            d=np.linalg.norm(np.subtract(x1,x2))
            if d - 0 < 1e-2:
                print("0 dist!!")
            all_dist.append(d)
    return sorted(all_dist)


if __name__ == '__main__':
    main()