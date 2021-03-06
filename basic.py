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
from args import get_args

from collections import OrderedDict, Counter
import sys
sys.path.append('./replay')


#GLOBAL VARIABLES
INIT_WEIGHT = False
CUMULATIVE = True
TOP_N_CONSTRIANTS = 60
N_SAMPLES = 25
STEP_SIZE = 0.01
LOW_REW_SET = 20
BAD_STATE_VAR = 0.3

# number of trajectories for evaluation
SAMPLE_TRAJ = 20
EVAL_TRAJ = 1

def main():

    args = get_args()
    dir_name = "results/%s/%s-%s"%(args.env, "basic", strftime("%m_%d_%H_%M", gmtime()))
    os.makedirs(dir_name, exist_ok=True)
    logfile = open(dir_name+"/log.txt", "w")

    with open(os.path.join(dir_name,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env = gym.make(args.env)

    num_hidden = args.hidden_size
    VARIANCE = args.var

    # just to make more robust for differnet envs
    if args.policy == "linear":
        device = torch.device("cpu")
        N_SAMPLES = args.n_samples if args.n_samples>0 else int(env.observation_space.shape[0]*1.5)
        LOW_REW_SET = N_SAMPLES*2
        TOP_N_CONSTRIANTS = int(N_SAMPLES*1.5)
        def make_policy():
            return Policy_lin(env.observation_space.shape[0],
                            env.action_space.shape[0]).to(device)
    elif args.policy == "nn": #assume it's 2 layer here
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        N_SAMPLES = args.n_samples if args.n_samples>0 else int(env.observation_space.shape[0]*2)
        print(env.observation_space.shape[0])
        LOW_REW_SET = N_SAMPLES*2
        TOP_N_CONSTRIANTS = int(N_SAMPLES*2)
        
        def make_policy():
            return Policy_quad(env.observation_space.shape[0],
                        env.action_space.shape[0],
                        num_hidden=num_hidden).to(device)

    print('Using device:', device)

    sample_policy, sample_eval = make_policy(), -1700

    replay_buffer = Replay_buffer(args.gamma)

    ep_no_improvement = 0
    iter_steps = args.iter_steps

    for i_episode in count(1):
        # hack
        if ep_no_improvement > 5:
            N_SAMPLES = int(N_SAMPLES * 1.15)
            TOP_N_CONSTRIANTS = int(N_SAMPLES*1.2)
            if TOP_N_CONSTRIANTS > iter_steps:
                iter_steps = TOP_N_CONSTRIANTS*1.5

            VARIANCE = VARIANCE/1.2
            print("Updated Var to: %.3f"%(VARIANCE))
            ep_no_improvement = 0

        # Exploration
        num_steps = 0
        explore_episodes = 0
        explore_rew =0

        while num_steps < iter_steps:
            state = env.reset()
            state_action_rew_env = []
            lowest_rew = []
            name_str = "expl_var" #explore
            for t in range(1000): 
                action = sample_policy.select_action(state, VARIANCE).flatten()
                next_state, reward, done, _ = env.step(action)
                if args.render:
                    env.render()
                explore_rew += reward

                replay_buffer.push((state,next_state,action, reward, done, (name_str, explore_episodes, t))) 

                if done:
                    break
                state = next_state

            num_steps += (t-1)
            explore_episodes += 1

        explore_rew /= explore_episodes

        print('\nEpisode {}\tExplore reward: {:.2f}\tAverage ep len: {:.1f}\n'.format(i_episode, explore_rew, num_steps/explore_episodes))
    
        low_rew_constraints_set = []

            
        best_tuples = replay_buffer.best_state_actions_replace(top_n_constraints=TOP_N_CONSTRIANTS, by='rewards', discard = True)

        # sample and solve
        max_policy, max_eval, max_set = sample_policy, sample_eval, best_tuples

        for branch in range(args.branches):

            branch_policy = make_policy()
            branch_buffer = Replay_buffer(args.gamma)

            if N_SAMPLES >= len(best_tuples):
                constraints = best_tuples
            else:   
                constraints = random.sample(best_tuples+low_rew_constraints_set, N_SAMPLES)
            print(all_l2_norm(constraints)[:5])

            # Get metadata of constraints
            states, actions, info, rewards, _ = zip(*constraints)
            print(actions)
            print("ep %d b %d: %d constraints mean: %.3f  std: %.3f  max: %.3f" % ( i_episode, branch, len(constraints), np.mean(rewards), np.std(rewards), max(rewards)))
            
            print(len(info))

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
                    action = branch_policy.select_action(state,0).flatten()
                    next_state, reward, done, _ = env.step(action)
                    if args.render:
                        env.render()
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
            logfile.write('Episode {}\tBranch: {}\tConstraints:{}\tEval reward: {:.2f}\n'.format(i_episode, branch, len(constraints), eval_rew))

            if eval_rew > max_eval:
                print("updated to this policy")
                max_eval, max_policy, max_set = eval_rew, branch_policy, constraints
                if args.keep_buffer:
                    replay_buffer.append(branch_buffer)
                else:
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
    if isinstance(states[0], torch.Tensor):
        states = [s.cpu().numpy() for s in states]
    all_dist = []
    zerodist = 0
    for i, x1 in enumerate(states):
        for x2 in states[i+1:]:
            d=np.linalg.norm(np.subtract(x1,x2))
            if d - 0 < 1e-2:
                zerodist +=1
            all_dist.append(d)
    if zerodist > 0 :
        print("0 distances: %d" % zerodist)
    return sorted(all_dist)

if __name__ == '__main__':
    main()