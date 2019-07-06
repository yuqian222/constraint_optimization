import argparse, gym, copy, math, pickle, torch, random, json
from copy import deepcopy
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
from policies import *


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

#GLOBAL VARIABLES
ENV = 'HalfCheetah-v2'
INIT_WEIGHT = False
CUMULATIVE = True
LOW_REW_SET = 35
#TOP_N_CONSTRIANTS = 30
N_SAMPLES = 25
VARIANCE = 0.2
BAD_STATE_VAR = 0.3
STEP_SIZE = 0.01
BRANCHES = 5
NOVELTY_SLACK = 0


def select_action(state, policy, variance=0.1, record=True):

    new_state = torch.from_numpy(state).unsqueeze(0)
    action = policy(new_state.float())
    action = action.data[0].numpy()
    action = np.random.normal(action, [variance]*len(action))

    if record:
        policy.saved_action.append(tuple(action))
        policy.saved_state.append(tuple(state))
    return action

def calculate_rewards(policy):
    R = 0
    rewards = []
    info = []
    if CUMULATIVE:
        for (r,step,name) in policy.rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
            info.insert(0, (step,name))
            if step == 0:
                R = 0
    else:
        rewards = policy.rewards
    return policy.saved_state, policy.saved_action, rewards, info


def best_state_actions(states, actions, rewards, info, top_n_constraints=-1): 
    if top_n_constraints > 0:
        top_n = nlargest(top_n_constraints, zip(states, actions, rewards, info), key=lambda s: s[2])
        return top_n
    else:
        return zip(states, actions, rewards, info)

def eval_policy(branch_policy, env):
    num_steps = 0
    eval_episodes = 0
    eval_rew = 0
    while num_steps < 6000:
        state = env.reset()
        eval_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state, branch_policy, variance=0, record=False)
            state, reward, done, _ = env.step(action)
            eval_rew += reward
            if args.render:
                env.render()
            if done:
                break
        num_steps += (t-1)
        eval_episodes += 1
    eval_rew /= eval_episodes
    branch_policy.clean()
    return eval_rew


def main():

    dir_name = "results/"+strftime("%m_%d_%H_%M", gmtime())
    os.makedirs(dir_name, exist_ok=True)
    logfile = open(dir_name+"/log.txt", "w")

    env = gym.make(ENV)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    q_function = Value(env.observation_space.shape[0] + env.action_space.shape[0], num_hidden=64)
    Policy = Policy_lin
  
    sample_policy, sample_eval = Policy(env.observation_space.shape[0], 
                                            env.action_space.shape[0]), -1700

    if INIT_WEIGHT:
        sample_eval = 1300
        with open("save_1000.p",'rb') as f:
            params=pickle.load(f)
            sample_policy.init_weight(params)

    for i_episode in count(1):
        # Exploration  

        state = env.reset()
        copied_env = deepcopy(env)
        
        state_action_rew_env = []
        lowest_rew = []

        num_steps = 0

        for t in range(999): #one trajectory, can change here
            action = select_action(state, sample_policy, variance=0)
            next_state, reward, done, _ = env.step(action)            
            sample_policy.rewards.append((reward, t, "init"))

            if len(lowest_rew) < LOW_REW_SET:
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
            copied_env = deepcopy(env)


        print("finished first trajectory")
        # explore better actions
        low_rew_constraints_set = []

        for s, a, r, saved_env in state_action_rew_env:
            max_r, max_a = r, a
            for i in range(10): #sample 10 different actions
                this_reward = []
                action_explore = select_action(s, sample_policy, variance=BAD_STATE_VAR, record=False)
                _, reward, _, _ = env.step(action_explore)
                if reward > max_r:
                    max_r, max_a = reward, action_explore
            if max_r > r:
                low_rew_constraints_set.append((s, max_a, max_r, "bad_states"))
                print("improved bad state from %.3f to %.3f" %(r, max_r))
                print(a)
                print(max_a)



        states, actions, rewards, info = calculate_rewards(sample_policy)
        
        best_tuples = [] #best_state_actions(states, actions, rewards, info, top_n_constraints=TOP_N_CONSTRIANTS)

        sample_policy.clean()

        # sample and solve
        
        max_policy, max_eval, max_set = sample_policy, sample_eval, best_tuples


        for branch in range(BRANCHES):
            
            branch_policy = copy.deepcopy(sample_policy)
            
            constraints = random.sample(low_rew_constraints_set, N_SAMPLES)

            # Get metadata of constraints
            states, actions, rewards, info = zip(*constraints)

            branch_policy.train(states, actions, epoch=1000)
            eval_rew = eval_policy(branch_policy, env)

            #log
            print('Episode {}\tBranch: {}\tReplay Eval reward: {:.2f}\t'.format(i_episode, branch, eval_rew, ))
            logfile.write('Episode {}\tBranch: {}\tEval reward: {:.2f}\n'.format(i_episode, branch, eval_rew))

                    
            if eval_rew > max_eval:
                print("updated to this policy")
                max_eval, max_policy, max_set = eval_rew, branch_policy, constraints

            
        # the end of branching
        if max_eval > sample_eval - NOVELTY_SLACK:
            with open("%s/%d_constraints.p"%(dir_name,i_episode), "wb") as f:
                pickle.dump({"all": best_tuples, "constraints": max_set}, f)

            with open("%s/%d_policy.p"%(dir_name,i_episode), 'wb') as out:
                pickle.dump(max_policy, out)


            sample_policy, sample_eval = max_policy, max_eval
        

def all_l2_norm(constraints):
    states = list(constraints.keys())
    all_dist = []
    for i, x1 in enumerate(states):
        for x2 in states[i+1:]:
            d=np.linalg.norm(np.subtract(x1,x2))
            if d - 0 < 1e-2:
                print("0 dist at state %s with action %s and %s" %(str(x1), str(list(constraints[x1].keys())[0]),str(list(constraints[x2].keys())[0])))
            all_dist.append(d)
    return sorted(all_dist)


if __name__ == '__main__':
    main()
