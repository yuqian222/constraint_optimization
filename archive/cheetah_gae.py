import argparse, gym, copy, math, pickle, torch, random, json
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
from gurobipy import *
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
INIT_WEIGHT = True
CUMULATIVE = True
PRINT_RESULTS = False
VAR_BOUND = 1.0
SLACK_BOUND = 0.005
TOP_N_CONSTRIANTS = 50
N_SAMPLES = 18
VARIANCE = 0.01
BRANCHES = 20
NOVELTY_SLACK = 0

env = gym.make('HalfCheetah-v2')
env.seed(args.seed)
torch.manual_seed(args.seed)
num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]


def select_action(state, policy, variance=0.1, record=True):
    action = policy(state).data.numpy()
    action = np.random.normal(action, [variance]*len(action))
    return action


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def best_state_actions(states, actions, rewards, info, top_n_constraints=-1): 
    if top_n_constraints > 0:
        top_n = nlargest(top_n_constraints, zip(states, actions, rewards, info), key=lambda s: s[2])
        return top_n
    else:
        return zip(states, actions, rewards, info)


def sample(policy, value_net, optimizer, env, total_steps, var=0):
    num_steps, rew, rollouts, t = 0, 0, 0, 0
    state = env.reset()
    states    = [state]
    actions   = []
    values    = []
    rewards   = []
    masks     = []
    info      = []

    while num_steps < total_steps:
        state = torch.FloatTensor(state)
        value = value_net(state)

        if num_steps < 20000: #TODO
            action = select_action(state, policy, variance=var)
            name_str = "expl" #explore
        else: 
            action = select_action(state, policy, variance=0) # 20% randomly good
            name_str = "eval" #exploit

        next_state, reward, done, _ = env.step(action)

        states.append(next_state)
        actions.append(action)
        values.append(value)
        rewards.append(torch.FloatTensor([reward]).unsqueeze(1))
        masks.append(torch.FloatTensor([1 - done]).unsqueeze(1))
        info.append((t, "%s_%d"%(name_str, rollouts)))
        rew += reward
        rollouts += done
        num_steps += 1
        if done:
            t = 0
            rollouts += 1
            state = env.reset()
        else:
            t += 1
            state = next_state

    next_value = value_net(torch.FloatTensor(next_state))
    returns = compute_gae(next_value, rewards, masks, values)
    advantages = torch.cat(returns).detach() - torch.cat(values)
    
    total_reward = rew/rollouts
    return states, actions, advantages.detach().numpy(), info, total_reward

def main():

    dir_name = "results/"+strftime("%m_%d_%H_%M", gmtime())
    os.makedirs(dir_name, exist_ok=True)
    logfile = open(dir_name+"/log.txt", "w")

    value_net = Value(num_inputs)
    optimizer = optim.RMSprop(value_net.parameters())

    sample_policy, sample_eval = Policy_lin(num_inputs, num_outputs), float("-inf")

    if INIT_WEIGHT:
        sample_eval = 1300
        with open("save_1000.p",'rb') as f:
            params=pickle.load(f)
            sample_policy.init_weight(params)


    for i_episode in count(1):
        # Exploration
        my_states = {}
        
        states, actions, advantages, info, sample_rew = sample(sample_policy, 
                                                   value_net, 
                                                   optimizer, 
                                                   env, 
                                                   total_steps=25000, var=VARIANCE)

        print('\nEpisode {}\tSample reward: {:.2f}\n'.format(i_episode, sample_rew))
        logfile.write('Episode {}\tSample reward: {:.2f}'.format(i_episode, sample_rew))


        # create constraints
        my_states = create_state_dict(states, actions, advantages, info, my_states)
        constraints_dict = bestStates(my_states, top_n_constraints=TOP_N_CONSTRIANTS)
        
        max_policy, max_eval, max_set = sample_policy, sample_eval, constraints_dict

        for branch in range(BRANCHES):
            branch_policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
            constraints = dict(random.sample(constraints_dict.items(), N_SAMPLES))

            prob = Model("mip1")
            (firstParam, firstBias) = initializeLimits(branch_policy, prob)

            # Get metadata of constraints
            constraint_info = list(constraints.values())
            vals = [list(v.values())[0][0] for v in constraint_info]
            print("ep %d b %d: constraint mean: %.3f  std: %.3f  max: %.3f" % (i_episode, branch, np.mean(vals), np.std(vals), max(vals)))
            print("constraint set's episode and step number:")
            print([list(v.values())[0] for v in constraint_info])

            # Solve
            exit = solvePolicy(constraints, branch_policy, firstParam, firstBias, prob)
            
            if exit == 0:
                updateParam(prob, branch_policy)
            else:
                print('Episode {}\tBranch: {}\tUnsat'.format(i_episode, branch))
                print("L2_NORM")
                print(all_l2_norm(constraints))
                continue

            # Evaluate
            states, actions, advantages, info, eval_rew = sample(branch_policy, 
                                                               value_net, 
                                                               optimizer, 
                                                               env, 
                                                               total_steps=6000, var=0.0)

            #log
            print('Episode {}\tBranch: {}\tEval reward: {:.2f}\tSample reward: {:.2f}'.format(
                i_episode, branch, eval_rew, sample_rew))
            logfile.write('Episode {}\tBranch: {}\tEval reward: {:.2f}'.format(i_episode, branch, eval_rew))

            if eval_rew > max_eval:
                max_eval, max_policy, max_set = eval_rew, branch_policy, constraints

        # the end of branching
        if max_eval > sample_eval - NOVELTY_SLACK:
            with open("%s/%d.p"%(dir_name,i_episode), "wb") as f:
                pickle.dump({"all": constraints_dict, "constraints": max_set}, f)
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
