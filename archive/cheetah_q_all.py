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
INIT_WEIGHT = False
CUMULATIVE = True
PRINT_RESULTS = False
SOLVE = False
VAR_BOUND = 1.0
SLACK_BOUND = 0.005
TOP_N_CONSTRIANTS = 100
N_SAMPLES = 60
VARIANCE = 0.1
BRANCHES = 20
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

def calculate_rewards(myround, policy):
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


def create_state_dict(states, actions, rewards, info, my_states):
    # update the state_reward dictionary
    for s, a, r, i in zip(states, actions, rewards, info):
        step,name = i[0], i[1]
        if s in my_states:
            print("duplicate state")
            if (a in my_states[s]):
                print ("duplicate a")
                my_states[s][a] = ((r+my_states[s][a][0])/2,step,name)
            else:
                my_states[s][a] = (r,step,name)
        else:
            my_states[s] = {}
            my_states[s][a] = (r,step,name)

    return my_states



def bestStates(my_states, top_n_constraints=-1): 
    max_act_dict = {}
    max_vals_dict = {}

    for chunk_s in my_states:
        max_action = max(my_states[chunk_s].items(), key=operator.itemgetter(1))[0]
        max_act_dict[chunk_s] = max_action
        max_vals_dict[chunk_s] = my_states[chunk_s][max_action][0]

    # Get metadata of values
    vals = list(max_vals_dict.values())

    print("Max values mean: %.3f  std: %.3f  max: %.3f" % (np.mean(vals), np.std(vals), max(vals)))

    if top_n_constraints > 0:
        top_n = nlargest(top_n_constraints, max_act_dict.keys(), key=lambda s: max_vals_dict[s])
        top_n_dict = {k: v for k, v in my_states.items() if k in top_n}
    else:
        top_n_dict= my_states

    return top_n_dict

    
def main():

    dir_name = "results/"+strftime("%m_%d_%H_%M", gmtime())
    os.makedirs(dir_name, exist_ok=True)
    logfile = open(dir_name+"/log.txt", "w")

    env = gym.make('HalfCheetah-v2')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    q_function = Value(env.observation_space.shape[0] + env.action_space.shape[0], num_hidden=48)
    Policy = Policy_lin
  
    sample_policy, sample_eval = Policy(env.observation_space.shape[0], 
                                            env.action_space.shape[0], 
                                            VAR_BOUND, SLACK_BOUND), float("-inf")

    if INIT_WEIGHT:
        sample_eval = 1300
        with open("save_1000.p",'rb') as f:
            params=pickle.load(f)
            sample_policy.init_weight(params)


    for i_episode in count(1):
        # Exploration
        num_steps = 0
        explore_episodes = 0
        explore_rew =0
        my_states = {}
        while num_steps < 15000:
            state = env.reset()
            for t in range(1000): # Don't infinite loop while learning
                if num_steps < 12000:
                    action = select_action(state, sample_policy, variance=VARIANCE)
                    name_str = "expl" #explore
                else: 
                    action = select_action(state, sample_policy, variance=0) # 20% randomly good
                    name_str = "eval" #exploit

                next_state, reward, done, _ = env.step(action)
                explore_rew += reward
                sample_policy.rewards.append((reward, t, "%s_%d"%(name_str, explore_episodes)))
                if args.render:
                    env.render()
                if done:
                    break
                state = next_state
            num_steps += (t-1)
            explore_episodes += 1
        
        explore_rew /= explore_episodes

        states, actions, rewards, info = calculate_rewards(explore_episodes, sample_policy)
        input_tensor = torch.cat((torch.Tensor(states), torch.Tensor(actions)),1)
        q_function.train(input_tensor, rewards, epoch=6)

        action_grad = []
        for s, a in zip(states, actions):
            a_new = q_function.calculate_action_grad(torch.Tensor(s), torch.Tensor(a))
            action_grad.append(tuple(a_new.detach().numpy()))

        print(actions[:10])
        print(action_grad[:10])

        print('\nEpisode {}\tExplore reward: {:.2f}\n'.format(i_episode, explore_rew))
        sample_policy.train(states, action_grad)
        sample_policy.clean()


        # Evaluate
        num_steps = 0
        eval_episodes = 0
        eval_rew = 0
        while num_steps < 6000:
            state = env.reset()
            eval_sum = 0
            for t in range(10000): # Don't infinite loop while learning
                action = select_action(state, sample_policy, variance=0)
                state, reward, done, _ = env.step(action)
                eval_rew += reward
                sample_policy.rewards.append((reward, t, "%s_%d"%(name_str, explore_episodes)))

                if args.render:
                    env.render()
                if done:
                    break
            num_steps += (t-1)
            eval_episodes += 1
        eval_rew /= eval_episodes

        
        states, actions, rewards, info = calculate_rewards(explore_episodes, sample_policy)
        input_tensor = torch.cat((torch.Tensor(states), torch.Tensor(actions)),1)
        q_function.train(input_tensor, rewards)
        sample_policy.clean()

        #log
        print('Episode {}\tEval reward: {:.2f}\tExplore reward: {:.2f}'.format(
            i_episode, eval_rew, explore_rew))
        logfile.write('Episode {}\tEval reward: {:.2f}\n'.format(i_episode, eval_rew))


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
