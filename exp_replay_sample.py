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
from collections import OrderedDict


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

parser = argparse.ArgumentParser(description='cheetah_q parser')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--env', type=str, default='HalfCheetah-v2',
                    help='enviornment (default: HalfCheetah-v2)')

parser.add_argument('--policy', type=str, default="linear", #or can be nn
                    help='policy trained. can be or linear or nn')

parser.add_argument('--branches', type=int, default=5, metavar='N',
                    help='branches per round (default: 5)')
parser.add_argument('--iter_steps', type=int, default=20000, metavar='N',
                    help='num steps per iteration (default: 20,000)')

parser.add_argument('--var', type=float, default=0.05,
                    help='sample variance (default: 0.1)')
parser.add_argument('--hidden_size', type=int, default=24,
                    help='hidden size of policy nn (default: 24)')

parser.add_argument('--training_epoch', type=int, default=500,
                    help='Training epochs for each policy update (default: 500)')
args = parser.parse_args()

#GLOBAL VARIABLES
ENV = args.env
INIT_WEIGHT = False
CUMULATIVE = True
TOP_N_CONSTRIANTS = 60
N_SAMPLES = 25
VARIANCE = args.var
STEP_SIZE = 0.01
BRANCHES = args.branches
POLICY = args.policy
MAX_STEPS = args.iter_steps
HIDDEN_SIZE = args.hidden_size
LOW_REW_SET = 20
BAD_STATE_VAR = 0.3

def select_action(state, policy, variance=0.1, record=True):

    new_state = torch.from_numpy(state).unsqueeze(0)
    action = policy(new_state.float().to(device))
    action = action.data[0].cpu().numpy()
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


def best_state_actions(states, actions, rewards, info, top_n_constraints=-1): 
    if top_n_constraints > 0:
        top_n = nlargest(top_n_constraints, zip(states, actions, rewards, info), key=lambda s: s[2])
        return top_n
    else:
        return zip(states, actions, rewards, info)


def main():

    dir_name = "results/%s/%s"%(ENV, strftime("%m_%d_%H_%M", gmtime()))
    os.makedirs(dir_name, exist_ok=True)
    logfile = open(dir_name+"/log.txt", "w")

    env = gym.make(ENV)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    num_hidden = HIDDEN_SIZE
    var = VARIANCE

    # just to make more robust for differnet envs
    if POLICY == "linear":
        N_SAMPLES = env.observation_space.shape[0]
        TOP_N_CONSTRIANTS = N_SAMPLES*2
        Policy = Policy_lin
    elif POLICY == "nn": #assume it's 2 layer here
        N_SAMPLES = int(env.observation_space.shape[0]*2) #(num_hidden) a little underdetermined
        LOW_REW_SET = N_SAMPLES
        TOP_N_CONSTRIANTS = int(N_SAMPLES*2)
        Policy = Policy_quad
            
  
    sample_policy, sample_eval = Policy(env.observation_space.shape[0], 
                                        env.action_space.shape[0], num_hidden=num_hidden), -1700
    sample_policy = sample_policy.to(device)

    if INIT_WEIGHT:
        sample_eval = 1300
        with open("save_1000.p",'rb') as f:
            params=pickle.load(f)
            sample_policy.init_weight(params)

    ep_no_improvement = 0

    for i_episode in count(1):

        # hack
        if ep_no_improvement > 3:
            N_SAMPLES = int(N_SAMPLES * 1.2)
            TOP_N_CONSTRIANTS = int(N_SAMPLES*2)
            var = var/2
            ep_no_improvement = 0

        # -------------------------------------------------------
        # bad state correction 
        # -------------------------------------------------------

        state = env.reset()
        copied_env = deepcopy(env)
        
        state_action_rew_env = []
        lowest_rew = []

        for t in range(999): #one trajectory, can change here
            action = select_action(state, sample_policy, variance=0)
            next_state, reward, done, _ = env.step(action)            
            sample_policy.rewards.append((reward, t, "first"))

            if len(lowest_rew) < LOW_REW_SET or done:
                if done:
                    reward = 0
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

        print("finished first trajectory, bad states") 
        print(len(state_action_rew_env))

        # explore better actions
        low_rew_constraints_set = []

        for s, a, r, saved_env in state_action_rew_env:
            max_r, max_a = r, a
            for i in range(20): #sample 10 different actions
                step_env = deepcopy(saved_env)
                action_explore = select_action(s, sample_policy, variance=BAD_STATE_VAR, record=False)
                _, reward, done, _ = step_env.step(action_explore)
                if reward > max_r and not done:
                    max_r, max_a = reward, action_explore
            if max_r > r:
                low_rew_constraints_set.append((s, max_a, max_r, "bad_states"))
                print("improved bad state from %.3f to %.3f" %(r, max_r))
                print(a)
                print(max_a)
        # _________________________________________________________

        # Exploration
        num_steps = 0
        explore_episodes = 0
        explore_rew =0

        while num_steps < MAX_STEPS:
            state = env.reset()
            for t in range(1000): # Don't infinite loop while learning
                action = select_action(state, sample_policy, variance=var)
                name_str = "expl_var" #explore
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

        print('\nEpisode {}\tExplore reward: {:.2f}\tAverage ep len: {:.1f}\n'.format(i_episode, explore_rew, num_steps/explore_episodes))


        states, actions, rewards, info = calculate_rewards(explore_episodes, sample_policy)

        best_tuples = best_state_actions(states, actions, rewards, info, top_n_constraints=TOP_N_CONSTRIANTS)

        sample_policy.clean()

        # sample and solve
        
        max_policy, max_eval, max_set = sample_policy, sample_eval, best_tuples


        for branch in range(BRANCHES):
            
            branch_policy = Policy(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

            if len(low_rew_constraints_set) > N_SAMPLES/2:
                corrective_constraints = random.sample(low_rew_constraints_set, int(N_SAMPLES/2))
            else:
                corrective_constraints = low_rew_constraints_set
            
            constraints = random.sample(best_tuples, N_SAMPLES-len(corrective_constraints)) + corrective_constraints


            #constraints = random.sample(best_tuples, N_SAMPLES) + low_rew_constraints_set
            # Get metadata of constraints
            states, actions, rewards, info = zip(*constraints)
            print("ep %d b %d: %d constraints mean: %.3f  std: %.3f  max: %.3f" % ( i_episode, branch, len(constraints), np.mean(rewards), np.std(rewards), max(rewards)))
            print(info)
            branch_policy.train(torch.tensor(states).float().to(device),
                                torch.tensor(actions).float().to(device), epoch=args.training_epoch)

            # Evaluate
            num_steps = 0
            eval_episodes = 0
            eval_rew = 0
            while num_steps < 10000:
                state = env.reset()
                eval_sum = 0
                for t in range(10000): # Don't infinite loop while learning
                    action = select_action(state, branch_policy, variance=0)
                    state, reward, done, _ = env.step(action)
                    eval_rew += reward
                    branch_policy.rewards.append((reward, t, "%s_%d"%(name_str, explore_episodes)))

                    if args.render:
                        env.render()
                    if done:
                        break
                num_steps += (t-1)
                eval_episodes += 1
            eval_rew /= eval_episodes

            branch_policy.clean()


            #log
            print('Episode {}\tBranch: {}\tEval reward: {:.2f}\tExplore reward: {:.2f}'.format(
                i_episode, branch, eval_rew, explore_rew))
            logfile.write('Episode {}\tBranch: {}\tEval reward: {:.2f}\n'.format(i_episode, branch, eval_rew))

                    
            if eval_rew > max_eval:
                print("updated to this policy")
                print(max_policy.affine1)
                max_eval, max_policy, max_set = eval_rew, branch_policy, constraints
            
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
