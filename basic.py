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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

parser = argparse.ArgumentParser(description='cheetah_q parser')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--env', type=str, default='HalfCheetah-v2',
                    help='enviornment (default: HalfCheetah-v2)')

parser.add_argument('--policy', type=str, default="nn", #or can be nn
                    help='policy trained. can be or linear or nn')

parser.add_argument('--branches', type=int, default=5, metavar='N',
                    help='branches per round (default: 5)')
parser.add_argument('--iter_steps', type=int, default=10000, metavar='N',
                    help='num steps per iteration (default: 10,000)')

parser.add_argument('--var', type=float, default=0.05,
                    help='sample variance (default: 0.05)')
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
STEP_SIZE = 0.01
BRANCHES = args.branches
POLICY = args.policy
MAX_STEPS = args.iter_steps
HIDDEN_SIZE = args.hidden_size
LOW_REW_SET = 20
BAD_STATE_VAR = 0.3

# number of trajectories for evaluation
SAMPLE_TRAJ = 20
EVAL_TRAJ = 20


def main():

    dir_name = "results/%s/%s-%s"%(ENV, "basic", strftime("%m_%d_%H_%M", gmtime()))
    os.makedirs(dir_name, exist_ok=True)
    logfile = open(dir_name+"/log.txt", "w")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env = gym.make(ENV)

    num_hidden = HIDDEN_SIZE
    VARIANCE = args.var


    # just to make more robust for differnet envs
    if POLICY == "linear":
        N_SAMPLES = env.observation_space.shape[0]
        TOP_N_CONSTRIANTS = N_SAMPLES*2
        Policy = Policy_lin
    elif POLICY == "nn": #assume it's 2 layer here
        N_SAMPLES = int(env.observation_space.shape[0]*2) #(num_hidden) a little underdetermined
        LOW_REW_SET = N_SAMPLES*2
        TOP_N_CONSTRIANTS = int(N_SAMPLES*1.5)
        Policy = Policy_quad


    sample_policy, sample_eval = Policy(env.observation_space.shape[0],
                                        env.action_space.shape[0],
                                        num_hidden=num_hidden).to(device), -1700

    replay_buffer = Replay_buffer(args.gamma)

    if INIT_WEIGHT:
        sample_eval = 1300
        with open("save_1000.p",'rb') as f:
            params=pickle.load(f)
            sample_policy.init_weight(params)

    def make_policy():
        pi = Policy(env.observation_space.shape[0],
                    env.action_space.shape[0],
                    num_hidden=num_hidden).to(device)
        return pi

    ep_no_improvement = 0

    for i_episode in count(1):

        # hack
        if ep_no_improvement > 3:
            N_SAMPLES = int(N_SAMPLES * 1.5)
            TOP_N_CONSTRIANTS = int(N_SAMPLES*1.5)
            VARIANCE = VARIANCE/1.5
            print("Updated Var to: %.3f"%(sample_policy.noise))
            ep_no_improvement = 0


        # Exploration
        num_steps = 0
        explore_episodes = 0
        explore_rew =0

        while num_steps < MAX_STEPS:
            state = env.reset()

            state_action_rew_env = []
            lowest_rew = []

            for t in range(1000): 
                action = sample_policy.select_action(state, VARIANCE)
                name_str = "expl_var" #explore
                if num_steps < 200:
                    copied_env = deepcopy(env)
                next_state, reward, done, _ = env.step(action)
                explore_rew += reward

                replay_buffer.push((state,next_state,action, reward, done, (name_str, explore_episodes, t))) 
                
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
                low_rew_constraints_set.append((s, max_a, max_r, "bad_states"))
                print("improved bad state from %.3f to %.3f" %(r, max_r))
            if len(low_rew_constraints_set) > N_SAMPLES/3:
                break #enough bad correction constraints

        
        best_tuples = replay_buffer.best_state_actions(top_n_constraints=TOP_N_CONSTRIANTS, by='rewards', discard = True)

        # sample and solve

        max_policy, max_eval, max_set = sample_policy, sample_eval, best_tuples


        for branch in range(BRANCHES):

            branch_policy = make_policy()
            '''
            if len(low_rew_constraints_set) > N_SAMPLES/2:
                corrective_constraints = low_rew_constraints_set[:int(N_SAMPLES/2)]
            else:
                corrective_constraints = low_rew_constraints_set
            '''
            constraints = random.sample(best_tuples+low_rew_constraints_set, N_SAMPLES)
            print(all_l2_norm(constraints)[:5])

            # Get metadata of constraints
            states, actions, info, rewards, _ = zip(*constraints)
            print("ep %d b %d: %d constraints mean: %.3f  std: %.3f  max: %.3f" % ( i_episode, branch, len(constraints), np.mean(rewards), np.std(rewards), max(rewards)))
            print(info)
            if isinstance(states[0], torch.Tensor):
                branch_policy.train(torch.cat(states).to(device),
                                    torch.cat(actions).to(device), epoch=args.training_epoch)
            else:
                branch_policy.train(torch.tensor(states).float().to(device),
                                    torch.tensor(actions).float().to(device), epoch=args.training_epoch)
            # Evaluate
            eval_rew = 0
            for i in range(EVAL_TRAJ):
                state, done = env.reset(), False
                while not done: # Don't infinite loop while learning
                    action = branch_policy.select_action(state,0)
                    next_state, reward, done, _ = env.step(action)
                    eval_rew += reward

                    #replay_buffer.push((state,next_state, action, reward, done, ("b_"+str(branch), i))) 
                    state = next_state

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
    states, _, _, _, _ = zip(*constraints)
    if isinstance(states[0], torch.Tensor):
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