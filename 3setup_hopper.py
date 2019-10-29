import argparse, gym, copy, math, pickle, torch, random, json, os
import numpy as np
from itertools import count
from heapq import nlargest
from time import gmtime, strftime
from operator import itemgetter
import torch.nn as nn
import torch as F
from torch.utils.data import Dataset,DataLoader
from torch.distributions import Categorical, Bernoulli
from copy import deepcopy

from policies import *
from args import get_args
from cem import run_cem

from collections import OrderedDict, Counter

from replay.replay import Trained_model_wrapper
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
SAMPLE_TRAJ = 8
EVAL_TRAJ = 6
prev_X, prev_Y, prev_A = [],[],[]

def main():
    args = get_args()
    dir_name = "results/%s/%s-%s"%(args.env, "3setup", strftime("%m_%d_%H_%M", gmtime()))
    os.makedirs(dir_name, exist_ok=True)
    logfile = open(dir_name+"/log.txt", "w")

    with open(os.path.join(dir_name,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env = gym.make(args.env)
    #env = outer_env.wrapped_env

    num_hidden = args.hidden_size
    VARIANCE = args.var
    iter_steps = args.iter_steps
    
    # has to be neural network policy
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N_SAMPLES = args.n_samples if args.n_samples>0 else int(env.observation_space.shape[0]*4)
    LOW_REW_SET = int(N_SAMPLES*0.2)
    TOP_N_CONSTRIANTS = int(N_SAMPLES*1.5)
    
    def make_policy(mean, var):
        if mean is not None:
            mean = torch.Tensor(mean).to(device)
        if var is not None:
            var = torch.Tensor(var).to(device)
        return Policy_quad_norm(env.observation_space.shape[0],
                            env.action_space.shape[0],
                            num_hidden=num_hidden, 
                            mean=mean, 
                            var=var).to(device)

    print('Using device:', device)

    sample_policy, sample_eval = make_policy(None, None), -1700

    replay_buffer = Replay_buffer(args.gamma)

    dynamics = DynamicsEnsemble(args.env, num_models=3)

    ep_no_improvement = 0


    for i_episode in count(1):

        # hack
        if ep_no_improvement > 3:
            N_SAMPLES = int(N_SAMPLES * 1.2)
            TOP_N_CONSTRIANTS = int(N_SAMPLES*1.5) #-1
            LOW_REW_SET = int(LOW_REW_SET*1.2)
            iter_steps = TOP_N_CONSTRIANTS*2

            if VARIANCE>1e-4:
                VARIANCE = VARIANCE/1.2
                print("Updated Var to: %.3f"%(VARIANCE))
            ep_no_improvement = 0

        print("constraints: {}, to correct: {}".format(N_SAMPLES, TOP_N_CONSTRIANTS))
        # Exploration
        num_steps = 0
        explore_episodes = 0
        explore_rew =0
        state_action_rew = []
        lowest_rew = []

        while num_steps < iter_steps:
            state = env.reset()
            for t in range(1000): 
                action = sample_policy.select_action(state, VARIANCE)
                action = action.flatten()
                name_str = "expl_var" #explore
                next_state, reward, done, _ = env.step(action)
                explore_rew += reward

                replay_buffer.push((state,next_state,action, reward, done, (name_str, explore_episodes, t))) 

                if args.correct and i_episode>0:
                    if (args.env == "Hopper-v2" or args.env == "Walker2d-v2") and done:
                        reward = float('-inf')
                    if len(state_action_rew) < LOW_REW_SET:# or (args.env == "Hopper-v2" or args.env == "Walker2d-v2" and done):
                        state_action_rew.append([state,action,reward])
                        lowest_rew.append(reward)
                    elif reward < max(lowest_rew):
                        state_action_rew = sorted(state_action_rew, key=lambda l: l[2]) #sort by reward
                        state_action_rew[-1] = [state,action,reward]
                        lowest_rew.remove(max(lowest_rew))
                        lowest_rew.append(reward)

                if done:
                    break
                
                state = next_state
            
            num_steps += (t-1)
            explore_episodes += 1

        explore_rew /= explore_episodes
        print('\nEpisode {}\tExplore reward: {:.2f}\tAverage ep len: {:.1f}\n'.format(i_episode, explore_rew, num_steps/explore_episodes))

        # do corrections. 
        low_rew_constraints_set = []
        if args.correct and i_episode>1:
            print("exploring better actions", len(state_action_rew))
            #sample possible corrections
            for s, a, r in state_action_rew:
                max_a, _ = run_cem(dynamics, s, horizon=1)
                low_rew_constraints_set.append((s, max_a, "bad_states", 0, 0))

        # Train Dynamics
        X, Y, A, _, _, _ = replay_buffer.sample(-1)

        if i_episode!=1:
            print("Previous model evaluation:", dynamics.get_accuracy(X,Y,A))

        if len(X) <1500:
            X = np.concatenate([X, prev_X])
            X = X if len(X)<1500 else X[:1500]
            Y = np.concatenate([Y, prev_Y])
            Y = Y if len(Y)<1500 else Y[:1500]
            A = np.concatenate([A, prev_A])
            A = A if len(A)<1500 else A[:1500]

        dynamics.fit(X, Y, A, epoch=args.model_training_epoch)
        
        prev_X, prev_Y, prev_A =  X, Y, A

        best_tuples = replay_buffer.best_state_actions_replace(top_n_constraints=TOP_N_CONSTRIANTS, 
                                                               by='one_step', discard = True)

        mean, var = replay_buffer.get_mean_var()

        # support
        num_support = int(N_SAMPLES*0.7)
        support_states = np.random.uniform(low=-5, high=5, size=[num_support, 
                                                    env.observation_space.shape[0]])
        confidence = sorted([(x, dynamics.get_uncertainty(x, 
                                sample_policy.select_action(x, 0)[0])) 
                            for x in support_states],
                            key = lambda t: t[1])
        support_tuples = []
        print("confidence here:")
        print(confidence[:5])
        sliice = int(len(confidence)/2)
        for s, conf in confidence:
            if conf < 10: #arbitrary bound. for later
                max_a, _ = run_cem(dynamics, s)
                support_tuples.append((s, max_a, "model", 0, 0))
            else:
                a = sample_policy.select_action(s, 0)[0].tolist()
                support_tuples.append((s, a, "support", 0, 0))

        # sample and solve
        max_policy, max_eval, max_set = sample_policy, sample_eval, best_tuples
        branch_buffer = Replay_buffer(args.gamma)

        print(TOP_N_CONSTRIANTS)
        print(len(best_tuples))
        print(len(low_rew_constraints_set))
        
        for branch in range(args.branches):

            branch_policy = make_policy(None, None)
            branch_buffer = Replay_buffer(args.gamma)

            if N_SAMPLES >= len(best_tuples): 
                constraints = best_tuples
            else:   
                constraints = random.sample(best_tuples+support_tuples, N_SAMPLES)

            # Get metadata of constraints
            states, actions, info, rewards, _ = zip(*constraints)
            print("ep %d b %d: %d constraints mean: %.3f  std: %.3f  max: %.3f" % ( i_episode, branch, len(constraints), np.mean(rewards), np.std(rewards), max(rewards)))
            
            print(info)

            if isinstance(states[0], torch.Tensor):
                states = torch.cat(states)
            else:
                states = torch.Tensor(states)
            
            if isinstance(actions[0], torch.Tensor):
                actions = torch.cat(actions)
            else:
                actions = torch.Tensor(actions)

            branch_policy.train(states.to(device), actions.to(device), epoch=args.training_epoch)
           
            # Evaluate
            eval_rew = 0
            for i in range(EVAL_TRAJ):
                state, done = env.reset(), False
                step = 0
                while not done: # Don't infinite loop while learning
                    action = branch_policy.select_action(state,0)
                    action = action.flatten()
                    next_state, reward, done, _ = env.step(action)
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

        if i_episode>50:
            break



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



