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
SAMPLE_TRAJ = 5
EVAL_TRAJ = 10



def main():
    args = get_args()
    dir_name = "results/%s/%s-%s"%(args.env, "dynamics", strftime("%m_%d_%H_%M", gmtime()))
    os.makedirs(dir_name, exist_ok=True)
    logfile = open(dir_name+"/log.txt", "w")

    with open(os.path.join(dir_name,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env = gym.make(args.env)

    num_hidden = args.hidden_size
    VARIANCE = args.var
    iter_steps = args.iter_steps
    
    # has to be neural network policy
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N_SAMPLES = args.n_samples if args.n_samples>0 else int(env.observation_space.shape[0]*2)
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

    dynamics = DynamicsEnsemble(env, num_models=1)

    ep_no_improvement = 0

    for i_episode in count(1):

        # hack
        if ep_no_improvement > 3 :
            N_SAMPLES = int(N_SAMPLES * 1.2)
            TOP_N_CONSTRIANTS = int(N_SAMPLES*1.5) #-1
            LOW_REW_SET = int(LOW_REW_SET*1.2)

            if TOP_N_CONSTRIANTS > iter_steps:
                iter_steps = TOP_N_CONSTRIANTS*1.5

            if VARIANCE>1e-4:
                VARIANCE = VARIANCE/1.2
                print("Updated Var to: %.3f"%(VARIANCE))
            ep_no_improvement = 0

        # Exploration
        num_steps = 0
        explore_episodes = 0
        explore_rew =0

        while num_steps < args.iter_steps:
            state = env.reset()

            state_action_rew = []
            lowest_rew = []

            for t in range(1000): 
                action = sample_policy.select_action(state, VARIANCE)
                action = action.flatten()
                name_str = "expl_var" #explore
                next_state, reward, done, _ = env.step(action)
                explore_rew += reward

                replay_buffer.push((state,next_state,action, reward, done, (name_str, explore_episodes, t))) 

                if args.correct and i_episode>1:
                    if (args.env == "Hopper-v2" or args.env == "Walker2d-v2") and done:
                        reward = float('-inf')
                    if len(lowest_rew) < LOW_REW_SET or (args.env == "Hopper-v2" or args.env == "Walker2d-v2" and done):
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

        # do corrections 
        if args.correct and i_episode>1:
            print("exploring better actions")
            low_rew_constraints_set = []

            #sample possible corrections
            for s, a, r in state_action_rew:
                max_r, max_a = r, a
                for i in range(30): #sample 20 different actions
                    action_explore = sample_policy.select_action(s, BAD_STATE_VAR)
                    action = action.flatten()
                    next_state, reward, done, _ = dynamics.step(action, use_states=state)
                    est = dynamics.estimate(next_state, sample_policy)
                    if est > max_r and not done:
                        max_r, max_a = reward, action_explore
                low_rew_constraints_set.append((s, max_a, "bad_states", 0, 0))
        else:
            low_rew_constraints_set = []

        # Train Dynamics
        X, Y, A, _, _, _ = replay_buffer.sample(-1)

        if i_episode!=1:
            print("Previous model evaluation:", dynamics.get_accuracy(X,Y,A))

        dynamics.update_normalization(replay_buffer.get_normalization())
        dynamics.fit(X, Y, A, epoch=10)
        print("dynamics eval:", dynamics.evaluate(sample_policy))
            
        best_tuples = replay_buffer.best_state_actions(top_n_constraints=TOP_N_CONSTRIANTS, by='rewards', discard = True)
        mean, var = replay_buffer.get_mean_var()
        print(mean)
        print(var)


        # sample and solve
        max_policy, max_eval, max_set = sample_policy, sample_eval, best_tuples
        branch_buffer = Replay_buffer(args.gamma)

        for branch in range(args.branches):

            branch_policy = make_policy(mean, var)

            print(len(best_tuples))
            print(N_SAMPLES)
            constraints = random.sample(best_tuples, N_SAMPLES) + low_rew_constraints_set
            #print(all_l2_norm(constraints)[:5])

            # Get metadata of constraints
            states, actions, info, rewards, _ = zip(*constraints)
            print("ep %d b %d: %d constraints mean: %.3f  std: %.3f  max: %.3f" % ( i_episode, branch, len(constraints), np.mean(rewards), np.std(rewards), max(rewards)))
            
            print(len(info))

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
            print("dynamics eval:", dynamics.evaluate(branch_policy))
            logfile.write('Episode {}\tBranch: {}\tConstraints:{}\tEval reward: {:.2f}\n'.format(i_episode, branch, len(constraints), eval_rew))
            logfile.write("dynamics eval: {}".format( dynamics.evaluate(branch_policy)))
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



def get_env(env_name):
    from envs.gym import env_name_to_gym_registry
    from envs.proxy_env import ProxyEnv
    unnormalized_env = gym.make(env_name_to_gym_registry[env_name])
    
    import builtins
    builtins.visualize = False

    return ProxyEnv(unnormalized_env)
