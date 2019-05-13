import argparse, gym, copy, math, pickle, torch
import numpy as np
from itertools import count
from heapq import nlargest
from time import gmtime, strftime
from operator import itemgetter
import torch.nn as nn
import torch as F
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli
from gurobipy import *


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
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
VAR_BOUND = 1.0
SLACK_BOUND = 0.05
TOP_N_CONSTRIANTS = 18
VARIANCE = 0.05
BRANCHES = 5
NOVELTY_SLACK = 50

env = gym.make('HalfCheetah-v2')
env.seed(args.seed)
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(17, 6)

        self.saved_action = []
        self.saved_state = []
        self.saved_log_probs = []
        self.rewards = []

    def init_weight(self, dic):

        for neuron_idx in range(self.affine1.weight.size(0)):
            self.affine1.bias.data[neuron_idx] = dic[("bias",neuron_idx)]
            for prev_neuron_idx in range(self.affine1.weight.size(1)):
                self.affine1.weight.data[neuron_idx][prev_neuron_idx] = dic[(neuron_idx,prev_neuron_idx)]

    def forward(self, x):
        # x = F.tanh(self.affine1(x))
        action_scores = self.affine1(x)
        return action_scores


def select_action(state, policy, variance=0.1, record=True):

    new_state = torch.from_numpy(state).unsqueeze(0)
    action = policy(new_state.float())
    action = action.data[0].numpy()
    action = np.random.normal(action, [variance]*len(action))
    chunk_state = []
    for i in range(len(state)):
        chunk_state.append(round(state[i], 5))

    chunk_state = tuple(chunk_state)
    if record:
        policy.saved_action.append(action)
        policy.saved_state.append(chunk_state)
    return action


def finish_episode(myround, policy, my_states):
    print ("finish_episode")
    R = 0
    rewards = []
    # calculate reward from the terminal state (add discount factor)
    '''
    for (r,step,name) in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, (R,step,name))
        if step == 0:
            R = 0
    '''
    rewards = policy.rewards
    # update the state_reward dictionary
    for i in range(len(policy.saved_state)):
        chunk_state = policy.saved_state[i]
        action = policy.saved_action[i]
        action = tuple(action)
        r,step,name = rewards[i]
        if chunk_state in my_states:
            if (action in my_states[chunk_state]):
                # print ("duplicate")
                my_states[chunk_state][action] = ((r+my_states[chunk_state][action][0])/2,step,name)
            else:
                my_states[chunk_state][action] = (r,step,name)
        else:
            my_states[chunk_state] = {}
            my_states[chunk_state][action] = (r,step,name)

    del policy.saved_state[:]
    del policy.saved_action[:]
    del policy.rewards[:]
    return my_states


    print("len of mystates is ")
    print(len(my_states))

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

    # Get metadata of constraints
    constraint_info = list(top_n_dict.values())
    vals = [list(v.values())[0][0] for v in constraint_info]
    print("constraint mean: %.3f  std: %.3f  max: %.3f" % (np.mean(vals), np.std(vals), max(vals)))
    print("constraint set's episode and step number:")
    print([list(v.values())[0][1:] for v in constraint_info])

    return top_n_dict


def solveNetwork(my_states, limits, policy_net, firstParam, firstBias, prob):
    currLimits = limits
    formulas = []
    s_actions = 0
    count = 0
    slack_vars = []
    
    # TODO 
    for state, action_dict in my_states.items():
        #print("constraint! state: %s  val: %.3f" % (state, max_vals_dict[state]))
        action = list(action_dict)[0]
        exprs = []
        for neuron_idx in range(policy_net.affine1.weight.size(0)): # 0, 1
            lin_expr = firstBias[neuron_idx]
            for prev_neuron_idx in range(0,policy_net.affine1.weight.size(1)): # 4
                # print neuron_idx + prev_neuron_idx*hidden_size
                var = firstParam[(neuron_idx, prev_neuron_idx)]
                lin_expr = lin_expr + state[prev_neuron_idx]*var

            exprs.append(lin_expr)
        index = 0
        for exp in exprs:
            slack = prob.addVar(lb=-SLACK_BOUND, ub=SLACK_BOUND, vtype=GRB.CONTINUOUS)
            slack_vars.append(slack)
            newexpr1 = (exp+slack == action[index])
            # newexpr2 = (exp >= action[index])
            count += 1
            prob.addConstr(newexpr1)
            index += 1

    # objective that minimizes sum of slack variables
    obj = 0
    for e in slack_vars:
        obj += e*e #abs value

    prob.setObjective(obj, GRB.MINIMIZE)

    print ("Number of constraints are ", count)

    if (count == 0):
        return (prob, 0)
    prob.optimize()
    prob.write("filewk.lp")
    return (prob, 1)


def updateParam(prob, policy_net):
    result = []
    result_name = []
    print ("Update parameter")
    for v in prob.getVars():
        result.append(v.x)
        result_name.append(v.varName)

    print(policy_net.affine1.weight)
    print(result)

    indices = 0
    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        policy_net.affine1.bias.data[neuron_idx] = result[indices]
        indices +=1
        for prev_neuron_idx in range(policy_net.affine1.weight.size(1)):
            val = result[indices]
           
            policy_net.affine1.weight.data[neuron_idx][prev_neuron_idx] = val
            indices += 1


def initializeLimits(policy_net, limits, prob):
    firstParam = {}
    firstBias = [None]*policy_net.affine1.bias.size(0)

    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        bias = prob.addVar(lb=-VAR_BOUND, ub=VAR_BOUND, vtype=GRB.CONTINUOUS, name="b" + str(neuron_idx))
        firstBias[neuron_idx] = bias
        for prev_neuron_idx in range(policy_net.affine1.weight.size(1)): #4
            coeff = "x" + str(neuron_idx) +"_"+ str(prev_neuron_idx)
            var = prob.addVar(lb=-VAR_BOUND, ub=VAR_BOUND, vtype=GRB.CONTINUOUS, name=coeff)
            firstParam[(neuron_idx, prev_neuron_idx)] = var

    return firstParam, firstBias

def solvePolicy(my_states, limits, policy, firstParam, firstBias, prob):
    print ("Length of mystate dictionary is" , len(my_states))

    my_states = bestStates(my_states, top_n_constraints=TOP_N_CONSTRIANTS) #only keep the best states
    (result, s_actions) = solveNetwork(my_states, limits, policy, firstParam, firstBias, prob)

    if s_actions == 0:
        print ("No valid constraint")
        return 1
    if prob.status == GRB.Status.OPTIMAL:
        print ("update Param using solved solution")
        return 0
    if prob.status == GRB.Status.INF_OR_UNBD:
        print ("Infeasible or unbounded")
        return 1
    if prob.status == GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % prob.status)
        print ("Infeasible!!!")
        return 1
    return 2 #what can happen here?

def main():
    global VARIANCE
    global SLACK_BOUND
    sample_policy, sample_eval = Policy(), float("-inf")
    
    if INIT_WEIGHT:
        with open("save_1000.p",'rb') as f:
            params=pickle.load(f)
            sample_policy.init_weight(params)

    my_policies = []
    initLimits = []
    timestr = strftime("%m_%d_%H_%M", gmtime())
    prob = Model("mip1")
    f = open("results/"+timestr+".txt", "w")

    (firstParam, firstBias) = initializeLimits(sample_policy, initLimits, prob)

    for i_episode in count(1):
        if i_episode > 50:
            VARIANCE = 0.05
            SLACK_BOUND = 0.1
        max_policy, max_eval = sample_policy, float("-inf")
        for branch in range(BRANCHES):

            # Exploration
            num_steps = 0
            explore_episodes = 0
            explore_rew =0
            branch_policy = Policy()
            branch_state_dict = {}
            while num_steps < 12000:
                state = env.reset()
                for t in range(10000): # Don't infinite loop while learning
                    if num_steps < 9000:
                        action = select_action(state, sample_policy, variance=VARIANCE)
                    else: 
                        action = select_action(state, sample_policy, variance=0) #randomly good

                    next_state, reward, done, _ = env.step(action)
                    explore_rew += reward
                    sample_policy.rewards.append((reward, t, "%d_expl_%d"%(i_episode, explore_episodes)))
                    if args.render:
                        env.render()
                    if done:
                        break
                    state = next_state
                num_steps += (t-1)
                explore_episodes += 1
            
            explore_rew /= explore_episodes
            branch_state_dict = finish_episode(explore_episodes, sample_policy, branch_state_dict)

            # Solve
            exit = solvePolicy(branch_state_dict, initLimits, branch_policy, firstParam, firstBias, prob)
            if exit == 0:
                updateParam(prob, branch_policy)
            elif exit == 2:
                print("Error: unhandled solvePolicy case here")
            prob = Model("mip1")
            (firstParam, firstBias) = initializeLimits(branch_policy, initLimits, prob)

            # Evaluate
            num_steps = 0
            eval_episodes = 0
            eval_rew = 0
            while num_steps < 10000:
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

            #log
            print('Episode {}\tBranch: {}\tEval reward: {:.2f}\tExplore reward: {:.2f}'.format(
                i_episode, branch, eval_rew, explore_rew))
            f.write('Episode {}\tBranch: {}\tEval reward: {:.2f}\tExplore reward: {:.2f}'.format(
                i_episode, branch, eval_rew, explore_rew))
            f.write('\n')

            if eval_rew > max_eval:
                max_eval, max_policy = eval_rew, branch_policy

        # the end of branching
        if max_eval > sample_eval - NOVELTY_SLACK:
            sample_policy, sample_eval = max_policy, max_eval


if __name__ == '__main__':
    main()
