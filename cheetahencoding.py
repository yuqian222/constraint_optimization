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
import matplotlib.pyplot as plt
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
INIT_WEIGHT = False
VAR_BOUND = 1.0
SLACK_BOUND = 0.5
TOP_N_CONSTRIANTS = 35
VARIANCE = 0.2


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

policy = Policy()

if INIT_WEIGHT:
    with open("save_1000.p",'rb') as f:
        params=pickle.load(f)
        policy.init_weight(params)


optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state, variance=0.1):

    new_state = torch.from_numpy(state).unsqueeze(0)
    action = policy(new_state.float())
    action = action.data[0].numpy()
    action = np.random.normal(action, [variance]*len(action))
    chunk_state = []
    for i in range(len(state)):
        chunk_state.append(round(state[i], 5))

    chunk_state = tuple(chunk_state)

    policy.saved_action.append(action)
    policy.saved_state.append(chunk_state)
    return action


def finish_episode(myround, my_states):
    print ("finish_episode")
    R = 0
    rewards = []
    # calculate reward from the terminal state (add discount factor)
    for (r,m) in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
        if m == 0:
            R = 0
    rewards = torch.tensor(rewards)

    # update the state_reward dictionary
    for i in range(len(policy.saved_state)):
        chunk_state = policy.saved_state[i]
        action = policy.saved_action[i]
        action = tuple(action)
        reward = rewards[i].item()
        if chunk_state in my_states:
            if (action in my_states[chunk_state]):
                # print ("duplicate")
                my_states[chunk_state][action] = ((reward+my_states[chunk_state][action][0])/2,1)
            else:
                my_states[chunk_state][action] = (reward,0)
        else:
            my_states[chunk_state] = {}
            my_states[chunk_state][action] = (reward,0)

    del policy.saved_state[:]
    del policy.saved_action[:]
    del policy.rewards[:]


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


def solveNetwork(my_states, limits, policy_net, firstParam, firstBias, prob, f):
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
    f.write('\nNumber of constraints are {}\t\n'.format(count))

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
    # print len(result)
    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        policy_net.affine1.bias.data[neuron_idx] = result[indices]
        indices +=1
        for prev_neuron_idx in range(policy_net.affine1.weight.size(1)):
            val = result[indices]
            # print (result_name[indices])
            # print ("nindex ", neuron_idx, " prev_nindex ", prev_neuron_idx)
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
            # print (prob.getVars())
            # prob.addConstr(var <= 500)
            # prob.addConstr(var >= -500)
            firstParam[(neuron_idx, prev_neuron_idx)] = var

    return firstParam, firstBias


def main():
    my_states = {}
    initLimits = []
    timestr = strftime("%m_%d_%H_%M", gmtime())
    prob = Model("mip1")
    f = open("results/"+timestr+".txt", "w")


    (firstParam, firstBias) = initializeLimits(policy, initLimits, prob)

    for i_episode in count(1):

        num_steps = 0
        eval_episodes = 0
        eval_rew = 0

        #Evaluation
        while num_steps < 15000:
            state = env.reset()
            eval_sum = 0
            for t in range(10000): # Don't infinite loop while learning
                action = select_action(state, variance=0)
                next_state, reward, done, _ = env.step(action)
                eval_rew += reward
                if done:
                    policy.rewards.append((reward, 0))
                else:
                    policy.rewards.append((reward, 1))
                if args.render:
                    env.render()
                if done:
                    break
                state = next_state
            num_steps += (t-1)
            eval_episodes += 1
        
        eval_rew /= eval_episodes

        num_steps = 0
        explore_episodes = 0
        explore_rew =0

        #exploration
        while num_steps < 15000:
            state = env.reset()
            for t in range(10000): # Don't infinite loop while learning
                action = select_action(state, variance=VARIANCE)

                next_state, reward, done, _ = env.step(action)
                explore_rew += reward
                if done:
                    policy.rewards.append((reward, 0))
                else:
                    policy.rewards.append((reward, 1))
                if args.render:
                    env.render()
                if done:
                    break
                state = next_state
            num_steps += (t-1)
            explore_episodes += 1
        
        explore_rew /= explore_episodes

        finish_episode(eval_episodes+explore_episodes, my_states)

        # if i_episode % args.log_interval == 0:
        if (1==1):
            print('Episode {}\tEval reward: {:.2f}\tExplore reward {:.2f}'.format(
                i_episode, eval_rew, explore_rew))
            f.write('Episode {}\tAverage epLen: {}\tAverage reward {:.2f}'.format(
                i_episode, eval_rew, explore_rew))
            f.write('\n')

        if i_episode == 100:
            f.close()
            break

        if i_episode > 0 and i_episode % 2 == 0:
            print ("Length of mystate dictionary is" , len(my_states))
            # print len(initLimits)
            limits = initLimits

            my_states = bestStates(my_states, top_n_constraints=TOP_N_CONSTRIANTS) #only keep the best states
            (result, s_actions) = solveNetwork(my_states, limits, policy, firstParam, firstBias, prob,f)

            if s_actions == 0:
                print ("No valid constraint")
                f.close()
                break

            if prob.status == GRB.Status.OPTIMAL:
                print ("update Param using solved solution")
                updateParam(prob, policy)
                prob = Model("mip1")
                (firstParam, firstBias) = initializeLimits(policy, initLimits, prob)


            elif prob.status == GRB.Status.INF_OR_UNBD:
                # prob.setParam(GRB.Param.Presolve, 0)
                # prob.optimize()
                print ("Infeasible or unbounded")
                break
            elif prob.status == GRB.Status.INFEASIBLE:
                print('Optimization was stopped with status %d' % prob.status)
                print ("Infeasible!!!")
                break

            #my_states = {}

if __name__ == '__main__':
    main()
