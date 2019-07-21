import argparse
import gym
import numpy as np
from itertools import count
import copy
import torch
import torch.nn as nn
import torch as F
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli
import matplotlib.pyplot as plt
import math
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

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 2)

        self.saved_action = []
        self.saved_state = []
        self.rewards = []

    def forward(self, x):
        action_scores = self.affine1(x)
        return F.softmax(action_scores, dim=1)
        # return F.sigmoid(action_scores)



policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state):

    new_state = torch.from_numpy(state).float().unsqueeze(0)

    probs = policy(new_state)

    # m = Bernoulli(probs)
    m = Categorical(probs)

    action = m.sample()
    numaction = action.data.numpy().astype(int)[0]

    chunk_state = (round(state[0], 2), round(state[1], 2),\
                   round(state[2], 2), round(state[3], 2))

    policy.saved_action.append(numaction)
    policy.saved_state.append(chunk_state)
    return numaction


def finish_episode(myround, my_states):
    R = 0
    rewards = []

    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)

    for i in range(len(policy.saved_state)):
        chunk_state = policy.saved_state[i]
        action = policy.saved_action[i]
        reward = rewards[i].item()

        if chunk_state in my_states:
            olddata = my_states[chunk_state]
            if action == 0:
                if olddata[0] == 0:
                    my_states[chunk_state] = (reward, olddata[1])
                else:
                    my_states[chunk_state] = ((olddata[0]+reward)/2, olddata[1])
            else:
                if olddata[1] == 0:
                    my_states[chunk_state] = (olddata[0], reward)
                else:
                    my_states[chunk_state] = (olddata[0], (olddata[1]+reward)/2)
        else:
            if action == 0:
                my_states[chunk_state] = (reward, 0)
            else:
                my_states[chunk_state] = (0, reward)


    del policy.saved_state[:]
    del policy.saved_action[:]
    del policy.rewards[:]


def filterStates(my_states):
    new_dict = {}
    for chunk_s in my_states:
        (action0, action1) = my_states[chunk_s]
        if action0 != 0 and action1 != 0:
            new_dict[chunk_s] = (action0, action1)
    return new_dict


def mergeStates(old_states, new_states):
    if (len(old_states) == 0):
        return new_states
    # only keep states with both actions
    old_states = filterStates(old_states)
    new_states = filterStates(new_states)

    result_states = old_states.copy()

    count = 0

    for new_s in new_states:
        similar = False
        conflicted_states = {}

        for s in old_states:

            norm = math.sqrt((s[0]-new_s[0])**2+(s[1]-new_s[1])**2+(s[2]-new_s[2])**2+(s[3]-new_s[3])**2)
            if (norm < 0.1):
                if ((old_states[s][0] > old_states[s][1] and new_states[new_s][0] > new_states[new_s][1])
                    or (old_states[s][0] < old_states[s][1] and new_states[new_s][0] < new_states[new_s][1])):
                    # print "no conflict"
                    continue
                else:
                    count += 1
                    conflicted_states[s] = old_states[s]

        #How to handle conflicted_states dictionary
        # no conflict with all existing states (large distance / same action)
        if len(conflicted_states) == 0:
            result_states[new_s] = new_states[new_s]

        # else:
        #     avg0 = 0
        #     avg1 = 0
        #     action0 = {}
        #     action1 = {}
        #     for s in conflicted_states:
        #         if (conflicted_states[s][0] > conflicted_states[s][1]):
        #             avg0 = (avg0 + conflicted_states[s][0])/2
        #             action0[s] = conflicted_states[s]
        #         else:
        #             avg1 = (avg1 + conflicted_states[s][1])/2
        #             action1[s] = conflicted_states[s]
        #
        #     num_action0 = len(action0)
        #     num_action1 = len(action1)
        #
        #
        #     if (num_action0 > num_action1 and new_states[new_s][0] > new_states[new_s][1]):
        #         for ss in action1:
        #             if ss in result_states:
        #                 del result_states[ss]
        #         result_states[new_s] = new_states[new_s]
        #     elif (num_action0 < num_action1 and new_states[new_s][0] < new_states[new_s][1]):
        #         for ss in action0:
        #             if ss in result_states:
        #                 del result_states[ss]
        #         result_states[new_s] = new_states[new_s]

    return result_states




def solveNetwork(my_states, limits, policy_net, firstParam, set, prob):
    hidden_size = 20
    currLimits = limits
    formulas = []
    s_actions = {}
    count = 0
    if (len(my_states) == 0):
        print("Empty state dictionary!!!")

    slack_vars = []
    selected_states = {}

    for chunk_s in my_states:

        input = chunk_s
        (action0, action1) = my_states[chunk_s]
        exprs = []
        for neuron_idx in range(policy_net.affine1.weight.size(0)): # 0, 1

            lin_expr = firstParam[(neuron_idx, "bias")]

            for prev_neuron_idx in range(0,policy_net.affine1.weight.size(1)): # 4
                var = firstParam[(neuron_idx, prev_neuron_idx)]
                lin_expr = lin_expr + input[prev_neuron_idx]*var

            exprs.append(lin_expr)

        if action0 != 0 and action1 != 0:

            if action0 - action1 > 0:
                val = action0 - action1
                selected_states[chunk_s] = (0, exprs, val)
            elif action1 - action0 > 0:
                val = action1 - action0
                selected_states[chunk_s] = (1, exprs, val)

    selected_states = sorted(selected_states.items(), key=lambda i: i[1][2], reverse=True)
    print ("length of selected_states is " + str(len(selected_states)))

    num_cons = 10
    if set == 1:
        num_cons = 10

    for i in range(min(num_cons, len(selected_states))):
        k, v = selected_states[i]
        exprs = v[1]
        val = v[2]
        print (val)
        if v[0] == 0:
            slack = prob.addVar(lb=0.0001, ub=0.1, vtype=GRB.CONTINUOUS, name="slack"+str(count))
            prob.addConstr(exprs[0] - exprs[1] >= slack)
        else:
            slack = prob.addVar(lb=0.0001, ub=0.1, vtype=GRB.CONTINUOUS, name="slack"+str(count))
            prob.addConstr(exprs[1] - exprs[0] >= slack)
        slack_vars.append(slack)
        count += 1


    obj = 0
    for i in slack_vars:
        obj += i
    prob.setObjective(obj, GRB.MAXIMIZE)


    print("Number of constraints are ", count)
    prob.optimize()
    prob.write("file.lp")
    # for v in prob.getVars():
    #     print(v.varName, v.x)

    # print('Obj:', prob.objVal)

    return (prob, s_actions)




def updateParam(prob, policy_net):
    result = []
    print("Update parameter")
    for v in prob.getVars():
        result.append(v.x)
        print(v.varName, v.x)

    indices = 0

    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        for prev_neuron_idx in range(policy_net.affine1.weight.size(1)):
            val = result[indices]
            policy_net.affine1.weight.data[neuron_idx][prev_neuron_idx] = val
            indices += 1


def initializeLimits(policy_net, limits, prob):
    firstParam = {}

    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        bias = prob.addVar(lb=-0.05, ub=0.05, vtype=GRB.CONTINUOUS, name="b" + str(neuron_idx))
        firstParam[(neuron_idx, "bias")] = bias

        for prev_neuron_idx in range(policy_net.affine1.weight.size(1)): #4
            coeff = "x" + str(neuron_idx) +"_"+ str(prev_neuron_idx)
            var = prob.addVar(lb=-10, ub=10, vtype=GRB.CONTINUOUS)
            firstParam[(neuron_idx, prev_neuron_idx)] = var

    return firstParam


def main():
    running_reward = 10
    myround = 0

    my_states = {}
    initLimits = []
    merged_s = {}
    prob = Model("mip1")

    firstParam = initializeLimits(policy, initLimits, prob)

    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode(myround, my_states)

        if myround > 0 and myround % 200 == 0:
            limits = []
            # print "before merging ", len(my_states)
            # my_states = mergeStates(merged_s, my_states)
            # print ("after merging ", len(my_states))

            if running_reward > 150:
                (prob, s_actions) = solveNetwork(my_states, limits, policy, firstParam, 1, prob)

            else:
                (prob, s_actions) = solveNetwork(my_states, limits, policy, firstParam, 0, prob)


            if prob.status == GRB.Status.OPTIMAL:
                print ("update Param using solved solution")
                updateParam(prob, policy)
            elif prob.status == GRB.Status.INF_OR_UNBD:

                print ("Infeasible or unbounded")
                break
            elif prob.status == GRB.Status.INFEASIBLE:
                print('Optimization was stopped with status %d' % prob.status)
                print("Infeasible!!!")
                break


            # merged_s = my_states
            my_states = {}

            prob = Model("mip1")
            firstParam = initializeLimits(policy, initLimits, prob)

        # if myround > 0 and myround % 1000 == 0:
        #     # merged_s = {}
        #     my_states = {}


        myround += 1

        if myround > 1 and myround % 10 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
