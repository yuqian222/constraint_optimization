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
import pickle


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


env = gym.make('HalfCheetah-v2')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(17, 24)
        # self.affine2 = nn.Linear(24, 6)
        # self.affine2 = nn.Linear(20, 1)

        self.saved_action = []
        self.saved_state = []
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # x = F.tanh(self.affine1(x))
        action_scores = self.affine1(x)

        return action_scores



policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    new_state = torch.from_numpy(state).unsqueeze(0)
    action = policy(new_state.float())
    action = action.data[0].numpy()

    chunk_state = []
    for i in range(len(state)):
        chunk_state.append(round(state[i], 5))

    chunk_state = tuple(chunk_state)
    # print (chunk_state)

    policy.saved_action.append(action)
    policy.saved_state.append(chunk_state)
    return action


def finish_episode(myround, my_states):
    print ("finish_episode")
    #
    R = 0
    rewards = []
    # calculate reward from the terminal state (add discount factor)
    for (r,m) in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
        if m == 0:
            R = 0
    rewards = torch.tensor(rewards)
    print (rewards)
    # rewards = policy.rewards

    # rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    #
    # for log_prob, reward in zip(policy.saved_log_probs, rewards):
    #     policy_loss.append(-log_prob * reward)

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
#


def solveNetwork(my_states, limits, policy_net, firstParam, secondParam, set, prob, f):
    currLimits = limits
    formulas = []
    s_actions = 0
    count = 0
    slack_vars = []

    for chunk_s in my_states:

        dict = my_states[chunk_s]

        newdict = {}
        maxreward = float("-inf")
        maxaction = 0

        for elmt in dict:
            newdict[elmt] = dict[elmt][0]
            if newdict[elmt] > maxreward:
                maxreward = newdict[elmt]
                maxaction = elmt

        # print ("values are ", newdict.values())
        # maxval = max(newdict.values())
        # print (maxval)
        # keys = [k for k,v in newdict.items() if v==maxval]

        maxval = maxreward
        action = maxaction

        input = chunk_s
        # action = keys[0]
        # print (input)
        val = 20

        if set == 2:
            val = 65

        if (maxval > val):
            exprs = []
            for neuron_idx in range(policy_net.affine1.weight.size(0)): # 0, 1
                # print neuron_idx
                # print policy_net.affine1.bias.data
                # print policy_net.affine1.weight.data
                lin_expr = policy_net.affine1.bias.data[neuron_idx].item()

                for prev_neuron_idx in range(0,policy_net.affine1.weight.size(1)): # 4
                    # print neuron_idx + prev_neuron_idx*hidden_size
                    var = firstParam[(neuron_idx, prev_neuron_idx)]
                    lin_expr = lin_expr + input[prev_neuron_idx]*var

                exprs.append(lin_expr)
            index = 0
            for exp in exprs:
                slack = prob.addVar(lb=-0.1, ub=0.1, vtype=GRB.CONTINUOUS)
                slack_vars.append(slack)
                newexpr1 = (exp+slack == action[index])
                # newexpr2 = (exp >= action[index])
                count += 1
                prob.addConstr(newexpr1)
                index += 1

    # objective that minimizes sum of slack variables
    obj = 0
    for e in slack_vars:
        obj += e

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

    # print result
    indices = 0
    # print len(result)
    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        for prev_neuron_idx in range(policy_net.affine1.weight.size(1)):
            val = result[indices]
            # print (result_name[indices])
            # print ("nindex ", neuron_idx, " prev_nindex ", prev_neuron_idx)
            policy_net.affine1.weight.data[neuron_idx][prev_neuron_idx] = val
            indices += 1


def initializeLimits(policy_net, limits, prob):
    firstParam = {}
    secondParam = {}

    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        for prev_neuron_idx in range(policy_net.affine1.weight.size(1)): #4
            coeff = "x" + str(neuron_idx) +"_"+ str(prev_neuron_idx)
            var = prob.addVar(lb=-50, ub=50, vtype=GRB.CONTINUOUS, name=coeff)
            # print (prob.getVars())
            # prob.addConstr(var <= 500)
            # prob.addConstr(var >= -500)

            firstParam[(neuron_idx, prev_neuron_idx)] = var

    return firstParam, secondParam


def main():
    my_states = {}
    initLimits = []
    merged_s = {}
    prob = Model("mip1")
    f = open("result9.txt", "w")


    (firstParam, secondParam) = initializeLimits(policy, initLimits, prob)

    for i_episode in count(1):

        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        while num_steps < 25000:
            state = env.reset()

            reward_sum = 0
            for t in range(10000): # Don't infinite loop while learning
                # print (state[0:5])
                action = select_action(state)
                # print(" %.2f " % state[0], " %.2f " % state[1]," %.2f " % state[2])
                # print(format(state[0], "f"))
                action = action.data[0]
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
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
            num_episodes += 1
            reward_batch += reward_sum
        print ("Numsteps is ", num_steps)
        reward_batch /= num_episodes
        finish_episode(num_episodes, my_states)

        # if i_episode % args.log_interval == 0:
        if (1==1):
            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum, reward_batch))
            f.write('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum, reward_batch))
            f.write('\n')

        if i_episode == 100:
            f.close()
            break

        # if i_episode > 0 and i_episode % 2 == 0:
        #     pickle.dump( my_states, open( "save.p", "wb" ) )
        #     break


        if i_episode > 0 and i_episode % 3 == 0:
            print ("Length of mystate dictionary is" , len(my_states))
            # print len(initLimits)
            limits = initLimits

            if reward_batch > 50:
                (result, s_actions) = solveNetwork(my_states, limits, policy, firstParam, secondParam, 2, prob,f)
            else:
                (result, s_actions) = solveNetwork(my_states, limits, policy, firstParam, secondParam, 0, prob,f)

            if s_actions == 0:
                print ("No valid constraint")
                break

            if prob.status == GRB.Status.OPTIMAL:
                print ("update Param using solved solution")
                updateParam(prob, policy)
                prob = Model("mip1")
                (firstParam, secondParam) = initializeLimits(policy, initLimits, prob)


            elif prob.status == GRB.Status.INF_OR_UNBD:
                # prob.setParam(GRB.Param.Presolve, 0)
                # prob.optimize()
                print ("Infeasible or unbounded")
                break
            elif prob.status == GRB.Status.INFEASIBLE:
                print('Optimization was stopped with status %d' % prob.status)
                print ("Infeasible!!!")
                break

            merged_s = my_states
            my_states = {}

if __name__ == '__main__':
    main()
