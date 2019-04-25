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
import random
import time


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
        self.affine1 = nn.Linear(17, 6)
        # self.affine2 = nn.Linear(24, 6)
        # self.affine2 = nn.Linear(20, 1)
        # self.affine1.weight.data.mul_(0.1)
        self.affine1.bias.data.mul_(0.0)
         # self.affine1.weight.data.mul_(0.3)
        torch.nn.init.uniform(self.affine1.weight.data, -0.1, 0.1)
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


def select_action(state, add, variance=0):
    new_state = torch.from_numpy(state).unsqueeze(0)

    if add == 1:
        # print ("call policy !!!!")
        action = policy(new_state.float())
        action = action.data[0].numpy()

        # action = np.random.multivariate_normal(action, np.identity(len(action)) * variance)
        action = np.random.normal(action, [variance]*len(action))

    chunk_state = []
    for i in range(len(state)):
        chunk_state.append(round(state[i], 4))

    chunk_state = tuple(chunk_state)
    # print (chunk_state)

    # if add == 0:
    #     action = []
    #     for i in range(6):
    #         action.append(round(random.uniform(-1, 1), 7))

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
        if m == 0:
            R = 0
        R = r + args.gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    # print (rewards)
    # rewards = policy.rewards

    # rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    #
    # for log_prob, reward in zip(policy.saved_log_probs, rewards):
    #     policy_loss.append(-log_prob * reward)
    # print ("length of my_states ", len(my_states))
    # update the state_reward dictionary
    for i in range(len(policy.saved_state)):
        chunk_state = policy.saved_state[i]
        action = policy.saved_action[i]
        action = tuple(action)
        reward = rewards[i].item()
        if chunk_state in my_states:
            if (action in my_states[chunk_state]):
                #print ("duplicate")
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



def sortState(my_states, threshold):
    pairs = []
    for s in my_states:
        for action in my_states[s]:
            if my_states[s][action][0] > threshold:
                pairs.append((s, action,my_states[s][action][0]))
    pairs.sort(key=lambda x: x[2])
    print (len(pairs))

    pairs = pairs[int(len(pairs)-30) : (len(pairs))]
    print (len(pairs))
    #
    # newpairs = []
    # # arr = []
    # for i in range(len(pairs)):
    #     s1 = pairs[i][0]
    #     similar = False
    #
    #     for j in range(len(pairs)):
    #         s2 = pairs[j][0]
    #         if s2 != s1:
    #             dist = 0
    #             for index in range(10):
    #                 # print (s1[index])
    #                 # print (s2[index])
    #                 dist += (s1[index] - s2[index])**2
    #                 # arr.append(round(dist, 4))
    #                 # print (dist)
    #             if dist < 0.75:
    #                 # print ("find similar " , i, " ", j )
    #                 similar = True
    #                 break
    #
    #     if similar == False:
    #         newpairs.append((s1, pairs[i][1], pairs[i][2]))
    #
    # print (len(newpairs))
    # # print (newpairs)
    # # arr.sort()
    # # print (arr)
    # # exit(0)
    return pairs


def solveNetwork2(my_states, limits, policy_net, firstParam, avgReward, set, prob, f, i_episode, threshold):
    currLimits = limits
    s_actions = 0
    count = 0
    slack_vars = []
    # statefile = open("stateFile2", "w")
    # rewardfile = open("rewardfile2", "w")
    pairs = sortState(my_states, threshold)

    for (input,action,maxval) in pairs:
        # statefile.write(str(list(input)))
        # statefile.write("\n")
        # rewardfile.write(str(maxval))
        # rewardfile.write('\n')
        exprs = []
        for neuron_idx in range(policy_net.affine1.weight.size(0)): # 0, 1
            lin_expr = firstParam[("bias", neuron_idx)]
            for prev_neuron_idx in range(policy_net.affine1.weight.size(1)): # 4
                var = firstParam[(neuron_idx, prev_neuron_idx)]
                lin_expr = lin_expr + input[prev_neuron_idx]*var
            exprs.append(lin_expr)

        index = 0
        for exp in exprs:
            slackname = "slack " + str(index)
            slack = prob.addVar(lb=-0.5, ub=0.5, vtype=GRB.CONTINUOUS)
            slack_vars.append(slack)
            newexpr1 = (exp+slack == action[index])
            # newexpr2 = (exp >= action[index])
            count += 1
            prob.addConstr(newexpr1)
            index += 1

    # objective that minimizes sum of slack variables
    obj = 0
    for e in slack_vars:
        obj += e*e

    prob.setObjective(obj, GRB.MINIMIZE)

    print ("Number of constraints are ", count)
    f.write('\nNumber of constraints are {}\t\n'.format(count))

    if (count == 0):
        return (prob, 0)

    prob.optimize()
    prob.write("filewk2.lp")
    return (prob, 1)



def print_weight(policy_net):
    for neuron_idx in range(policy_net.affine1.weight.size(0)): # 0, 1
        # print neuron_idx
        # print policy_net.affine1.bias.data
        # print policy_net.affine1.weight.data
        for prev_neuron_idx in range(0,policy_net.affine1.weight.size(1)): # 4
            print (policy_net.affine1.weight[neuron_idx][prev_neuron_idx])

# def solveNetwork(my_states, limits, policy_net, firstParam, avgReward, set, prob, f, i_episode, threshold):
#     currLimits = limits
#     formulas = []
#     s_actions = 0
#     count = 0
#     slack_vars = []
#     statefile = open("stateFile", "w")
#     rewardfile = open("rewardfile", "w")
#     filename = "reward" + str(i_episode)
#     writeReward(my_states,filename)
#
#
#     for chunk_s in my_states:
#         dict = my_states[chunk_s]
#
#         newdict = {}
#         maxreward = float("-inf")
#         maxaction = 0
#
#         for elmt in dict:
#             newdict[elmt] = dict[elmt][0]
#             if newdict[elmt] > maxreward:
#                 maxreward = newdict[elmt]
#                 maxaction = elmt
#
#         # if (len(dict) != 1):
#         #    print ("max reward is ", maxreward)
#         # else:
#         #    continue
#         # # If only count state with more than one possible actions, than there
#         # # will be so few valid constraints for properly encoding and solving
#         # if len(my_states) > 100000 and len(dict) == 1:
#         #     continue
#
#         # print ("values are ", newdict.values())
#         # maxval = max(newdict.values())
#         # print (maxval)
#         # keys = [k for k,v in newdict.items() if v==maxval]
#
#         maxval = maxreward
#         action = maxaction
#
#         input = chunk_s
#         # action = keys[0]
#         # print (input)
#         val = threshold
#
#         if set == 2:
#             val = 70
#
#         if (maxval > val):
#
#             statefile.write(str(list(chunk_s)))
#             statefile.write("\n")
#             rewardfile.write(str(maxval))
#             rewardfile.write('\n')
#             exprs = []
#             for neuron_idx in range(policy_net.affine1.weight.size(0)): # 0, 1
#                 # print neuron_idx
#                 # print policy_net.affine1.bias.data
#                 # print policy_net.affine1.weight.data
#                 # lin_expr = policy_net.affine1.bias.data[neuron_idx].item()
#                 # lin_expr2 = "input is " + str(policy_net.affine1.bias.data[neuron_idx].item())
#                 lin_expr = firstParam[("bias", neuron_idx)]
#                 # print ("weight is ", policy_net.affine1.weight[neuron_idx])
#                 for prev_neuron_idx in range(policy_net.affine1.weight.size(1)): # 4
#                     # print neuron_idx + prev_neuron_idx*hidden_size
#                     var = firstParam[(neuron_idx, prev_neuron_idx)]
#                     lin_expr = lin_expr + input[prev_neuron_idx]*var
#                     # print (input[prev_neuron_idx])
#                     # lin_expr2 = lin_expr2 + str(input[prev_neuron_idx]) + "  "
#                 exprs.append(lin_expr)
#
#             index = 0
#             for exp in exprs:
#                 slackname = "slack " + str(index)
#                 slack = prob.addVar(lb=-0.1, ub=0.1, vtype=GRB.CONTINUOUS)
#                 slack_vars.append(slack)
#                 newexpr1 = (exp+slack == action[index])
#                 # newexpr2 = (exp >= action[index])
#                 count += 1
#                 prob.addConstr(newexpr1)
#                 index += 1
#         # print ("for current state, count is ", count)
#
#     # objective that minimizes sum of slack variables
#     obj = 0
#     for e in slack_vars:
#         obj += e
#
#     prob.setObjective(obj, GRB.MINIMIZE)
#
#     print ("Number of constraints are ", count)
#     f.write('\nNumber of constraints are {}\t\n'.format(count))
#
#     if (count == 0):
#         return (prob, 0)
#
#     prob.optimize()
#     prob.write("filewk1.lp")
#     return (prob, 1)



def writeReward(my_states, filename):
    f1 = open(filename, "w")
    for state in my_states:
        for key in my_states[state]:
            f1.write(str(my_states[state][key][0]))
            f1.write('\n')
    f1.close()


def printParam(policy_net, result):
    # print result
    indices = 0
    # print len(result)
    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        for prev_neuron_idx in range(policy_net.affine1.weight.size(1)):
            val = result[indices]
            print ("original value is ", policy_net.affine1.weight.data[neuron_idx][prev_neuron_idx])
            print ("Update to ", val)
            # print (result_name[indices])
            # print ("nindex ", neuron_idx, " prev_nindex ", prev_neuron_idx)
            # policy_net.affine1.weight.data[neuron_idx][prev_neuron_idx] = val
            indices += 1


def updateParam(prob, policy_net):
    result = []
    result_name = []
    print ("Update parameter")
    for v in prob.getVars():
        result.append(v.x)
        result_name.append(v.varName)
        # print (v.x)
        # print (v.varName)
    printParam(policy_net, result)
    time.sleep(2)

    indices = 0

    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        for prev_neuron_idx in range(policy_net.affine1.weight.size(1)):
            val = result[indices]
            policy_net.affine1.weight.data[neuron_idx][prev_neuron_idx] = val
            indices += 1
    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        policy_net.affine1.bias.data[neuron_idx] = result[indices]
        indices += 1


def initializeLimits(policy_net, limits, prob):
    firstParam = {}

    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        for prev_neuron_idx in range(policy_net.affine1.weight.size(1)): #4
            coeff = "x" + str(neuron_idx) +"_"+ str(prev_neuron_idx)
            var = prob.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS, name=coeff)
            # print (prob.getVars())
            # prob.addConstr(var <= 500)
            # prob.addConstr(var >= -500)
            firstParam[(neuron_idx, prev_neuron_idx)] = var


    for neuron_idx in range(policy_net.affine1.weight.size(0)):
        bias = "bias" + str(neuron_idx)
        varbias = prob.addVar(lb=-0.5, ub=0.5, vtype=GRB.CONTINUOUS, name=bias)
        firstParam[("bias", neuron_idx)] = varbias

    return firstParam


def filterStates(my_states, threshold):
    if len(my_states) == 0:
        return my_states

    print ("before filter, length is ", len(my_states))
    new_mystates = {}
    for s in my_states:
        for a in my_states[s]:
            if my_states[s][a][0] > threshold:
                if not s in new_mystates:
                    new_mystates[s] = {}
                new_mystates[s][a] = my_states[s][a]

    print ("after filter, length is ", len(new_mystates))

    return new_mystates


def mergeStates(merged_s, my_states, f2, reward_batch):
    merged_s = filterStates(merged_s, 0)
    my_states = filterStates(my_states, 0)

    print ("Merge States")
    print ("length of merged_s is ", len(merged_s))
    print ("length of my_states is ", len(my_states))
    count = 0
    if len(merged_s) == 0:
        return my_states

    else:
        for new_s in my_states:
            # if state already exists in the original dictionary
            if new_s in merged_s:
                old_actions = merged_s[new_s]
                for action in my_states[new_s]:
                    if action in old_actions:
                        f2.write("For the same action, old reward is "+ str(old_actions[action][0])+"\n")
                        f2.write(" new reward is "+str(my_states[new_s][action][0])+"\n")
                    else:
                        merged_s[new_s][action] = my_states[new_s][action]
                count += 1
                # print ("Duplicate states")
            else:
                merged_s[new_s] = my_states[new_s]
        print ("Number of duplicated states comparing with original dict is ", count, "\n")
        f2.write("Number of duplicated states comparing with original dict is " + str(count)+"\n")
        # merged_s = filterStates(merged_s, 30)
        return merged_s






def main():
    my_states = {}
    initLimits = []
    merged_s = {}
    prob = Model("mip1")
    f = open("result14.txt", "w")
    f2 = open("mergeStateresult", "w")

    firstParam = initializeLimits(policy, initLimits, prob)
    # for neuron_idx in range(policy.affine1.weight.size(0)):
    #     for prev_neuron_idx in range(policy.affine1.weight.size(1)): #4
    #         print (policy.affine1.bias.data[neuron_idx])

    mydict = pickle.load( open( "save_1000.p", "rb" ) )
    for neuron_idx in range(policy.affine1.weight.size(0)):
        for prev_neuron_idx in range(policy.affine1.weight.size(1)):
            # print (mydict[(neuron_idx, prev_neuron_idx)])
            policy.affine1.weight.data[neuron_idx][prev_neuron_idx] = mydict[(neuron_idx, prev_neuron_idx)]

    for neuron_idx in range(policy.affine1.weight.size(0)):
            # print ("bias is ",mydict[("bias", neuron_idx)] )
            policy.affine1.bias.data[neuron_idx] = mydict[("bias", neuron_idx)]


    # mystate = pickle.load(open("savestate.p", "rb"))
    # my_states = []
    # for s in mystate:
    #     action = mystate[s]
    #     action = tuple(action)
    #     print (action)
    #

    for i_episode in count(1):

        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        while num_steps < 25000:
            state = env.reset()

            reward_sum = 0
            for t in range(10000): # Don't infinite loop while learning
                # print (state[0:5])
                # if i_episode > 1 and (i_episode % 16 == 0 or i_episode % 17 == 1):
                # if i_episode > 0 or (i_episode % 5 != 0):
                if (1 == 1):
                    action = select_action(state, 1)
                # else:
                #     action = select_action(state, 0)
                # print(" %.2f " % state[0], " %.2f " % state[1]," %.2f " % state[2])
                # print(format(state[0], "f"))
                # print (action)
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                # discount
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
            # print ("reward_sum is ", reward_sum)
            # print ("num_episodes is ", num_episodes)
            # print ("Numsteps is ", t-1)
        reward_batch /= num_episodes
        finish_episode(num_episodes, my_states)

        # if i_episode % args.log_interval == 0:
        if (1==1):
            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum, reward_batch))
            f.write('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum, reward_batch))
            f.write('\n')


        # if i_episode == 100:
        #     f.close()
        #     break

        # if i_episode > 0 and i_episode % 2 == 0:
        #     pickle.dump( my_states, open( "save.p", "wb" ) )
        #     break
        #
        # if 1 == 2:
        if i_episode > 0 and i_episode % 2 == 0:
            f2.write(("\n result for episode {}\n").format(i_episode))
            # merged_s = filterStates(merged_s, reward_batch+5)
            # my_states = filterStates(my_states, reward_batch+5)

            # my_states = mergeStates(merged_s, my_states, f2, 0)
            print ("Length of mystate dictionary is" , len(my_states))
            # print len(initLimits)
            limits = initLimits

            # if reward_batch > 50:
                # (result, s_actions) = solveNetwork(my_states, limits, policy, firstParam, reward_batch, 2, prob,f, i_episode,1000)
            # else
            (result, s_actions) = solveNetwork2(my_states, limits, policy, firstParam, reward_batch, 0, prob,f, i_episode,110)

            if s_actions == 0:
                print ("No valid constraint")
                f.close()
                break

            if prob.status == GRB.Status.OPTIMAL:
                print ("update Param using solved solution")
                updateParam(prob, policy)

                prob = Model("mip1")
                firstParam = initializeLimits(policy, initLimits, prob)
                # merged_s = my_states
                # my_states = {}


            elif prob.status == GRB.Status.INF_OR_UNBD:
                prob.setParam(GRB.Param.Presolve, 0)
                prob.optimize()
                print ("Infeasible or unbounded")
                break
            elif prob.status == GRB.Status.INFEASIBLE:
                print('Optimization was stopped with status %d' % prob.status)
                print ("Infeasible!!!")
                break

        if i_episode > 0 and i_episode % 80 == 0:
            # merged_s = my_states
            # prob = Model("mip1")
            # firstParam = initializeLimits(policy, initLimits, prob)
            my_states = {}

if __name__ == '__main__':
    main()
