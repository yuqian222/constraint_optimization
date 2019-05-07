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
from gurobipy import *


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
INIT_WEIGHT = True
CUMULATIVE = True
PRINT_RESULTS = False
VAR_BOUND = 1.0
SLACK_BOUND = 0.005
TOP_N_CONSTRIANTS = 50
N_SAMPLES = 18
VARIANCE = 0.01
BRANCHES = 20
NOVELTY_SLACK = 0

env = gym.make('HalfCheetah-v2')
env.seed(args.seed)
torch.manual_seed(args.seed)
num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, initialize = True):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, num_outputs)

        if initialize:
            self.random_initialize()

        self.saved_action = []
        self.saved_state = []
        self.saved_log_probs = []
        self.rewards = []


    def random_initialize(self):
        nn.init.uniform_(self.affine1.weight.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.affine1.bias.data, 0.0)

    def init_weight(self, dic):

        for neuron_idx in range(self.affine1.weight.size(0)):
            self.affine1.bias.data[neuron_idx] = dic[("bias",neuron_idx)]
            for prev_neuron_idx in range(self.affine1.weight.size(1)):
                self.affine1.weight.data[neuron_idx][prev_neuron_idx] = dic[(neuron_idx,prev_neuron_idx)]

    def forward(self, x):
        # x = F.tanh(self.affine1(x))
        action_scores = self.affine1(x)
        return action_scores

class value_dataset(Dataset):
    def __init__(self, x, y):
        self.x = Variable(torch.Tensor(x))
        self.y = Variable(torch.Tensor(y))
        assert(len(x) == len(y))
    def  __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}

class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 24)
        self.affine2 = nn.Linear(24, 24)
        self.value_head = nn.Linear(24, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        state_values = self.value_head(x)
        return state_values


def select_action(state, policy, variance=0.1, record=True):
    action = policy(state).data.numpy()
    action = np.random.normal(action, [variance]*len(action))
    return action



def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def create_state_dict(states, actions, rewards, info, my_states):
    # update the state_reward dictionary
    for i in range(len(states)):
        chunk_state = states[i]
        action = actions[i]
        r = rewards[i]
        step,name = info[i]
        if chunk_state in my_states:
            my_states[chunk_state][action] = (r,step,name)
        else:
            my_states[chunk_state] = {}
            my_states[chunk_state][action] = (r,step,name)
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


def solveNetwork(my_states,  policy_net, firstParam, firstBias, prob):
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
    if PRINT_RESULTS:
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


def initializeLimits(policy_net, prob):
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

def solvePolicy(constraints, policy, firstParam, firstBias, prob):
    print ("Length of mystate dictionary is" , len(constraints))

    (result, s_actions) = solveNetwork(constraints, policy, firstParam, firstBias, prob)

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
    return 2 

def sample(policy, value_net, optimizer, env, total_steps, var=0):
    num_steps, rew, rollouts, t = 0, 0, 0, 0
    state = env.reset()
    states    = [state]
    actions   = []
    values    = []
    rewards   = []
    masks     = []
    info      = []

    while num_steps < total_steps:
        state = torch.FloatTensor(state)
        value = value_net(state)

        if num_steps < 20000: #TODO
            action = select_action(state, policy, variance=var)
            name_str = "expl" #explore
        else: 
            action = select_action(state, policy, variance=0) # 20% randomly good
            name_str = "eval" #exploit

        next_state, reward, done, _ = env.step(action)

        states.append(next_state)
        actions.append(action)
        values.append(value)
        rewards.append(torch.FloatTensor([reward]).unsqueeze(1))
        masks.append(torch.FloatTensor([1 - done]).unsqueeze(1))
        info.append((t, "%s_%d"%(name_str, rollouts)))
        rew += reward
        rollouts += done
        num_steps += 1
        if done:
            t = 0
            rollouts += 1
            state = env.reset()
        else:
            t += 1
            state = next_state

    next_value = value_net(torch.FloatTensor(next_state))
    returns = compute_gae(next_value, rewards, masks, values)
    advantages = torch.cat(returns).detach() - torch.cat(values)

    # train value function  
    vloss = advantages.pow(2).mean()
    optimizer.zero_grad()
    vloss.backward()
    optimizer.step()

    total_reward = rew/rollouts
    return states, actions, advantages.numpy(), info, total_reward

def main():

    dir_name = "results/"+strftime("%m_%d_%H_%M", gmtime())
    os.makedirs(dir_name, exist_ok=True)
    logfile = open(dir_name+"/log.txt", "w")

    value_net = Value(num_inputs)
    optimizer = optim.RMSprop(value_net.parameters())

    sample_policy, sample_eval = Policy(num_inputs, num_outputs), float("-inf")

    if INIT_WEIGHT:
        sample_eval = 1300
        with open("save_1000.p",'rb') as f:
            params=pickle.load(f)
            sample_policy.init_weight(params)


    for i_episode in count(1):
        # Exploration
        my_states = {}
        
        states, actions, advantages, info, sample_rew = sample(sample_policy, 
                                                   value_net, 
                                                   optimizer, 
                                                   env, 
                                                   total_steps=25000, var=VARIANCE)

        print('\nEpisode {}\tSample reward: {:.2f}\n'.format(i_episode, sample_rew))
        logfile.write('Episode {}\tSample reward: {:.2f}'.format(i_episode, sample_rew))


        # create constraints
        my_states = create_state_dict(states, actions, advantages, info, my_states)
        constraints_dict = bestStates(my_states, top_n_constraints=TOP_N_CONSTRIANTS)
        
        max_policy, max_eval, max_set = sample_policy, sample_eval, constraints_dict

        for branch in range(BRANCHES):
            branch_policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
            constraints = dict(random.sample(constraints_dict.items(), N_SAMPLES))

            prob = Model("mip1")
            (firstParam, firstBias) = initializeLimits(branch_policy, prob)

            # Get metadata of constraints
            constraint_info = list(constraints.values())
            vals = [list(v.values())[0][0] for v in constraint_info]
            print("ep %d b %d: constraint mean: %.3f  std: %.3f  max: %.3f" % (i_episode, branch, np.mean(vals), np.std(vals), max(vals)))
            print("constraint set's episode and step number:")
            print([list(v.values())[0] for v in constraint_info])

            # Solve
            exit = solvePolicy(constraints, branch_policy, firstParam, firstBias, prob)
            
            if exit == 0:
                updateParam(prob, branch_policy)
            else:
                print('Episode {}\tBranch: {}\tUnsat'.format(i_episode, branch))
                print("L2_NORM")
                print(all_l2_norm(constraints))
                continue

            # Evaluate
            states, actions, advantages, info, eval_rew = sample(branch_policy, 
                                                               value_net, 
                                                               optimizer, 
                                                               env, 
                                                               total_steps=6000, var=0.0)

            #log
            print('Episode {}\tBranch: {}\tEval reward: {:.2f}\tSample reward: {:.2f}'.format(
                i_episode, branch, eval_rew, sample_rew))
            logfile.write('Episode {}\tBranch: {}\tEval reward: {:.2f}'.format(i_episode, branch, eval_rew))

            if eval_rew > max_eval:
                max_eval, max_policy, max_set = eval_rew, branch_policy, constraints

        # the end of branching
        if max_eval > sample_eval - NOVELTY_SLACK:
            with open("%s/%d.p"%(dir_name,i_episode), "wb") as f:
                pickle.dump({"all": constraints_dict, "constraints": max_set}, f)
            sample_policy, sample_eval = max_policy, max_eval

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
