import argparse, gym, copy, math, pickle, torch, random, json
import numpy as np
from itertools import count
from heapq import nlargest
from time import gmtime, strftime
from operator import itemgetter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.distributions import Categorical, Bernoulli

class Policy_lin(nn.Module):
    def __init__(self, num_inputs, num_outputs, var_bound=1.0, slack_bound=0.01, initialize = True):
        super(Policy_lin, self).__init__()
        self.var_bound = var_bound
        self.slack_bound = slack_bound

        self.affine1 = nn.Linear(num_inputs, num_outputs)

        if initialize:
            self.random_initialize()

        self.saved_action = []
        self.saved_state = []
        self.rewards = []

        self.optimizer = optim.RMSprop(self.parameters(),  lr=1e-3)
        self.criterion = nn.MSELoss()

    def random_initialize(self):
        nn.init.uniform_(self.affine1.weight.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.affine1.bias.data, 0.0)

    def init_weight(self, dic):
        for neuron_idx in range(self.affine1.weight.size(0)):
            self.affine1.bias.data[neuron_idx] = dic[("bias",neuron_idx)]
            for prev_neuron_idx in range(self.affine1.weight.size(1)):
                self.affine1.weight.data[neuron_idx][prev_neuron_idx] = dic[(neuron_idx,prev_neuron_idx)]

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        action = self.affine1(x)
        return action

    def updateParam(self, prob, print_results =False):
        result = []
        result_name = []
        print ("Update parameter")
        for v in prob.getVars():
            result.append(v.x)
            result_name.append(v.varName)
        if print_results:
            print(self.affine1.weight)
            print(result)

        indices = 0
        for neuron_idx in range(self.affine1.weight.size(0)):
            self.affine1.bias.data[neuron_idx] = result[indices]
            indices +=1
            for prev_neuron_idx in range(self.affine1.weight.size(1)):
                val = result[indices]
               
                self.affine1.weight.data[neuron_idx][prev_neuron_idx] = val
                indices += 1


    def initializeLimits(self, prob):
        firstParam = {}
        firstBias = [None]*self.affine1.bias.size(0)

        for neuron_idx in range(self.affine1.weight.size(0)):
            bias = prob.addVar(lb=-self.var_bound, ub=self.var_bound, vtype=GRB.CONTINUOUS, name="b" + str(neuron_idx))
            firstBias[neuron_idx] = bias
            for prev_neuron_idx in range(self.affine1.weight.size(1)): #4
                coeff = "x" + str(neuron_idx) +"_"+ str(prev_neuron_idx)
                var = prob.addVar(lb=-self.var_bound, ub=self.var_bound, vtype=GRB.CONTINUOUS, name=coeff)
                firstParam[(neuron_idx, prev_neuron_idx)] = var

        return firstParam, firstBias

    def solve(self, constraints_dict):
        
        prob = Model("mip1")
        firstParam, firstBias = self.initializeLimits(prob)

        formulas = []
        s_actions = 0
        count = 0
        slack_vars = []

        for state, action_dict in constraints_dict.items():
            action = list(action_dict)[0]
            exprs = []
            for neuron_idx in range(self.affine1.weight.size(0)): # 0, 1
                lin_expr = firstBias[neuron_idx]
                for prev_neuron_idx in range(0,self.affine1.weight.size(1)): # 4
                    var = firstParam[(neuron_idx, prev_neuron_idx)]
                    lin_expr = lin_expr + state[prev_neuron_idx]*var

                exprs.append(lin_expr)

            for index, exp in enumerate(exprs):
                slack = prob.addVar(lb=-self.slack_bound, ub=self.slack_bound, vtype=GRB.CONTINUOUS)
                slack_vars.append(slack)
                newexpr1 = (exp+slack == action[index])
                count += 1
                prob.addConstr(newexpr1)

        # objective that minimizes sum of slack variables
        obj = 0
        for e in slack_vars:
            obj += e*e 

        prob.setObjective(obj, GRB.MINIMIZE)

        print ("Number of constraints are ", count)

        if (count == 0):
            return (prob, 0)
        prob.optimize()
        return (prob, 1)

    def train(self, x, y, batches = 5, epoch = 3):
        training_set = value_dataset(x, y)
        training_generator = DataLoader(training_set,  batch_size=batches, shuffle=True)
        for epoch in range(epoch):
            running_loss = []
            for data in training_generator:
                pred = self.forward(data["x"]).squeeze()
                loss = self.criterion(pred, data["y"])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.item())
            if epoch % 100 == 0:
                print("Policy trianing: epoch %d, loss = %.3f" %(epoch, sum(running_loss)/len(running_loss)))


