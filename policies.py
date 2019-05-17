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
from gurobipy import *

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

    def train_Q(self, states, Q, epoch = 3):
        for ep in range(epoch):
            actions = self(states)
            loss = -Q(torch.cat((states,actions), dim=1)).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("policy trianing: epoch %d, loss = %.3f" %(ep, loss.item()))

    def clean(self):
        del self.saved_state[:]
        del self.saved_action[:]
        del self.rewards[:]


class Policy_quad(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden=24, initialize = True):
        super(Policy_quad, self).__init__()

        self.affine1 = nn.Linear(num_inputs, num_hidden)
        self.affine2 = nn.Linear(num_hidden, num_outputs)

        if initialize:
            self.random_initialize()

        self.saved_action = []
        self.saved_state = []
        self.rewards = []

        self.optimizer = optim.RMSprop(self.parameters(),  lr=1e-3)
        self.criterion = nn.MSELoss()

    def random_initialize(self):
        for l in [self.affine1, self.affine2]:
            nn.init.uniform_(l.weight.data, a=-0.1, b=0.1)
            nn.init.uniform_(l.bias.data, 0.0)

    def init_weight(self, dic):
        for neuron_idx in range(self.affine1.weight.size(0)):
            self.affine1.bias.data[neuron_idx] = dic[("bias",neuron_idx)]
            for prev_neuron_idx in range(self.affine1.weight.size(1)):
                self.affine1.weight.data[neuron_idx][prev_neuron_idx] = dic[(neuron_idx,prev_neuron_idx)]
    
    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        action = torch.tanh(self.affine2(x))
        return action
    

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

    def train_Q(self, states, Q, epoch = 3):
        for ep in range(epoch):
            actions = self(states)
            loss = -Q(torch.cat((states,actions), dim=1)).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("policy trianing: epoch %d, loss = %.3f" %(ep, loss.item()))

    def clean(self):
        del self.saved_state[:]
        del self.saved_action[:]
        del self.rewards[:]

class value_dataset(Dataset):
    def __init__(self, x, y):
        self.x = Variable(x)
        self.y = Variable(y)
        assert(len(x) == len(y))
    def  __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}

class Value(nn.Module):
    def __init__(self, num_inputs, num_hidden=24):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, num_hidden)
        self.affine2 = nn.Linear(num_hidden, num_hidden)
        self.affine3 = nn.Linear(num_hidden, num_hidden)
        self.value_head = nn.Linear(num_hidden, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        self.optimizer = optim.RMSprop(self.parameters(), weight_decay=0.0005)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        x = torch.tanh(self.affine3(x))
        state_values = self.value_head(x)
        return state_values

    def train(self, x, y, batch_size = 5, epoch = 3):
        training_set = value_dataset(x, y)
        training_generator = DataLoader(training_set,  batch_size= batch_size, shuffle=True)
        for epoch in range(epoch):
            running_loss = 0
            for data in training_generator:
                pred = self.forward(data["x"]).squeeze()
                loss = self.criterion(pred, data["y"])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print("value trianing: epoch %d, ave. loss = %.3f" %(epoch, running_loss/len(training_generator)))

    def calculate_action_grad(self, state, action, rew_delta=0.01, step_size = 0.005): 
        #only make sense if this is used as q function
        if step_size == 0:
            return action
        action_var = torch.autograd.Variable(action, requires_grad=True)
        input_tensor = torch.cat((state, action_var))
        desired = self(input_tensor) + rew_delta
        desired.backward()
        grad_norm = torch.norm(action_var.grad).detach()
        step = step_size*(action_var.grad/grad_norm)
        if torch.isnan(step).any():
            return action
        return action + step

class Policy_Q(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, initialize = True):
        super(Policy_lin, self).__init__()

        self.policy = nn.Linear(num_inputs, num_outputs)

        self.Q1 = nn.Linear(num_inputs+num_outputs, num_hidden)
        self.Q2 = nn.Linear(num_hidden, num_hidden)
        self.Q_head = nn.Linear(num_hidden, 1)
        
        if initialize:
            nn.init.uniform_(self.affine1.weight.data, a=-0.1, b=0.1)
            nn.init.uniform_(self.affine1.bias.data, 0.0)
            self.Q_head.weight.data.mul_(0.1)
            self.Q_head.bias.data.mul_(0.0)


        self.saved_action = []
        self.saved_state = []
        self.rewards = []

        self.optimizer = optim.RMSprop(self.parameters(), weight_decay=0.01)
        self.criterion = nn.MSELoss()

    def forward(self, s):
        a = self.policy(s)
        return a 

    def forward_Q(self, s_a):

        q = F.relu(self.Q1(s_a))
        q = F.relu(self.Q2(q))
        q = self.Q_head(q)

        return q


    def train_Q(self, x, y, batch_size = 5, epoch = 3):
        training_set = value_dataset(x, y)
        training_generator = DataLoader(training_set,  batch_size=batch_size, shuffle=True)
        for epoch in range(epoch):
            running_loss = 0
            for data in training_generator:
                pred = self.forward_Q(data["x"]).squeeze()
                loss = self.criterion(pred, data["y"])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print("value trianing: epoch %d, loss = %.3f" %(epoch, running_loss))

    def train_policy(self, states, epoch = 5):
        for ep in range(epoch):
            actions = self(states)
            loss = -self.forward_Q(torch.cat((states,actions), dim=1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("policy trianing: epoch %d, loss = %.3f" %(ep, loss.item()))



# trash
class Policy_relu(nn.Module):
    def __init__(self, num_inputs, num_outputs,  var_bound, slack_bound, num_hidden=24, initialize = True):
        super(Policy_relu, self).__init__()

        self.var_bound = var_bound
        self.slack_bound = slack_bound

        self.affine1 = nn.Linear(num_inputs, num_hidden)
        self.affine2 = nn.Linear(num_hidden, num_outputs)
        self.layers = [self.affine1, self.affine2]

        if initialize:
            self.random_initialize()

        self.saved_action = []
        self.saved_state = []
        self.rewards = []

    def random_initialize(self):
        nn.init.uniform_(self.affine1.weight.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.affine1.bias.data, 0.0)
        nn.init.uniform_(self.affine2.weight.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.affine2.bias.data, 0.0)

    def forward(self, x):
        x = torch.relu(self.affine1(x))
        action = self.affine2(x)
        return action

    def updateParam(self, prob, print_results=False):
        result = []
        result_name = []
        print ("Update parameter")
        for v in prob.getVars():
            result.append(v.x)
            result_name.append(v.varName)

        indices = 0
        for layer in self.layers:
            for neuron_idx in range(layer.weight.size(0)):
                for prev_neuron_idx in range(layer.weight.size(1)):
                    layer.weight.data[neuron_idx][prev_neuron_idx] = result[indices]
                    indices += 1
            for idx in range(layer.bias.size(0)):
                layer.bias.data[idx] = result[indices]
                indices +=1

    def initializeLimits(self, prob):
        params = []
        biases = []

        for i, layer in enumerate(self.layers):
            param = {}
            bias  = []
            for neuron_idx in range(layer.weight.size(0)):
                for prev_neuron_idx in range(layer.weight.size(1)): #4
                    coeff = "%d_w_%d_%d" %(i, neuron_idx, prev_neuron_idx)
                    var = prob.addVar(lb=-self.var_bound, ub=self.var_bound, vtype=GRB.CONTINUOUS, name=coeff)
                    param[(neuron_idx, prev_neuron_idx)] = var
            for idx in range(layer.bias.size(0)):
                bi = prob.addVar(lb=-self.var_bound, ub=self.var_bound, vtype=GRB.CONTINUOUS, name="%d_b_%d"%(i,neuron_idx))
                bias.append(bi)

            params.append(param)
            biases.append(bias)

        x_pos = [] #variables for hidden state neuron values
        x_neg = []

        # the relu implementation references
        # Deep Neural Networks as 0-1 MILP: A Feasibility Study
        # https://arxiv.org/pdf/1712.06174.pdf
        for i, layer in enumerate(self.layers[:-1]):
            xi, si = [], []
            for idx in range(layer.weight.size(0)):
                xj = prob.addVar(lb=0, vtype=GRB.CONTINUOUS, name="%d_x_%d"%(i, idx)) #positive part
                sj = prob.addVar(lb=0, vtype=GRB.CONTINUOUS, name="%d_s_%d"%(i, idx)) #negative part
                '''
                # at least 1 of x and s is zero 
                zj = prob.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="%d_z_%d"%(i, idx))
                prob.addConstr((zj == 1) >> (xj <= 0))
                prob.addConstr((zj == 0) >> (sj <= 0))
                '''
                xi.append(xj)
                si.append(sj)
            
            x_pos.append(xi)
            x_neg.append(si)

        return params, biases, x_pos, x_neg

    def solve(self, constraints_dict):
        
        prob = Model("mip1")
        params, biases, x_pos, x_neg = self.initializeLimits(prob)

        formulas = []
        s_actions = 0
        count = 0
        slack_vars = []

        for state, action_dict in constraints_dict.items():
            action = list(action_dict)[0]
            
            input_vec = list(state)
            fianl_layer_exprs = []

            for i, layer in enumerate(self.layers):
                
                output=[]

                for neuron_idx in range(layer.weight.size(0)): # 0, 1
                    lin_expr = QuadExpr(biases[i][neuron_idx])
                    for prev_neuron_idx in range(0,layer.weight.size(1)): # 4
                        var = params[i][(neuron_idx, prev_neuron_idx)]
                        lin_expr = lin_expr + input_vec[prev_neuron_idx]*var

                    if i < len(x_pos):
                        prob.addConstr(lin_expr == x_pos[i][neuron_idx] )
                        prob.addConstr(x_neg[i][neuron_idx] == max_(x_pos[i][neuron_idx], 0))
                        output.append(x_neg[i][neuron_idx])
                    else:
                        fianl_layer_exprs.append(lin_expr)

                input_vec=output
           
            for index, exp in enumerate(fianl_layer_exprs):
                slack = prob.addVar(lb=-self.slack_bound, ub=self.slack_bound, vtype=GRB.CONTINUOUS)
                slack_vars.append(slack)
                prob.addConstr((exp - action[index] <= slack))
                prob.addConstr((action[index] - exp <= slack)) #workaround for quadratic equality constraint
                count += 1
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



