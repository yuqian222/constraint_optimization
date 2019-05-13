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

class Policy_value(nn.Module):
    def __init__(self, num_inputs, num_outputs, initialize = True):
        super(Policy_lin, self).__init__()

        self.policy = nn.Linear(num_inputs, num_outputs)
        self.value1 = nn.Linear(num_inputs, num_hidden)
        self.value2 = nn.Linear(num_hidden, num_hidden)
        self.value_head = nn.Linear(num_hidden, 1)
        
        if initialize:
            nn.init.uniform_(self.affine1.weight.data, a=-0.1, b=0.1)
            nn.init.uniform_(self.affine1.bias.data, 0.0)
            self.value_head.weight.data.mul_(0.1)
            self.value_head.bias.data.mul_(0.0)


        self.saved_action = []
        self.saved_state = []
        self.rewards = []

        self.optimizer = optim.RMSprop(self.parameters())
        self.criterion = nn.MSELoss()

    def init_weight(self, dic):
        for neuron_idx in range(self.policy.weight.size(0)):
            self.policy.bias.data[neuron_idx] = dic[("bias",neuron_idx)]
            for prev_neuron_idx in range(self.policy.weight.size(1)):
                self.policy.weight.data[neuron_idx][prev_neuron_idx] = dic[(neuron_idx,prev_neuron_idx)]
    
    def forward(self, x):
        action = self.policy(x)
        x = F.relu(self.value1(action))
        x = F.relu(self.value2(x))
        value = self.value_head(x)
        return action, value


    def train(self, x, y, batches = 5, epoch = 5):
        training_set = value_dataset(x, y)
        training_generator = DataLoader(training_set,  batch_size=batches, shuffle=True)
        for epoch in range(epoch):
            running_loss = 0
            for data in training_generator:
                pred = self.forward(data["x"]).squeeze()
                loss = self.criterion(pred, data["y"])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print("Policy trianing: epoch %d, loss = %.3f" %(epoch, running_loss))

    def clean(self):
        del self.saved_state[:]
        del self.saved_action[:]
        del self.rewards[:]


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
    def __init__(self, num_inputs, num_hidden=24):
        super(Value, self).__init__()
        self.policy = nn.Linear(num_inputs, num_hidden)
        self.affine2 = nn.Linear(num_hidden, num_hidden)
        self.value_head = nn.Linear(num_hidden, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        self.optimizer = optim.RMSprop(self.parameters())
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
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
            print("value trianing: epoch %d, loss = %.3f" %(epoch, running_loss/batch_size))

    def calculate_action_grad(self, state, action, rew_delta=0.1, step_size = 0.1): 
        #only make sense if this is used as q function
        action_var = torch.autograd.Variable(action, requires_grad=True)
        input_tensor = torch.cat((state, action_var))
        desired = self(input_tensor) + rew_delta
        desired.backward()
        return action+step_size*action_var.grad


