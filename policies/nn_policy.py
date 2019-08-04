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


class Policy_quad(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden=24, initialize=True):
        super(Policy_quad, self).__init__()

        self.affine1 = nn.Linear(num_inputs, num_hidden)
        self.affine2 = nn.Linear(num_hidden, num_hidden)
        self.affine3 = nn.Linear(num_hidden, num_outputs)

        if initialize:
            self.random_initialize()

        self.saved_action = []
        self.saved_state = []
        self.rewards = []

        self.optimizer = optim.RMSprop(self.parameters())#, weight_decay=0.001)
        self.criterion = nn.MSELoss()

    def random_initialize(self):
        for l in [self.affine1, self.affine2, self.affine3]:
            torch.nn.init.uniform_(l.weight, -0.1, 0.1)
            nn.init.uniform_(l.bias.data, 0.0)

    def select_action(self, x, noise=0):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).unsqueeze(0).float().to(next(self.parameters()).device)
            
            a = self.forward(x)

            if noise != 0:
                noise = Variable(a.data.new(a.size()).normal_(0, noise))
                a = a + noise

        return a.data.cpu().numpy()
        

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        a = self.affine3(x)
        return a
    
    def train(self, x, y, epoch = 1):
        tol = torch.Tensor([5*1e-4])
        prev_loss = torch.Tensor([0])
        for e in range(epoch):
            pred = self.forward(x).squeeze()
            loss = self.criterion(pred, y)
            
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
                       
            if e % 500 == 0:
                print("Policy trianing: epoch %d, loss = %.3f" %(e, loss.item()))
            if torch.abs(prev_loss-loss) < tol:
                print("converged: epoch %d, loss = %.3f" %(e, loss.item()))
                return
            elif e == epoch-1:
                print("max iter: epoch %d, loss = %.3f" %(e, loss.item()))
                return
        prev_loss = loss

    def clean(self):
        del self.saved_state[:]
        del self.saved_action[:]
        del self.rewards[:]

    '''
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
            if epoch % 1000 == 0:
                print("Policy trianing: epoch %d, loss = %.3f" %(epoch, sum(running_loss)/len(running_loss)))
    '''


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