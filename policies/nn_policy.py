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
    def __init__(self, num_inputs, num_outputs, num_hidden=24, discrete=False, initialize=True):
        super(Policy_quad, self).__init__()
        
        self.discrete = discrete

        self.affine1 = nn.Linear(num_inputs, num_hidden)
        self.affine2 = nn.Linear(num_hidden, num_hidden)
        self.affine3 = nn.Linear(num_hidden, num_outputs)

        if initialize:
            self.random_initialize()

        self.saved_action = []
        self.saved_state = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters())
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
                if self.discrete: #only for acrobot now
                    p = torch.Tensor([noise, noise, noise])
                    index = a+1
                    p[index.long()] = 1 - 2*noise
                    a = torch.multinomial(p, 1)[0] - 1
                else:
                    noise = Variable(a.data.new(a.size()).normal_(0, noise))
                    a = a + noise

        return a.data.cpu().numpy()
        

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        a = self.affine3(x)
        if self.discrete:
            a = torch.tanh(a).round()[0][0]
        return a
    
    def train(self, x, y, epoch = 1, tol=1e-4):
        tol = torch.Tensor([5*1e-4])
        prev_loss = torch.Tensor([0])
        for e in range(epoch):
            pred = self.forward(x).squeeze()
            loss = self.criterion(pred, y.float())
            
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
                       
            if e % 500 == 0:
                print("Policy trianing: epoch %d, loss = %.3f" %(e, loss.item()))
            if torch.abs(prev_loss - loss).item() < tol:
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

class value_dataset(Dataset):
    def __init__(self, x, y):
        self.x = Variable(x)
        self.y = Variable(y)
        assert(len(x) == len(y))
    def  __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}