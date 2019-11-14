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


class Policy_quad_classification(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden=24, initialize=True):
        super(Policy_quad_classification, self).__init__()

        self.affine1 = nn.Linear(num_inputs, num_hidden)
        self.affine2 = nn.Linear(num_hidden, num_hidden)
        self.affine3 = nn.Linear(num_hidden, num_outputs)

        if initialize:
            self.random_initialize()

        self.saved_action = []
        self.saved_state = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def random_initialize(self):
        for l in [self.affine1, self.affine2, self.affine3]:
            torch.nn.init.uniform_(l.weight, -0.1, 0.1)
            nn.init.uniform_(l.bias.data, 0.0)

    def select_action(self, x, noise=0):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).unsqueeze(0).float().to(next(self.parameters()).device)
            
            a = self.forward(x)
            #print(a)
            if noise == 0:
                a = torch.max(a, dim=-1)[1]
            else:
                a = torch.multinomial(F.softmax(a, dim=0), 1)
        return a.data.cpu().numpy()
        
    def forward(self, x):
        x = torch.relu(self.affine1(x))
        x = torch.relu(self.affine2(x))
        a = self.affine3(x)
        return a
    
    def train(self, x, y, epoch = 1, tol=1e-5):
        tol = torch.Tensor([5*1e-4])
        prev_loss = torch.Tensor([0])
        for e in range(epoch):

            pred = self.forward(x)
            loss = self.criterion(pred, y.flatten())

            self.optimizer.zero_grad()
            loss.backward()#retain_graph=True)
            self.optimizer.step()
                       
            if e % 1000 == 0:
                print("Policy trianing: epoch %d, loss = %.3f" %(e, loss.item()))

            if loss.item() < tol:
                print("converged: epoch %d, loss = %.3f" %(e, loss.item()))
                return loss.item()
            elif e == epoch-1:
                print("max iter: epoch %d, loss = %.3f" %(e, loss.item()))
                return loss.item()
            
            prev_loss = loss
