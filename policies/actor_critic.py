

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.distributions import Categorical, Bernoulli

from nn_policy import Policy_quad

class Actor_critic(nn.Module):
    def __init__(self, num_inputs, num_outputs, noise, expl_noise=0.3, num_hidden=24, initialize=True):
        super(Actor_critic, self).__init__()

        self.actor = Policy_quad(num_inputs, num_outputs, noise, expl_noise=expl_noise)
        self.value_loss_coef = 0
        self.entropy_coef = 0

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

