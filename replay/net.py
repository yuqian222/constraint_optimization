import torch, pickle
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from nn_policy import Policy_quad


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden=10, initialize=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.affine1 = nn.Linear(num_inputs, num_hidden)
        self.affine2 = nn.Linear(num_hidden, num_hidden)
        self.affine3 = nn.Linear(num_hidden, num_outputs)

        if initialize:
            self.random_initialize()

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        y = self.affine3(x)
        return y

    def random_initialize(self):
        for l in [self.affine1, self.affine2, self.affine3]:
            torch.nn.init.uniform_(l.weight, -1, 1)
            nn.init.uniform_(l.bias.data, 0.0)

    def train(self, x, y, epoch = 500):
        for e in range(epoch):
            pred = self.forward(x).squeeze()
            loss = self.criterion(pred, y)
            
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            if e % 100 == 0:
                print("Policy trianing: epoch %d, loss = %.3f" %(e, loss.item()))
            if e == epoch-1:
                return loss.item()

def random_sample(net, n):
    x = np.random.rand(n, 11)*20 - 10 #input range [-10, 10]
    x = torch.Tensor(x)
    y = net.forward(x)
    return x, y

def load_policy(path):
    model = Policy_quad(11,3, num_hidden=24)
    sd = pickle.load(open(path, "rb" ))
    #print(sd)
    model.load_state_dict(sd)
    return model

def main():

    path = "./trained_models/209_"
    constraints = pickle.load( open( path+"constraints.p", "rb" ))
    target = load_policy(path+"policy.p")
    for n in range(3,4): #10 to the n-th power
        x, y = random_sample(target,10**n)
        learner = Net(11, 3, num_hidden=24)

        loss = learner.train(x, y, epoch = 50000)

        test_x, test_y = target.random_sample(100)
        learner_y = learner(test_x)

        mse = ((test_y.detach().numpy() - learner_y.detach().numpy())**2).mean(axis=0)
        print("Sample size: 10^%d"%n,"Training loss: %.2f"%loss, mse)

if __name__ == '__main__':
    main()