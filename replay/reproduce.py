'''
Test script to see how well we can reproduce a 
random given neural network
'''
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import gym, sys
sys.path.append('./replay')
from replay import *


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden=24, initialize=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.affine1 = nn.Linear(num_inputs, num_hidden)
        self.affine2 = nn.Linear(num_hidden, num_hidden)
        self.affine3 = nn.Linear(num_hidden, num_outputs)
        if initialize:
            self.random_initialize()

        self.optimizer = optim.RMSprop(self.parameters())
        self.criterion = nn.MSELoss()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
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
            
            if e % 500 == 0:
                print("Policy trianing: epoch %d, loss = %.3f" %(e, loss.item()))
            if loss.item() < 1e-3 or e == epoch-1:
                return loss.item()

def random_sample(network, dim, n, range_=10):
    x = np.random.rand(n, dim)*(range_*2) - range_ #input range [-5, 5]
    #x = torch.Tensor(x)
    y = network.select_action(x,0)
    return x, y

def random_sample_hopper(network, n):
    dim0 = np.random.rand(n, 1) * 1.6 + 0.3
    dim1 = np.random.rand(n, 1) * 0.3 - 0.1
    dim2 = np.random.rand(n, 1) * 1.4 - 1.2
    dim3 = np.random.rand(n, 1) * 2 -1
    dim4 = np.random.rand(n, 1) * 2 -1
    dim5 = np.random.rand(n, 1) * 5
    dim6 = np.random.rand(n, 1) * 8 - 6
    dim7 = np.random.rand(n, 1) * 4 - 2
    dim8 = np.random.rand(n, 1) * 16 - 8
    dim9 = np.random.rand(n, 1) * 20 - 10
    dim10 = np.random.rand(n, 1) * 20 -10
    x = np.column_stack((dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10))
    y = network.select_action(x, 0)
    return x, y


def eval(policy, env, n):
    # Evaluate
    eval_rew = []
    allstates = []
    for i in range(n):
        ep_rew = 0
        state, done = env.reset(), False
        step = 0
        while not done: # Don't infinite loop while learning
            action = policy(state).detach().numpy()
            next_state, reward, done, _ = env.step(action)
            allstates.append(next_state)
            ep_rew += reward
            state = next_state
            if done:
                eval_rew.append(ep_rew)
                ep_rew = 0
                break
    return eval_rew

def main():
    env = gym.make('Hopper-v2')
    target = Trained_model_wrapper("Hopper-v2", "./trained_models/", 567)
    print("built target")
    target.play(3) # run original
   
    x_test, y_test = random_sample_hopper(target, 200)

    for i in range(3,7):
        print("Experiment %d" % i)
        #print("exp %d"%i)
        x,y = random_sample_hopper(target, 10**i)
        learner = Net(env.observation_space.shape[0],
                env.action_space.shape[0], num_hidden=24)

        loss = learner.train(torch.Tensor(x), torch.Tensor(y), epoch = 50000)
        print("Sample size: 10^%d"%i,"Training loss: %.2f"%loss)

        r = eval(learner, env, 5)
        print(r)

        y_learner = learner(x_test)
        mse = ((y_test - y_learner.detach().numpy())**2).mean(axis=0)
        print(mse)

if __name__ == '__main__':
    main()