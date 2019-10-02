import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


from policies.normalization import Normalization


class Dynamics(nn.Module):

    def __init__(self, env, num_hidden=100):

        super(Dynamics, self).__init__()
        self.env = env

        self.obs_dim = self.env.observation_space.shape[0]
        self.acts_dim = self.env.action_space.shape[0]
        
        self.affine1 = nn.Linear(self.obs_dim +  self.acts_dim, num_hidden)
        self.affine2 = nn.Linear(num_hidden, num_hidden)
        self.affine3 = nn.Linear(num_hidden, self.obs_dim)

        self.epsilon = 1e-10
        self.normlization = {}
        
        self.optimizer = optim.RMSprop(self.parameters())#, weight_decay=0.001)
        self.mse = nn.MSELoss()

    def set_normalization(self, norm):
        norm = {k: torch.Tensor(v) for (k,v) in norm.items()}
        self.normlization = norm

    def normal_to(self, device):
        self.normlization = {k: v.to(device) for (k,v) in self.normlization.items()}


    def forward(self, obs, acts):
        delta = self.forward_delta(obs, acts)
        return obs+delta

    def forward_delta(self, obs, acts):
        x = self.normalize_obs_acts(obs,acts)
        x = torch.relu(self.affine1(x))
        x = torch.relu(self.affine2(x))
        delta = self.affine3(x)
        delta = self.unnormalize_delta(delta)
        return delta

    def normalize_obs_acts(self, obs, acts):
        norm = self.normlization
        if norm['obs_mean'] is None:
            normalized_obs = obs
        else: 
            normalized_obs = (obs - norm['obs_mean']) / (norm['obs_std'] + self.epsilon)
        if norm['acts_mean'] is None:
            normalized_acts = acts
        else:
            normalized_acts = (acts - norm['acts_mean'])/ (norm['acts_std'] + self.epsilon)

        return torch.cat((normalized_obs, normalized_acts), 1)

    def unnormalize_delta(self, normalized_deltas):

        if self.normlization['delta_mean'] is None:
            return normalized_deltas

        return normalized_deltas * self.normlization['delta_std'] + self.normlization['delta_mean']

    def normalize_delta(self, deltas):
        
        if self.normlization['delta_mean'] is None:
            return deltas

        return (deltas - self.normlization['delta_mean'])/ (self.normlization['delta_std'] + self.epsilon)

    def train(self, training_generator, epoch = 300):
        for ep in range(epoch):
            running_loss = []
            for data in training_generator:
                x = data['x']
                y = data['y']
                a = data['a']
                loss = self.mse(self.forward_delta(x,a), y-x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.item())
            if ep % 50 == 0 or ep == epoch-1:
                print("Dynamics trianing: epoch %d, loss = %.3f" %(ep, sum(running_loss)/len(running_loss)))

    def get_accuracy(self, X, Y, A):

        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)
            Y = torch.Tensor(Y)
            A = torch.Tensor(A)
        loss = self.mse(self.forward_delta(X,A), Y-X)
        return loss.detach().cpu().numpy()

class DynamicsEnsemble(object):
    def __init__(self, env, num_models=1, num_hidden=100):
        self.env = gym.make(env)
        self.ts = 0
        self.models = []
        self.num_models = num_models
        self.init_dynamic_models(self.env, num_hidden)
        
        self.obs_dim = self.env.observation_space.shape[0]
        self.acts_dim = self.env.action_space.shape[0]
        
        self.ac_ub = self.env.action_space.high
        self.ac_lb = self.env.action_space.low

    def init_dynamic_models(self, env, num_hidden):
        for model_index in range(self.num_models):
            self.models.append(Dynamics(env, num_hidden=num_hidden))
        print("An ensemble of {} dynamics model initialized".format(self.num_models))
    
    def fit(self, X, Y, A, batch_size = 32, epoch = 100):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        training_set = dynamics_dataset(X,Y,A, device=device)
        training_generator = DataLoader(training_set,  batch_size=batch_size, shuffle=True)

        for i, model in enumerate(self.models):
            model.to(device)
            model.normal_to(device)
            print("fitting model %d in the emsemble" %i)
            model.train(training_generator, epoch=epoch)
            model.cpu()
            model.normal_to('cpu')


    def update_normalization(self, new_normalization):
        for model in self.models:
            model.set_normalization(new_normalization)

    def predict(self, x, a): # returns the mean among all models, can also chose a random one                                                                                                                                                                                                                       
        ys = []
        x = torch.Tensor(x)
        if len(x.shape) == 1:
            x = x[None,:] 
        a = torch.Tensor([a])
        for model in self.models:
            ys.append(model(x,a).detach().numpy())
        return np.array(ys).mean(axis=0)[0]


    def _generate_random_model_indices_for_prediction(self, k):
        model_indices = random.sample(range(self.num_models), k=k)
        #self.last_model_indices_used_for_prediction = model_indices
        return model_indices


    def step(self, action, state): #mostly from mbble-metrpo
        self.ts += 1
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        next_observation = self.predict(state, action)
        next_observation = np.clip(next_observation, self.env.observation_space.low, self.env.observation_space.high)
        next_observation = np.clip(next_observation, -1e5, 1e5)

        reward, done, d = self.get_rew(state,action) 
        return next_observation, reward, done, d

    def to(self, device):
        for model in self.models:
            model.to(device)

    def shoot_sequence(self, state, acts):
        horizon = int(acts.shape[0]/self.acts_dim)
        acts_ = acts.reshape([horizon, self.acts_dim])
        state_ = state

        rew = 0
        for i in range(horizon):
            next_state, reward, done, _ = self.step(acts_[i], state_)
            rew += reward
            state_ = next_state
            if done:
                break
        return rew


    def shoot(self, state, pi, length=10, num_branches=30):
        actions = []
        rews = []
        self.env.reset()

        for br in range(num_branches):
            state_ = state
            rew = 0
            for i in range(length):
                action = pi.select_action(state_, 0.5).flatten() #this can be debated
                if i == 0:
                    actions.append(action)
                next_state, reward, done, _ = self.step(action, state_)
                rew += reward
                state_ = next_state
                if done:
                    break
            rews.append(rew)
        best_act = actions[np.argmax(np.array(rews))]
        return best_act
    
    def get_rew(self, state, action):
        split = int(len(state)/2)
        qpos = np.concatenate([[0], state[:split]])
        qval = state[split:]
        self.env.set_state(qpos, qval)
        
        next_state, reward, done, d = self.env.step(action)
       
        return reward, done, d


    def evaluate(self, pi, num_traj = 3):
        eval_rew =0
        for i in range(num_traj):
            state, done = self.env.reset(), False
            for _ in range(999): # Don't infinite loop while learning
                action = pi.select_action(state,0).flatten()
                next_state, reward, done, d = self.step(action, state)
                
                if 'reward_run' in list(d.keys()):
                    if abs(d['reward_run']) > 10:
                        break
                eval_rew += reward
                state = next_state
                if done:
                    break
        eval_rew /= num_traj
        return eval_rew

    def get_accuracy(self, X, Y, A):
        return [ev.get_accuracy(X, Y, A) for ev in self.models]



class dynamics_dataset(Dataset):
    def __init__(self, x, y, a, device = 'cpu'):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)
            a = torch.Tensor(a).to(device)
        self.x = Variable(x)
        self.y = Variable(y)
        self.a = Variable(a)

        assert(len(x) == len(y))
        assert(len(y) == len(a))
    def  __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx], "a": self.a[idx]}