import numpy as np
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

        self.obs_dim = env.observation_space.shape[0]
        self.acts_dim = env.action_space.shape[0]
        
        self.affine1 = nn.Linear(self.obs_dim +  self.acts_dim, num_hidden)
        self.affine2 = nn.Linear(num_hidden, num_hidden)
        self.affine3 = nn.Linear(num_hidden, self.obs_dim)

        self.epsilon = 1e-10
        self.normlization = Normalization({})
        
        self.optimizer = optim.RMSprop(self.parameters())#, weight_decay=0.001)
        self.mse = nn.MSELoss()

    def set_normalization(self, norm):
        norm = {k: torch.Tensor(v) for (k,v) in norm.items()}
        self.normlization = norm

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

    def train(self, X, Y, A, batch_size = 32, epoch = 300):
        training_set = dynamics_dataset(X,Y,A)
        training_generator = DataLoader(training_set,  batch_size=batch_size, shuffle=True)

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
            if ep % 100 == 0 or ep == epoch-1:
                print("Dynamics trianing: epoch %d, loss = %.3f" %(epoch, sum(running_loss)/len(running_loss)))

    def get_accuracy(self, X, Y, A):

        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)
            Y = torch.Tensor(Y)
            A = torch.Tensor(A)
        loss = self.mse(self.forward_delta(X,A), Y-X)
        return loss.detach().cpu().numpy()

class DynamicsEnsemble(object):
    def __init__(self, env, num_models=1, num_hidden=100):
        self.env = env
        self.ts = 0
        self.models = []
        self.num_models = num_models
        self.init_dynamic_models(env, num_hidden=num_hidden)

    def init_dynamic_models(self, env, num_hidden):
        for model_index in range(self.num_models):
            self.models.append(Dynamics(env, num_hidden=num_hidden))
        print("An ensemble of {} dynamics model initialized".format(self.num_models))
    
    def fit(self, X, Y, A, batch_size = 32, epoch = 100):
        for i, model in enumerate(self.models):
            print("fitting model %d in the emsemble" %i)
            model.train(X, Y, A, batch_size=batch_size, epoch=epoch)

    def update_normalization(self, new_normalization):
        for model in self.models:
            model.set_normalization(new_normalization)

    def predict(self, x, a): # returns the mean among all models, can also chose a random one                                                                                                                                                                                                                       
        ys = []
        x = torch.Tensor(x)
        if len(x.shape) == 1:
            x = x[None,:] 
        a =torch.Tensor([a])
        for model in self.models:
            ys.append(model(x,a).detach().numpy())
        return np.array(ys).mean(axis=0)


    def _generate_random_model_indices_for_prediction(self, k):
        model_indices = random.sample(range(self.num_models), k=k)
        #self.last_model_indices_used_for_prediction = model_indices
        return model_indices


    def step(self, actions, use_states=None): #mostly from mbble-metrpo
        self.ts += 1
        actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        next_observations = self.get_next_observation(actions, use_states=use_states)
        next_observations = np.clip(next_observations, self.env.observation_space.low, self.env.observation_space.high)
        next_observations = np.clip(next_observations, -1e5, 1e5)

        rewards = 1.0 - 1e-3 * np.square(actions).sum() #only survival
        self.states = next_observations
        s = self.states[0]

        dones = not (self.ts < 998 and
                     np.isfinite(s).all() and
                    (np.abs(s[1:]) < 100).all() and
                    (s[0] > .7) and (abs(s[1]) < .2)) 
        return self.states, rewards, dones, dict()


    def get_next_observation(self, actions, use_states=None):
        if use_states is not None:
            return self.predict(use_states, actions)
        return self.predict(self.states, actions)

    def evaluate(self, pi, num_traj = 3):
        eval_rew =0
        for i in range(num_traj):
            self.ts = 0
            state, done = self.env.reset(), False
            self.states = [state]
            while not done: # Don't infinite loop while learning
                self.ts += 1
                action = pi.select_action(state,0)
                action = action.flatten()
                next_state, reward, done, _ = self.step(action, use_states=state)
                eval_rew += reward
                state = next_state
                if done:
                    break
        eval_rew /= num_traj
        return eval_rew

    def estimate(self, start_state, pi, num_traj = 5):
        eval_rew =0
        for i in range(num_traj):
            self.ts = 0
            self.states = [start_state]
            state, done = start_state, False
            while not done: # Don't infinite loop while learning
                self.ts += 1
                action = pi.select_action(state, 0.05)
                action = action.flatten()
                next_state, reward, done, _ = self.step(action, use_states=state)
                eval_rew += reward
                state = next_state
                if done:
                    break
        eval_rew /= num_traj
        return eval_rew

    def get_accuracy(self, X, Y, A):
        return [ev.get_accuracy(X, Y, A) for ev in self.models]
    #def plan(self):




class dynamics_dataset(Dataset):
    def __init__(self, x, y, a):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            a = torch.Tensor(a)
        self.x = Variable(x)
        self.y = Variable(y)
        self.a = Variable(a)

        assert(len(x) == len(y))
        assert(len(y) == len(a))
    def  __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx], "a": self.a[idx]}