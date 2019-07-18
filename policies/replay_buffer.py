import os, sys, random
import numpy as np

class Replay_buffer():
    '''
    Expects tuples of (state, next_state, action, reward, done, info)
    '''
    def __init__(self, gamma, max_size=100000):
        
        self.max_size = max_size
        self.gamma = gamma
        self.ptr = 0

        self.storage = []
        self.culmulative_rewards = []
        self.advantage = []

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, a, r, d, info = [], [], [], [], [], []

        for i in ind:
            X, Y, A, R, D, I = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            a.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            info.append(np.array(Info, copy=False))


        return np.array(x), np.array(y), np.array(a), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), info

    def calculate_rewards(self):
        R = 0
        rewards = []
        for _, _, _, r, _, _ in self.storage:
            R = r + self.gamma * R * (1-D)
            rewards.insert(0, R)
        self.culmulative_rewards = rewards
        return rewards

    def calculate_advantage(self, critic): # how is this done?
        advantage = [] # use TD error to approximate
        for i, record in enumerate(self.storage):
            x, y, u, r, _, info = record
            delta = self.culmulative_rewards - r - self.gamma * critic(y)
            advantage.append(delta)
        self.advantage = advantage
        return advantage

    def best_state_actions(self, top_n_constraints=-1, by='advantage'):
        if top_n_constraints > 0:
            
            X, Y, A, R, D, I = zip(*self.storage)
            if by == 'advantage':
                self.calculate_advantage()
                candidate = zip(X, A, I, self.advantage)
            else: # by rewards
                candidate = zip(X, A, I, self.culmulative_rewards)

            top_n = nlargest(top_n_constraints, candidate, key=lambda s: s[-1])
            return top_n
        
        else:
            return []

