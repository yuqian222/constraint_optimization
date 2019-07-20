import os, sys, random
import numpy as np
from heapq import nlargest


class Replay_buffer():
    '''
    Expects tuples of (state, next_state, action, reward, done, info)
    '''
    def __init__(self, gamma, max_size=500000):
        
        self.max_size = max_size
        self.gamma = gamma
        self.ptr = 0

        self.storage = []
        self.culmulative_rewards = []
        self.td_error = []

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def clear(self):
        self.ptr=0
        self.storage=[]

    def sample(self, batch_size):
        if batch_size == -1:
            ind = range(len(self.storage))
        else:
            ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, a, r, d, info = [], [], [], [], [], []

        for i in ind:
            X, Y, A, R, D, I = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            a.append(np.array(A, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            info.append(np.array(I, copy=False))
        return np.array(x), np.array(y), np.array(a), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1).astype(int), info

    def calculate_rewards(self):
        R = 0
        rewards = []
        for x, y, a, r, d, info in reversed(self.storage):
            R = r + self.gamma * R * (1-d)
            rewards.insert(0, R)
        self.culmulative_rewards = rewards
        return rewards


    def best_state_actions(self, top_n_constraints=-1, by='td_error', discard = False):
        if top_n_constraints > 0:
            X, Y, A, R, D, I = zip(*self.storage)
            if by == 'td_error':
                candidate = zip(X, A, I, self.td_error, range(len(X)))
            else: # by rewards
                rew = self.calculate_rewards()
                candidate = zip(X, A, I, rew, range(len(X)))

            top_n = nlargest(top_n_constraints, candidate, key=lambda s: s[-2])
            if discard:
                ind = [x[-1] for x in top_n]
                new_storage = [self.storage[i] for i in ind]
                self.storage = new_storage
            return top_n
        else:
            return []

