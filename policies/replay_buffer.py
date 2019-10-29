import os, sys, random, torch
import numpy as np
from heapq import nlargest, nsmallest
from policies.normalization import Normalization


class Replay_buffer():
    '''
    Expects tuples of (state, next_state, action, reward, done, info)
    '''
    def __init__(self, gamma, max_size=10000):
        
        self.max_size = max_size
        self.gamma = gamma
        self.ptr = 0

        self.storage = []
        self.culmulative_rewards = []
        self.td_error = []

        self.xmean = 0
        self.xvar = 0
        self.n = 0 

    def push(self, data):
        k = len(self.storage)
        if k == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
        allx = np.array([x[0] for x in self.storage])
    
    def get_mean_var(self):
        allx = np.array([x[0] for x in self.storage])
        self.xmean = np.mean(allx, axis=0) 
        self.xstd = np.std(allx, axis=0)
        return self.xmean, self.xstd
    
    def get_normalization(self):
        print("get normalization")
        X, Y, A, R, D, I = zip(*self.storage)
        norm = {}
        norm['obs_mean'] = np.mean(X, axis=0) 
        norm['obs_std'] = np.std(X, axis=0) 
        norm['acts_mean'] = np.mean(A, axis=0) 
        norm['acts_std'] = np.std(A, axis=0)
        delta = np.array(Y) - np.array(X)
        norm['delta_mean'] = np.mean(delta, axis=0) 
        norm['delta_std'] = np.std(delta, axis=0) 
        return norm


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


    def best_state_actions(self, top_n_constraints=-1, by='reward', discard = False):

        X, Y, A, R, D, I = zip(*self.storage)
        if by == 'td_error':
            candidates = zip(X, A, I, self.td_error, range(len(X)))
        elif by == 'one_step':
            candidates = zip(X, A, I, R, range(len(X)))
        else: # by rewards
            rew = self.calculate_rewards()
            candidates = zip(X, A, I, rew, range(len(X)))

        if top_n_constraints > 0:
            top_n = nlargest(top_n_constraints, candidates, key=lambda s: s[-2])
        else:
            top_n = list(candidates)

        if discard:
            ind = [x[-1] for x in top_n]
            new_storage = [self.storage[i] for i in ind]
            self.storage = new_storage
        return top_n


    def best_state_actions_replace(self, top_n_constraints=-1, by='reward', discard = False):
        X, Y, A, R, D, I = zip(*self.storage)
        if by == 'td_error':
            candidates = zip(X, A, I, self.td_error, range(len(X)))
        elif by == 'one_step':
            candidates = zip(X, A, I, R, range(len(X)))
        else: # by rewards
            rew = self.calculate_rewards()
            candidates = zip(X, A, I, rew, range(len(X)))

        if top_n_constraints > 0:
            top_n_candidate = nlargest(min([top_n_constraints*3, len(X)]), candidates, key=lambda s: s[-2]) #should just sort all, not sure if it's faster
            top_n = []
            top_n_states = []
            for tup in top_n_candidate:
                if no_0_dist(tup[0], top_n_states):
                    top_n.append(tup)
                    top_n_states.append(tup[0])
                    if len(top_n) > top_n_constraints:
                        break
        else:
            top_n = list(candidates)

        if discard:
            ind = [x[-1] for x in top_n]
            new_storage = [self.storage[i] for i in ind]
            self.storage = new_storage
        return top_n

    def low_rew_set(self, n):
        X, Y, A, R, D, I = zip(*self.storage)
        #assume reward is already calculated
        candidates = zip(X, A, self.culmulative_rewards)
        smallest_n_can = nsmallest(n*3, candidates, key=lambda s: s[-1])
        smallest_n = []
        smallest_n_states = []
        print('low rew set n', n)
        print("low res set", len(list(smallest_n_can)))
        for tup in smallest_n_can:
            if no_0_dist(tup[0], smallest_n_states):
                smallest_n.append(tup)
                smallest_n_states.append(tup[0])
                if len(smallest_n) > n:
                    break

        print("low res set after",len(smallest_n))

        return smallest_n







def no_0_dist(state, slist):
    if isinstance(state, torch.Tensor):
        for x in slist:
            d = (state - x).pow(2).sum()
            if d < 1e-4:
                return False
    else:
        for x in slist:
            d = np.linalg.norm(np.subtract(state,x))
            if d < 1e-2:
                return False
    return True

def all_l2_norm(constraints):
    states, _, _, _ ,_ = zip(*constraints)
    if isinstance(states[0], torch.Tensor):
        states = [s.cpu().numpy() for s in states]
    all_dist = []
    zerodist = 0
    for i, x1 in enumerate(states):
        for x2 in states[i+1:]:
            d=np.linalg.norm(np.subtract(x1,x2))
            if d - 0 < 1e-2:
                zerodist +=1
            all_dist.append(d)
    if zerodist > 0 :
        print("0 distances: %d" % zerodist)
    return sorted(all_dist)
