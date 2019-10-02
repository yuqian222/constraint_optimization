import heapq
from multiprocessing import Pool
from functools import partial
import os
from tqdm import tqdm

import numpy as np

def get_elite_indicies(num_elite, rewards):
    return heapq.nlargest(num_elite, range(len(rewards)), rewards.take)


def run_cem(
        dynamics,
        state,
        epochs=6,
        batch_size=2000,
        elite_frac=0.2,
        num_process=4,
        horizon=10,
        epsilon=0.001,
        alpha=0.25
):
    num_elite = int(batch_size * elite_frac)
    lb, ub = dynamics.ac_lb, dynamics.ac_ub

    mean = np.tile((lb + ub) / 2, [horizon]) # dimention is ac_dim*horizon
    var = np.tile(np.square(ub - lb) / 16, [horizon])
    lb, ub = np.tile(lb, [horizon]), np.tile(ub, [horizon])
    
    for t in tqdm(range(epochs)):
        if np.max(var) <= epsilon:
            break

        lb_dist, ub_dist = mean - lb, ub - mean
        constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

        samples = np.random.multivariate_normal(
            mean=mean,
            cov=np.diag(np.array(constrained_var)),
            size=batch_size
        ).astype(np.float32)

        with Pool(num_process) as p:
            rewards = p.map(partial(dynamics.shoot_sequence, state=state), samples)

        rewards = np.array(rewards)

        indicies = get_elite_indicies(num_elite, rewards)
        elites = thetas[indicies]

        new_mean = np.mean(elites, axis=0)
        new_var = np.var(elites, axis=0)

        mean = alpha * mean + (1 - alpha) * new_mean
        var = alpha * var + (1 - alpha) * new_var

        t += 1
    
    return mean[:dynamics.acts_dim], mean 
