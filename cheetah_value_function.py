import argparse, gym, copy, math, pickle, torch, random, json
import numpy as np
from itertools import count
from heapq import nlargest
from time import gmtime, strftime
from operator import itemgetter
import torch.nn as nn
import torch as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.distributions import Categorical, Bernoulli
from gurobipy import *


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

#GLOBAL VARIABLES
INIT_WEIGHT = False
CUMULATIVE = True
PRINT_RESULTS = False
VAR_BOUND = 1.0
SLACK_BOUND = 0.005
TOP_N_CONSTRIANTS = 50
N_SAMPLES = 18
VARIANCE = 0.01
BRANCHES = 20
NOVELTY_SLACK = 0

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, initialize = True):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, num_outputs)

        if initialize:
            self.random_initialize()

        self.saved_action = []
        self.saved_state = []
        self.saved_log_probs = []
        self.rewards = []


    def random_initialize(self):
        nn.init.uniform_(self.affine1.weight.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.affine1.bias.data, 0.0)

    def init_weight(self, dic):
        for neuron_idx in range(self.affine1.weight.size(0)):
            self.affine1.bias.data[neuron_idx] = dic[("bias",neuron_idx)]
            for prev_neuron_idx in range(self.affine1.weight.size(1)):
                self.affine1.weight.data[neuron_idx][prev_neuron_idx] = dic[(neuron_idx,prev_neuron_idx)]
    def forward(self, x):
        # x = F.tanh(self.affine1(x))
        action_scores = self.affine1(x)
        return action_scores


class value_dataset(Dataset):
    def __init__(self, x, y):
        self.x = Variable(torch.Tensor(x))
        self.y = Variable(torch.Tensor(y))
        assert(len(x) == len(y))
    def  __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}

class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 24)
        self.affine2 = nn.Linear(24, 24)
        self.value_head = nn.Linear(24, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        self.optimizer = optim.RMSprop(self.parameters())
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        state_values = self.value_head(x)
        return state_values

    def train(self, x, y, batch_size = 5, epoch = 3):
        training_set = value_dataset(x, y)
        training_generator = DataLoader(training_set,  batch_size= batch_size, shuffle=True)
        for epoch in range(epoch):
            running_loss = 0
            for data in training_generator:
                pred = self.forward(data["x"]).squeeze()
                loss = self.criterion(pred, data["y"])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print("value trianing: epoch %d, loss = %.3f" %(epoch, running_loss/batch_size))


def select_action(state, policy, variance=0.1, record=True):

    new_state = torch.from_numpy(state).unsqueeze(0)
    action = policy(new_state.float())
    action = action.data[0].numpy()
    action = np.random.normal(action, [variance]*len(action))

    if record:
        policy.saved_action.append(tuple(action))
        policy.saved_state.append(tuple(state))
    return action



def calculate_rewards(myround, policy):
    R = 0
    rewards = []
    info = []
    if CUMULATIVE:
        for (r,step,name) in policy.rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
            info.insert(0, (step,name))
            if step == 0:
                R = 0
    else:
        rewards = policy.rewards
    return policy.saved_state, rewards, info


def create_state_dict(policy, rewards, info, my_states):
    # update the state_reward dictionary
    for i in range(len(policy.saved_state)):
        chunk_state = policy.saved_state[i]
        action = policy.saved_action[i]
        r = rewards[i]
        step,name = info[i]
        if chunk_state in my_states:
            print("duplicate state")
            if (action in my_states[chunk_state]):
                print ("duplicate action")
                my_states[chunk_state][action] = ((r+my_states[chunk_state][action][0])/2,step,name)
            else:
                my_states[chunk_state][action] = (r,step,name)
        else:
            my_states[chunk_state] = {}
            my_states[chunk_state][action] = (r,step,name)

    del policy.saved_state[:]
    del policy.saved_action[:]
    del policy.rewards[:]
    return my_states



def bestStates(my_states, top_n_constraints=-1): 
    max_act_dict = {}
    max_vals_dict = {}

    for chunk_s in my_states:
        max_action = max(my_states[chunk_s].items(), key=operator.itemgetter(1))[0]
        max_act_dict[chunk_s] = max_action
        max_vals_dict[chunk_s] = my_states[chunk_s][max_action][0]

    # Get metadata of values
    vals = list(max_vals_dict.values())

    print("Max values mean: %.3f  std: %.3f  max: %.3f" % (np.mean(vals), np.std(vals), max(vals)))

    if top_n_constraints > 0:
        top_n = nlargest(top_n_constraints, max_act_dict.keys(), key=lambda s: max_vals_dict[s])
        top_n_dict = {k: v for k, v in my_states.items() if k in top_n}
    else:
        top_n_dict= my_states

    return top_n_dict





def solvePolicy(constraints, policy, firstParam, firstBias, prob):
    print ("Length of mystate dictionary is" , len(constraints))

    (result, s_actions) = solveNetwork(constraints, policy, firstParam, firstBias, prob)

    if s_actions == 0:
        print ("No valid constraint")
        return 1
    if prob.status == GRB.Status.OPTIMAL:
        print ("update Param using solved solution")
        return 0
    if prob.status == GRB.Status.INF_OR_UNBD:
        print ("Infeasible or unbounded")
        return 1
    if prob.status == GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % prob.status)
        print ("Infeasible!!!")
        return 1
    return 2 #what can happen here?

def main():

    dir_name = "results/"+strftime("%m_%d_%H_%M", gmtime())
    os.mkdir(dir_name)
    logfile = open(dir_name+"/log.txt", "w")

    env = gym.make('HalfCheetah-v2')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    value_net = Value(env.observation_space.shape[0])
  
    sample_policy, sample_eval = Policy(env.observation_space.shape[0], env.action_space.shape[0]), float("-inf")
    prob = Model("mip1")
    (firstParam, firstBias) = initializeLimits(sample_policy, prob)

    if INIT_WEIGHT:
        sample_eval = 1300
        with open("save_1000.p",'rb') as f:
            params=pickle.load(f)
            sample_policy.init_weight(params)


    for i_episode in count(1):
        # Exploration
        num_steps = 0
        explore_episodes = 0
        explore_rew =0
        my_states = {}
        while num_steps < 25000:
            state = env.reset()
            for t in range(1000): # Don't infinite loop while learning
                if num_steps < 20000:
                    action = select_action(state, sample_policy, variance=VARIANCE)
                    name_str = "expl" #explore
                else: 
                    action = select_action(state, sample_policy, variance=0) # 20% randomly good
                    name_str = "eval" #exploit

                next_state, reward, done, _ = env.step(action)
                explore_rew += reward
                sample_policy.rewards.append((reward, t, "%s_%d"%(name_str, explore_episodes)))
                if args.render:
                    env.render()
                if done:
                    break
                state = next_state
            num_steps += (t-1)
            explore_episodes += 1
        
        explore_rew /= explore_episodes

        states, rewards, info = calculate_rewards(explore_episodes, sample_policy)
        values = value_net(torch.Tensor(states))
        advantages = np.subtract(np.array(rewards), values.detach().numpy().flatten())
        print("rewards:")
        print(rewards[:10])
        print("values:")
        print(values[:10].flatten())
        value_net.train(states, rewards)

        my_states = create_state_dict(sample_policy, advantages, info, my_states)

        # sample and solve
        constraints_dict = bestStates(my_states, top_n_constraints=TOP_N_CONSTRIANTS) #only keep the best states
        max_policy, max_eval, max_set = sample_policy, sample_eval, constraints_dict

        print('\nEpisode {}\tExplore reward: {:.2f}\n'.format(i_episode, explore_rew))

        for branch in range(BRANCHES):
            branch_policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
            constraints = dict(random.sample(constraints_dict.items(), N_SAMPLES))

            prob = Model("mip1")
            (firstParam, firstBias) = initializeLimits(branch_policy, prob)

            # Get metadata of constraints
            constraint_info = list(constraints.values())
            vals = [list(v.values())[0][0] for v in constraint_info]
            print("ep %d b %d: constraint mean: %.3f  std: %.3f  max: %.3f" % (i_episode, branch, np.mean(vals), np.std(vals), max(vals)))
            print("constraint set's episode and step number:")
            print([list(v.values())[0] for v in constraint_info])

            # Solve
            exit = solvePolicy(constraints, branch_policy, firstParam, firstBias, prob)
            
            if exit == 0:
                updateParam(prob, branch_policy)
            elif exit == 2:
                print("Error: unhandled solvePolicy case here")
                print('Episode {}\tBranch: {}\tUnsat'.format(i_episode, branch))
                print("L2_NORM")
                print(all_l2_norm(constraints))
                continue
            elif exit == 1:
                print('Episode {}\tBranch: {}\tUnsat'.format(i_episode, branch))
                print("L2_NORM")
                print(all_l2_norm(constraints))
                continue

            # Evaluate
            num_steps = 0
            eval_episodes = 0
            eval_rew = 0
            while num_steps < 6000:
                state = env.reset()
                eval_sum = 0
                for t in range(10000): # Don't infinite loop while learning
                    action = select_action(state, branch_policy, variance=0)
                    state, reward, done, _ = env.step(action)
                    eval_rew += reward
                    branch_policy.rewards.append((reward, t, "%s_%d"%(name_str, explore_episodes)))

                    if args.render:
                        env.render()
                    if done:
                        break
                num_steps += (t-1)
                eval_episodes += 1
            eval_rew /= eval_episodes

            states, rewards, _ = calculate_rewards(eval_episodes, branch_policy)
            value_net.train(states, rewards)

            #log
            print('Episode {}\tBranch: {}\tEval reward: {:.2f}\tExplore reward: {:.2f}'.format(
                i_episode, branch, eval_rew, explore_rew))
            logfile.write('Episode {}\tBranch: {}\tEval reward: {:.2f}\n'.format(i_episode, branch, eval_rew))

            if eval_rew > max_eval:
                max_eval, max_policy, max_set = eval_rew, branch_policy, constraints

        # the end of branching
        if max_eval > sample_eval - NOVELTY_SLACK:
            with open("%s/%d.p"%(dir_name,i_episode), "wb") as f:
                pickle.dump({"all": constraints_dict, "constraints": max_set}, f)
            sample_policy, sample_eval = max_policy, max_eval

def all_l2_norm(constraints):
    states = list(constraints.keys())
    all_dist = []
    for i, x1 in enumerate(states):
        for x2 in states[i+1:]:
            d=np.linalg.norm(np.subtract(x1,x2))
            if d - 0 < 1e-2:
                print("0 dist at state %s with action %s and %s" %(str(x1), str(list(constraints[x1].keys())[0]),str(list(constraints[x2].keys())[0])))
            all_dist.append(d)
    return sorted(all_dist)


if __name__ == '__main__':
    main()
