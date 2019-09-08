'''
Test script to see how well we can reproduce a 
random given neural policy
'''
import gym, pickle, torch 
import numpy as np
from nn_policy import Policy_quad


def play(policy, env, device):
    observation = env.reset()
    cum_reward = 0
    for t in range(1000):
        obs =  torch.from_numpy(observation).unsqueeze(0).float().to(device)
        action = policy(obs).cpu().detach().numpy()
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        if done:
            break
    return cum_reward

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

def load_policy(path, env):
    model = Policy_quad(env.observation_space.shape[0],
                env.action_space.shape[0],
                num_hidden=24)
    sd = pickle.load(open(path, "rb" ))
    #print(sd)
    model.load_state_dict(sd)
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = "./trained_models/209_"
    env = gym.make('Hopper-v2')
    constraints = pickle.load( open( path+"constraints.p", "rb" ))
    model = load_policy(path+"policy.p", env).to(device)
    
    '''
    print("______________Experience Set________________") # result set from on-policy data
    for _ in range(10):
        policy209 = Policy_quad(env.observation_space.shape[0],
                    env.action_space.shape[0],
                    num_hidden=24).to(device)
        states = torch.Tensor([x[0] for x in constraints])
        actions = torch.Tensor([x[1] for x in constraints])
        policy209.train(states.to(device), actions.to(device), 10000)

        print([play(policy209, env,device) for _ in range(5)])

    '''
    print("______________Policy________________") # sample from state domain policy
    x_test, y_test = random_sample_hopper(model, 200)

    for i in range(5, 7):
        print("Data size: 10^%d" % i)

        x, y = random_sample_hopper(model, 10**i)
        learner = Policy_quad(env.observation_space.shape[0],
                    env.action_space.shape[0],
                    num_hidden=24).to(device)

        states = torch.Tensor(x)
        actions = torch.Tensor(y)
        learner.train(states.to(device), actions.to(device), 10000)

        y_learner = learner(torch.Tensor(x_test).to(device))
        mse = ((y_test - y_learner.detach().cpu().numpy())**2)[0].mean(axis=0)
        print("Mean square error on test set:", mse)

        print("Rollout performance")
        print([play(learner, env,device) for _ in range(5)])



