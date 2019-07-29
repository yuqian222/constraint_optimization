import numpy as np
import gym



class Trained_model_wrapper_lin():
    def __init__(self, env_name, load_file_path):

        print('loading trained linear policy')
        
        # a hack: save np.load
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        lin_policy = np.load(load_file_path)
        lin_policy = list(lin_policy.items())[0][1]

        # restore np.load for future normal usage
        np.load = np_load_old
            
        self.M = lin_policy[0]
        # mean and std of state vectors estimated online by ARS. 
        self.mean = lin_policy[1]
        self.std = lin_policy[2]
        print(self.mean)
        print(self.std)

    def select_action(self, state, variance): #variance is a dummy
        return np.dot(self.M, (state - self.mean)/self.std)

    def play(self, num_rollouts=20):
        self.env = gym.make(env_name)
        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = np.dot(self.M, (obs - self.mean)/self.std)
                observations.append(obs)
                actions.append(action)
        
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1
                if steps >= self.env.spec.timestep_limit:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        
if __name__ == '__main__':
    wrap = Trained_model_wrapper_lin("Hopper-v2", 
        "./trained_models/Hopper_lin_policy.npz")
    wrap.play()