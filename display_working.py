from imageio import *
from matplotlib import animation
import matplotlib.pyplot as plt

import gym, pickle,os,torch, time
from policies import *

def display_frames_as_gif(frames, name="untitled"):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    #anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    #anim.save("gifs/%s.gif"%(name), writer='imagemagick', fps=60)
    #display(IPython_display.display_animation(anim, default_mode='loop'))
    imageio.mimsave("gifs/%s.gif"%(name),frames)

def getframes(policy, env, n):
    observation = env.reset()
    cum_reward = 0
    frames = []
    for t in range(n):
        # Render into buffer. 
        fr = env.render(mode = 'rgb_array')
        time.sleep(0.1)
        frames.append(fr)
        obs =  torch.from_numpy(observation).unsqueeze(0).float()
        action = policy(obs).detach().numpy()
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        if t % 20 ==0:
            print("got to %d"%t)
        if done:
            break
    print("reward: %.3f" % cum_reward)
    return frames

def play(policy, env):
    observation = env.reset()
    cum_reward = 0
    for t in range(1000):
        obs =  torch.from_numpy(observation).unsqueeze(0).float()
        action = policy(obs).detach().numpy()
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        if done:
            break
    return cum_reward

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

    path = "results/Hopper-v2/imitation-test-08_04_04_32/209_"
    env = gym.make('Hopper-v2')
    constraints = pickle.load( open( path+"constraints.p", "rb" ))
    model = load_policy(path+"policy.p", env)

    print("______________209________________")
    for _ in range(10):
        policy209 = Policy_quad(env.observation_space.shape[0],
                    env.action_space.shape[0],
                    num_hidden=24)
        states = torch.Tensor([x[0] for x in constraints])
        actions = torch.Tensor([x[1] for x in constraints])
        policy209.train(states.to(device), actions.to(device), 5000)

        print([play(policy209, env) for _ in range(5)])


    path = "results/Hopper-v2/imitation-test-08_04_04_32/210_"
    constraints210 = pickle.load( open( path+"constraints.p", "rb" ))

    print("______________210________________")
    for _ in range(10):
        policy210 = Policy_quad(env.observation_space.shape[0],
                    env.action_space.shape[0],
                    num_hidden=24)
        states = torch.Tensor([x[0] for x in constraints210])
        actions = torch.Tensor([x[1] for x in constraints210])
        policy210.train(states.to(device), actions.to(device), 5000)

        print([play(policy210, env) for _ in range(5)])





    '''
    path = "replay/trained_models/209_"
    env = gym.make('Hopper-v2')
    constraints = pickle.load(open( path+"constraints.p", "rb" ))
    model = load_policy(path+"policy.p", env)

    for i in range(2):
        print(play(model, env))

    frames_1000 = getframes(model, env, 100)
    mimsave("gifs/%s.gif"%("hopper3000"),frames_1000)
    '''


