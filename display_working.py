from notebooks.JSAnimation import IPython_display
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

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save("gifs/%s.gif"%(name), writer='imagemagick', fps=60)
    #display(IPython_display.display_animation(anim, default_mode='loop'))

def getframes(policy, env):
    observation = env.reset()
    cum_reward = 0
    frames = []
    for t in range(1000):
        # Render into buffer. 
        fr = env.render(mode = 'rgb_array')
        print(fr)
        frames.append(fr)
        obs =  torch.from_numpy(observation).unsqueeze(0).float()
        action = policy(obs).detach().numpy()
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        if done:
            break
    print("reward: %.3f" % cum_reward)
    return frames[:steps]

def play(policy, env):
    observation = env.reset()
    cum_reward = 0
    for t in range(5):
        
        fr = env.render(mode = 'rgb_array')
        print(fr)
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
    model.load_state_dict(pickle.load(open(path, "rb" )))
    return model

path = "results/Hopper-v2/imitation-test-08_04_04_32/209_"
env = gym.make('Hopper-v2')
constraints = pickle.load( open( path+"constraints.p", "rb" ))
model = load_policy(path+"policy.p", env)

for i in range(5):
    print(play(model, env))

frames_1000 = getframes(model, env)
display_frames_as_gif(frames_1000, name="hopper3000")