import gym
from time import time #just to have timestamps in the files

env_to_wrap = gym.make(ENV_NAME)
env = gym.wrappers.Monitor(env, './videos/' + str(time()) + '/')

observation = env.reset()



env.close()
env_to_wrap.close()

