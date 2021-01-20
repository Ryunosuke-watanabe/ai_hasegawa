import numpy as np
import matplotlib.pyplot as plt
import gym

ENV = 'CartPole-v0'
NUM_DIGITIZED = 6
GAMMA = 0.99
ETA = 0.5
MAX_STEP = 200
NUM_EPISODES = 1000

env = gym.make(ENV)

observation = env.reset()

def bins(min, max, num):
    return np.linspace(min, max, num+1)

def digitize_state(observation):
    cart_pos, cart_v, pole_angel, pole_v = observation

    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, NUM_DIGITIZED)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, NUM_DIGITIZED)),
        np.digitize(pole_angel, bins=bins(-0.5, 0.5, NUM_DIGITIZED)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, NUM_DIGITIZED))
    ]

    return sum([x * NUM_DIGITIZED**i for i, x in enumerate(digitized)])

from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

def save_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
        
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save('RL-cartpole-demo.mp4')
    display(display_animation(anim, default_mode='loop'))

class Agent:

    def __init__(self, num_status, num_actions):
        self.brain = Brain(num_status, num_actions)
    
    def update_Q_functions(self, observation, action, reward, observation_next):
        self.brain.update_Q_table(observation, action, reward, observation_next)

    def get_antion(self, observation, step):
        return self.brain.decide_action(observation, step)

class Brain:


class Enviroment:
    
