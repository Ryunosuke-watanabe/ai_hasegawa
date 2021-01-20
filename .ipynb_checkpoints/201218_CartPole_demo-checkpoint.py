import gym
import numpy as np
import matplotlib.pyplot as plt

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
    anim.save('cartpole-demo.mp4')
    display(display_animation(anim, default_mode='loop'))

# CartPoleを指定
# 環境を指定
env = gym.make("CartPole-v0")
# 環境をリセット
observation = env.reset()

frames = []

for step in range(0, 100):
    # カートを動かす方向を指定(今回はランダム)
    # 0:左, 1:右
    action = np.random.choice(2)
    observation, reward, done, info = env.step(action)

    frames.append(env.render(mode='rgb_array'))
    # env.render(mode='rgb_array')
# env.reset()

save_as_gif(frames)