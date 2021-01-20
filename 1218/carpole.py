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
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        # Qテーブルを作成。行数は状態を分割数^（4変数）にデジタル変換した値、列数は行動数を示す
        self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIGITIZED**num_states, num_actions))


    # カート情報を離散化するための閾値を求める
    def bins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    # カートの情報を離散化する
    def digitize_state(self, observation):
        '''観測したobservation状態を、離散値に変換する'''
        cart_pos, cart_v, pole_angle, pole_v = observation
        digitized = [
            np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIGITIZED)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIGITIZED)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIGITIZED)),
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIGITIZED))
        ]
        return sum([x * (NUM_DIGITIZED**i) for i, x in enumerate(digitized)])

    # QテーブルをQ学習により更新
    def update_Q_table(self, observation, action, reward, observation_next):
        if s_next == 8:
            Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
        else:
            Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])
        return Q
        


    # ε-greedy法で徐々に最適行動を行うようにする
    def decide_action(self, observation, episode):
        # ほげほげ
        return action

class Environment:

    def __init__(self):
        self.env = gym.make(ENV)  # 実行する課題を設定
        num_states = self.env.observation_space.shape[0]  # 課題の状態を取得
        num_actions = self.env.action_space.n  # CartPoleの行動を取得
        self.agent = Agent(num_states, num_actions)  # 環境内で行動するAgentを生成

    ## 実行関数
    def run(self):

        complete_episodes = 0  # 195step以上連続で立ち続けた試行数
        is_episode_final = False  # 最終試行フラグ
        frames = []  # 動画用に画像を格納する変数

        # 全エピソードループ
            # ほげほげ

            # 1エピソードループ
            # ほげほげ



    # 最終試行では動画を保存と描画


    # 10エピソード連続成功なら次の試行を描画を行う最終試行とする


