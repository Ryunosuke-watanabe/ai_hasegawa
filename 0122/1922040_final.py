#%%
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# %%
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

# %%
Transition = namedtuple('Transition', 'state actions next_state reward')

# 学習に使う変数を整理
ENV = 'CartPole-v0'
# 報酬割引率
GAMMA = 0.99
# 1試行の最大ステップ数
MAX_STEP = 200
# 最大試行回数
NUM_EPISODES = 1000
# バッチサイズ
BATCH_SIZE = 32
# キャパ
CAPACITY = 10000

# %%
# ミニバッチ学習のための経験データを保存するクラス
class ReplayMemory:

    def __init__(self, CAPACITY):
        # メモリ容量
        self.capacity = CAPACITY
        # 経験を保存する
        self.memory = []
        # 保存場所を示す変数
        self.index = 0

    def push(self, state, actions, next_state, reward):

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        # 状態をメモリに保存
        self.memory[self.index] = Transition(state, actions, next_state, reward)
        # インデックスを1ずらす
        self.index = (self.index + 1) % self.capacity

    # 指定したバッチサイズ分、ランダムに経験を取り出す
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # memoryの長さを返す
    def __len__(self):
        return len(self.memory)

# %%
# エージェントが行う行動を与えられた状態によって判断する部分（深層強化学習（DQN）を行う部分）

class Brain():

    def __init__(self, num_state, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)

        # ニューラルネットワーク
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_state, 32))
        self.model.add_module('relu', nn.ReLU())
        # 好きに組んでみる

        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu', nn.ReLU())

        # self.model.add_module('fc3', nn.Linear(32, 32))
        # self.model.add_module('relu', nn.ReLU())

        self.model.add_module('fc4', nn.Linear(32, num_actions))

        # 最適化関数
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def replay(self):
        # 最初にメモリサイズを確認する
        # 指定したバッチサイズより小さい場合は何もしない
        if len(self.memory) < BATCH_SIZE:
            return
        
        # ミニバッチ用のデータを取得(ランダム)
        transition = self.memory.sample(BATCH_SIZE)

        # transitionは(state, actions, next_state, reward) * BATCH_SIZE
        # (state * BATCH_SIZE, actions * BATCH_SIZE, next_state * BATCH_SIZE, reward * BATCH_SIZE)
        batch = Transition(*zip(*transition))

        state_batch = torch.cat(batch.state)
        actions_batch = torch.cat(batch.actions)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # 教師信号となるQ(s_t, a_t)を求める
        # モデルを訓練モードに
        self.model.eval()

        state_actions_values = self.model(state_batch).gather(1, actions_batch)

        # CarPoleがdoneになっていない、かつ、next_stateがあるかをチェックするためのマスクを作成
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        # maxQ(s_t+1, a)を求める
        next_state_values = torch.zeros(BATCH_SIZE)

        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        # Q学習の行動価値関数更新式からQ(s_t, a_t)を求める
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # モデルを訓練モードに切替
        self.model.train()

        # 二乗誤差の代わりにHuder関数を使う
        loss = F.smooth_l1_loss(state_actions_values, expected_state_action_values.unsqueeze(1))

        # 勾配をリセット
        self.optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()
        # ニューラルネットワークの重み更新
        self.optimizer.step()

    def decide_action(self, state, episode):
        # ε-greedy法で徐々に最適行動を採用するようにする
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            # 推論モードに切り替える
            self.model.eval()
            with torch.no_grad():
                actions = self.model(state).max(1)[1].view(1, 1)

        else:
            # 右、左ランダムに移動する
            # actionは[torch.LongTensor of size 1 * 1]
            actions = torch.LongTensor([[random.randrange(self.num_actions)]])
        
        return actions

# %%
# エージェントクラス
class Agent:
    def __init__(self, num_states, num_actions):
        # Brainクラスをインタンス化
        self.brain = Brain(num_states, num_actions)

    # Q学習の更新
    def update_q_function(self):
        self.brain.replay()

    # アクションを決定する
    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action
    
    # 状態を保存
    def memorize(self, state, action, next_state, reward):
    # def memorize(self, state, next_state, reward):
        self.brain.memory.push(state, action, next_state, reward)

# %%
class Environment():

    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)

    def run(self):

        episode_10_list = np.zeros(10)
        complete_episodes = 0  # 195step以上連続で立ち続けた試行数
        is_episode_final = False  # 最終試行フラグ
        frames = []  # 動画用に画像を格納する変数

        # 全エピソードループ
        for i in range(NUM_EPISODES):
            observation = self.env.reset()
            # episode_reward = 0

            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            # 1エピソードループ
            for l in range(MAX_STEP):
                if is_episode_final is True:
                    # framesに各時刻の画像を追加していく
                    frames.append(self.env.render(mode='rgb_array'))

                # 最初の行動を決める
                action = self.agent.get_action(state, i)

                # 最初の行動から次の状態を求める
                observation_next, reward_notuse, done, info_notuse = self.env.step(action.item())

                # 報酬を与える
                if done:
                    # 次の状態はないのでNoneを代入
                    state_next = None

                    # 直前10エピソードで立てた平均ステップ数を格納
                    episode_10_list = np.hstack((episode_10_list[1:], l + 1))

                    if l < 195:
                        reward = torch.FloatTensor([-1.0])
                        self.complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes += 1
                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)
                
                # メモリに経験を追加
                self.agent.memorize(state, action, state_next, reward)
                
                self.agent.update_q_function()
                
                state = state_next
                
                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(i, l+1))
                    break

            if is_episode_final is True:
                # 動画を保存と描画
                save_as_gif(frames)
                break

            if complete_episodes >= 10:
                print('10回連続成功')
                frames = []
                is_episode_final = True  # 次の試行を描画を行う最終試行とする

# %%
cartpole_env = Environment()
cartpole_env.run()
# %%
