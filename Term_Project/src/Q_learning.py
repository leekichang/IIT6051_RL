import gym
import math
import random
import numpy as np
from tqdm import tqdm
from CartPole import *
""" 
Implementation reference:
https://arxiv.org/abs/2006.04938
"""

random.seed(2022314416)
  
class QL(CartPole):
    def __init__(self, is_show=False):
        super().__init__(is_show)
        self.path = './saved_weights'
        self.STATE_BOUNDS = list(zip(self.env.observation_space.low,\
                                     self.env.observation_space.high))
        self.STATE_BOUNDS[1] = [-0.5, 0.5]
        self.STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
        
        self.NUM_BUCKETS = (1, 1, 6, 3)
        self.NUM_ACTIONS = 2
        self.q_table = np.zeros(self.NUM_BUCKETS + (self.NUM_ACTIONS,))
        
        self.MIN_EXPLORE_RATE = 0.01
        self.MIN_LEARNING_RATE = 0.1

        self.NUM_EPISODES = 1000
        self.MAX_T = 250
        self.STREAK_TO_END = 100
        self.SOLVED_T = 199
        
    def state_to_bucket(self, state):
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= self.STATE_BOUNDS[i][0]:
                bucket_index = 0
            elif state[i] >= self.STATE_BOUNDS[i][1]:
                bucket_index = self.NUM_BUCKETS[i] - 1
            else:
                bound_width = self.STATE_BOUNDS[i][1] - self.STATE_BOUNDS[i][0]
                offset = (self.NUM_BUCKETS[i]-1)*self.STATE_BOUNDS[i][0]/bound_width
                scaling = (self.NUM_BUCKETS[i]-1)/bound_width
                bucket_index = int(round(scaling*state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)
    
    def select_action(self, state, explore_rate):
        if random.random() < explore_rate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
        
    def get_explore_rate(self, t):
        return max(self.MIN_EXPLORE_RATE, min(1.0, 1.0 - math.log10((t+1)/25)))
        # if t >= 24:
        #     return max(self.MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))
        # else:
        #     return 1.0
        
    def get_learning_rate(self, t):
        return max(self.MIN_LEARNING_RATE, min(1.0, 1.0 - math.log10((t+1)/25)))
        # if t >= 24:
        #     return max(self.MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))
        # else:
        #     return 1.0
        
    def fit(self):
        learning_rate = self.get_learning_rate(0)
        explore_rate  = self.get_explore_rate(0)
        discount_factor = 0.99

        num_streaks = 0

        for episode in tqdm(range(self.NUM_EPISODES)):
            obv     = self.env.reset()
            state_0 = self.state_to_bucket(obv[0])

            for t in range(self.MAX_T):
                action = self.select_action(state_0, explore_rate)
                obv, reward, done, _, _ = self.env.step(action)

                state = self.state_to_bucket(obv)

                best_q = np.amax(self.q_table[state])
                self.q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - self.q_table[state_0 + (action,)])

                state_0 = state

                if done:
                    print(f"Episode {episode} finished after {t} time steps")
                    if (t >= self.SOLVED_T):
                        num_streaks += 1
                    else:
                        num_streaks = 0
                    break
            if num_streaks > self.STREAK_TO_END:
                break

            explore_rate  = self.get_explore_rate(episode)
            learning_rate = self.get_learning_rate(episode)
            if (episode+1) % 100 == 0:
                self.save_q_table(episode)
                print(f'Episode {episode+1} saved!')
                
    def save_q_table(self, episode):
        import os
        os.makedirs(self.path, exist_ok=True)
        np.save(f'{self.path}/q_table_{episode+1}.npy', self.q_table)
    
    def infer(self, episode):
        self.q_table = np.load(f'{self.path}/q_table_{episode}.npy')
        print(f"Infer with {episode} Episode Q table")
        for _ in range(1):
            obv     = self.env.reset()
            state_0 = self.state_to_bucket(obv[0])
            for t in tqdm(range(self.MAX_T)):
                #self.env.render()
                action = self.select_action(state_0, 0)
                obv, reward, done, _, _ = self.env.step(action)

                state = self.state_to_bucket(obv)

                state_0 = state

                if done:
                    print(f"Episode {episode} finished after {t} time steps")
                    if (t >= self.SOLVED_T):
                        num_streaks += 1
                    else:
                        num_streaks = 0
                    break

    
if __name__ == '__main__':
    ql = QL(is_show=False)
    ql.fit()
    ql = QL(is_show=True)
    ql.infer(1000)
    