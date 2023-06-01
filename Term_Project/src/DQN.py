import random
import numpy as np
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from CartPole import *

random.seed(2022314416)

class Net(nn.Module):
    def __init__(self, action_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=4 , out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=24)
        self.fc3 = nn.Linear(in_features=24, out_features=action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        self.memory = deque(maxlen=2000)
        self.model = Net(action_size)
        self.target_model = Net(action_size)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()
        
    def update_target_model(self):
        self.target_model = copy.deepcopy(self.model)
  
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
  
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state = torch.tensor(state)
            action = torch.argmax(self.model(state)).numpy()
        return action
  
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        mini_batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor(np.array([sample[0][0] for sample in mini_batch]))
        actions = torch.tensor(np.array([sample[1] for sample in mini_batch]))
        rewards = torch.tensor(np.array([sample[2] for sample in mini_batch]))
        next_states = torch.tensor(np.array([sample[3][0] for sample in mini_batch]))
        dones = torch.tensor(np.array([sample[4] for sample in mini_batch]), dtype=torch.float32)
        
        self.model.train(); self.target_model.eval()
        
        predicts = self.model(states)
        one_hot_action = F.one_hot(actions.to(torch.int64), num_classes=self.action_size)
        predicts = torch.sum(one_hot_action*predicts, -1)
        with torch.no_grad():
            target_predict = self.target_model(next_states)

        max_q = torch.amax(target_predict, dim=-1)
        
        targets = (rewards + (1 - dones) * self.discount_factor * max_q).to(torch.float32)
        loss = self.criterion(predicts, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

class DQN(CartPole):
    def __init__(self, is_show=False):
        super(DQN, self).__init__(is_show)
        self.state_size  = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = copy.deepcopy(DQNAgent(self.state_size, self.action_size))
    
    def explore(self):
        scores, episodes = [], []  
        score_avg = 0
        num_episode = 1000
        converge = False
        num_streak = 0
        e = 0
        while not converge and e < num_episode:
            done = False
            state = self.env.reset()
            state = state[0].reshape(1, -1)
            score = 0
            while not done:
                action = self.agent.choose_action(state)

                next_state, reward, done, trunc, info = self.env.step(action)
                next_state = next_state.reshape(1, -1)

                score += reward
                reward = reward if not done else -100

                self.agent.remember(state, action, reward, next_state, done)
            
                if len(self.agent.memory) >= self.agent.train_start:
                    self.agent.train_model()

                state = next_state

                if done or trunc:
                    self.agent.update_target_model()

                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    print(f'episode: {e:3d} | score {score:6.2f} | memory length: {len(self.agent.memory):4d} | epsilon: {self.agent.epsilon:.4f}')

                    scores.append(score)
                    episodes.append(e)
                    plt.plot(episodes, scores, 'b')
                    plt.xlabel('episode')
                    plt.ylabel('average score')
                    plt.savefig('cartpole_graph.png')

                    if score > 475:
                        num_streak += 1
                    else:
                        num_streak = 0
                    
                    if num_streak > 25:
                        save_path = f'./save_model'
                        torch.save(self.agent.model.state_dict(), f'{save_path}/{str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))}.pth')
                        converge = True
                    break          
            e += 1       
if __name__ == "__main__":
    dqn = DQN()
    dqn.explore()