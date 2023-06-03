import random
import numpy as np
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from CartPole import *

SEED = 2022314416
torch.manual_seed(SEED)
random.seed(2022314416)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

class Net(nn.Module):
    def __init__(self, hidden=24):
        super().__init__()
        self.fc1  = nn.Linear(4, hidden)
        self.fc2  = nn.Linear(hidden, hidden)
        self.fc3  = nn.Linear(hidden, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, memory_size=2000, hidden_size=24):
        self.state_size      = state_size
        self.action_size     = action_size
        self.discount_factor = 0.99
        self.learning_rate   = 0.001
        self.epsilon         = 1.0
        self.epsilon_decay   = 0.999
        self.epsilon_min     = 0.01
        self.batch_size      = 64
        self.train_start     = 1000
        self.memory          = deque(maxlen=memory_size)
        self.model           = Net(hidden_size)
        self.model.apply(init_weights)
        self.target_model    = Net(hidden_size)
        self.optimizer       = optim.Adam(params=self.model.parameters(),
                                    lr=self.learning_rate)
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
    def __init__(self, is_show=False, memory_size=2000, hidden_size=24):
        super(DQN, self).__init__(is_show)
        self.state_size  = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = copy.deepcopy(DQNAgent(self.state_size, self.action_size, memory_size, hidden_size))
    
    def fill_memory(self): 
        while len(self.agent.memory) < self.agent.train_start:
            done = False
            state = self.env.reset()
            state = state[0].reshape(1, -1)
            score = 0
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                next_state = next_state.reshape(1, -1)
                score += reward
                reward = reward if not done else -100
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
    
    def explore(self):
        self.fill_memory()
        scores, episodes, mean_scores = [], [], []
        #score_avg = 0
        max_episode = 1000
        converge = False
        num_streak = 0
        N_TRIAL = 25
        e = 0
        while not converge and e < max_episode:
            done = False
            state = self.env.reset()
            state = state[0].reshape(1, -1)
            score = 0
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                next_state = next_state.reshape(1, -1)
                score += reward
                reward = reward if not done else -100
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.train_model()
                state = next_state
                if done or trunc:
                    self.agent.update_target_model()
                    #score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    print(f'episode: {e:3d} | score {score:6.2f} | memory length: {len(self.agent.memory):4d} | epsilon: {self.agent.epsilon:.4f}')
                    scores.append(score)
                    mean_scores.append(np.mean(scores[-N_TRIAL:]))
                    episodes.append(e)
                    plt.cla()
                    plt.plot(episodes, scores, 'b')
                    plt.xlabel('episode')
                    plt.ylabel('average score')
                    plt.savefig('cartpole_graph.png')
                    if score > 475:
                        num_streak += 1
                    else:
                        num_streak = 0
                    if num_streak > 25:
                        self.save_model(e)
                        converge = True
                    break          
            e += 1
        return scores, mean_scores
            
    def infer(self):
        self.env = gym.make('CartPole-v1', render_mode='human')
        saved_models = os.listdir('./saved_models')
        saved_model = Net()
        saved_model.load_state_dict(torch.load(f'./saved_models/{saved_models[0]}'))
        self.agent.model = copy.deepcopy(saved_model)
        for i in range(3):
            done, trunc = False, False
            state = self.env.reset()
            state = state[0].reshape(1, -1)
            score = 0
            action = self.env.action_space.sample()
            with torch.no_grad():
                while not done and not trunc:
                    next_state, reward, done, trunc, _ = self.env.step(action)
                    state = torch.tensor(next_state.reshape(1, -1))
                    action = torch.argmax(self.agent.model(state)).numpy()
                    score += reward
            print(f'Trial {i+1}: score = {score}')
            
    def save_model(self, episode):
        save_path = f'./saved_models'
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.agent.model.state_dict(), f'{save_path}/DQN_{str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))}_{episode}episode.pth')
if __name__ == "__main__":
    fig = plt.figure(figsize=(6,3))
    elapsed_times = []
    for i in range(10):
        dqn = DQN()
        start = time.process_time()
        r, mr = dqn.explore()
        end = time.process_time()
        elapsed_times.append(end-start)
        np.save(f'./results/DQN_{i}.npy', np.array(r))
        np.save(f'./results/DQN_mean_{i}.npy', np.array(mr))
        
    np.save('./results/DQN_time.npy', np.array(elapsed_times))
    plt.plot(r)
    plt.plot(mr)
    plt.show()
    #dqn.infer()