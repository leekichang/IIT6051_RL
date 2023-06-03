import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
from CartPole import *

train_env = gym.make('CartPole-v1')
test_env = gym.make('CartPole-v1')

SEED = 2022314416

np.random.seed(SEED)
torch.manual_seed(SEED)

class Net(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1  = nn.Linear(4, hidden)
        self.fc2  = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
        
class REINFORCE(CartPole):
    def __init__(self, is_show=False):
        super(REINFORCE, self).__init__(is_show)
        HIDDEN_DIM = 128
        self.model = Net(HIDDEN_DIM)
        self.model.apply(init_weights)
        self.learning_rate = 0.01
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

    def train(self):
        self.model.train()
        log_prob_actions = []
        rewards = []
        done, trunc = False, False
        episode_reward = 0
        state = self.env.reset()[0]
        while not done and not trunc:
            state = torch.FloatTensor(state).unsqueeze(0)
            action_pred = self.model(state)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)
            state, reward, done, trunc, _ = self.env.step(action.item())
            log_prob_actions.append(log_prob_action)
            rewards.append(reward)
            episode_reward += reward
        log_prob_actions = torch.cat(log_prob_actions)
        returns = self.calc_returns(rewards, self.gamma)
        loss = self.update_policy(returns, log_prob_actions)
        return loss, episode_reward
    
    def calc_returns(self, rewards, normalize=True):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R*self.gamma
            returns.insert(0,R)
        if normalize:
            returns = (returns - np.mean(returns)) / np.std(returns)
        return torch.FloatTensor(returns)

    def update_policy(self, returns, log_prob_actions):
        returns = returns.detach()
        loss = - (returns * log_prob_actions).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self):
        self.model.eval()
        done, trunc = False, False
        episode_reward = 0
        state = self.env.reset()[0]
        while not done and not trunc:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_pred = self.model(state)
                action_prob = F.softmax(action_pred, dim = -1)
            action = torch.argmax(action_prob, dim = -1)
            state, reward, done, trunc, _ = self.env.step(action.item())
            episode_reward += reward
        return episode_reward
    
    def explore(self):
        MAX_EPISODES = 1000
        N_STREAK = 25
        N_TRIALS = 25
        REWARD_THRESHOLD = 475
        train_rewards, test_rewards = [], []
        mean_train_rewards, mean_test_rewards = [], []
        streak = 0
        for episode in range(1, MAX_EPISODES+1):
            loss, train_reward = self.train()
            test_reward = self.evaluate()
            train_rewards.append(train_reward)
            test_rewards.append(test_reward)
            mean_train_rewards.append(np.mean(train_rewards[-N_TRIALS:]))
            mean_test_rewards.append(np.mean(test_rewards[-N_TRIALS:]))
            print(f'| Episode: {episode:3} | Mean Train Rewards: {train_rewards[-1]:5.1f} | Mean Test Rewards: {test_rewards[-1]:5.1f} |')
            if test_reward >= REWARD_THRESHOLD:
                streak += 1
                self.save_model(episode)
                if streak >= N_STREAK:
                    print(f'Reached reward threshold in {episode} episodes')
                    break
            else:
                streak = 0
        return test_rewards, mean_test_rewards
    
    def save_model(self, episode):
        save_path = f'./saved_models'
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), f'{save_path}/REINFORCE_{str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))}_{episode}episode.pth')
    
    def infer(self):
        self.env = gym.make('CartPole-v1', render_mode='human')
        saved_models = [f for f in os.listdir('./saved_models') if 'REINFORCE' in f]
        saved_model = Net()
        saved_model.load_state_dict(torch.load(f'./saved_models/{saved_models[0]}'))
        self.model = copy.deepcopy(saved_model)
        for i in range(3):
            done, trunc = False, False
            state = self.env.reset()
            self.model.eval()
            done, trunc = False, False
            score = 0
            state = self.env.reset()[0]
            with torch.no_grad():
                while not done and not trunc:
                    state = torch.FloatTensor(state).unsqueeze(0)
                    action_pred = self.model(state)
                    action_prob = F.softmax(action_pred, dim = -1)
                    action = torch.argmax(action_prob, dim = -1)
                    state, reward, done, trunc, _ = self.env.step(action.item())
                    score += reward
            print(f'Trial {i+1}: score = {score}')
        pass
        
if __name__ == '__main__':
    elapsed_times = []
    for i in range(10):
        reinforce = REINFORCE()
        start = time.process_time()
        r, mr = reinforce.explore()
        end = time.process_time()
        elapsed_times.append(end-start)
        np.save(f'./results/REINFORCE_{i}.npy', np.array(r))
        np.save(f'./results/REINFORCE_mean_{i}.npy', np.array(mr))
    np.save('./results/REINFORCE_time.npy', np.array(elapsed_times))
    plt.plot(r)
    plt.plot(mr)
    plt.show()
    #reinforce.infer()
    