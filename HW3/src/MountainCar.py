from typing import Any
from config import *
#import gymnasium as gym
from tqdm import tqdm
import gym
import sys
from FunctionApproximator import *
import pickle

class MountainCar:
    def __init__(self, n_step=1, is_show=False):
        # self.env = gym.make('MountainCar-v0', render_mode="human")
        self.is_show = is_show
        self.n_step  = n_step
        if is_show:
            self.env = gym.make('MountainCar-v0', render_mode="human")
        else:
            self.env = gym.make('MountainCar-v0')
        #self.env.metadata['render_fps'] = 60
        
        self.actions    = list(ACTIONS.keys())
        self.agent      = None
        self.currState  = None
        
    def set_agent(self, agent):
        self.agent = agent
        self.agent.set_len(self.get_len())
    
    def get_len(self):
        posMax = self.env.observation_space.high[0]
        posMin = self.env.observation_space.low[0]
        velMax = self.env.observation_space.high[1]
        velMin = self.env.observation_space.low[1]
        self.pos_len = posMax-posMin
        self.vel_len = velMax-velMin
        return [self.pos_len, self.vel_len]
    
    def step(self, action):
        next_state, reward, is_terminal, _, _= self.env.step(action)
        next_action = self.agent.getAction(next_state)
        return (next_state, next_action, reward, is_terminal)
    
    def sarsa(self):
        time, is_terminal, T     = 0, False, float('inf')
        self.currState, _        = self.env.reset()
        action                   = self.agent.getAction(self.currState)
        states, actions, rewards = [self.currState], [action], [0]

        while True:
            time += 1
            if time < T:
                next_state, next_action, reward, is_terminal = self.step(action)
                states.append(next_state)
                actions.append(next_action)
                rewards.append(reward)

                if is_terminal:
                    T = time
        
            update_time = time - self.n_step
            if time >= self.n_step:
                returns = 0.0
                for t in range(update_time + 1, \
                            min(T, update_time + self.n_step) + 1):
                    returns += rewards[t]

                if time <= T:
                    returns += self.agent.getValue(states[update_time+self.n_step],
                                                actions[update_time+self.n_step])
                
                if not is_terminal:
                    self.agent.learn(states[update_time], actions[update_time], returns)

            if update_time == T-1:
                break
            self.currState, action = next_state, next_action
        return time
    
    def fit(self, episodes):
        for episode in tqdm(range(episodes)):
            self.sarsa()            
            if episode + 1 in SAVE_EPISODE:
                self.save_model(episode+1)
        self.env.close()
        
        
    
    def show(self, runs):
        self.env = gym.make('MountainCar-v0', render_mode="human")
        self.env.reset()
        for _ in range(runs):
            is_terminal = False
            self.currState, _ = self.env.reset()
            action = self.agent.getAction(self.currState)
            while not is_terminal:
                self.currState, action, _, is_terminal = self.step(action)

    def save_model(self, episode):
        with open(f'{SAVE_PATH}/{self.agent.alpha * self.agent.n_tiles:.1f}_{self.n_step}_{episode}.pickle',"wb") as f:
            pickle.dump(self,f)

    def plot_cost(self, episode, ax):
        grid_size = 40
        positions = np.linspace(-1.2, 0.6, grid_size)
        velocities = np.linspace(-0.07, 0.07, grid_size)
        axis_x = []
        axis_y = []
        axis_z = []
        for position in positions:
            for velocity in velocities:
                axis_x.append(position)
                axis_y.append(velocity)
                axis_z.append(self.agent.cost_to_go((position, velocity)))

        ax.scatter(axis_x, axis_y, axis_z)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Cost to go')
        ax.set_title('Episode %d' % (episode))
    
    def plot_policy(self, episode, ax):
        positions = np.linspace(-1.2, 0.6, 100)
        velocities = np.linspace(-0.07, 0.07, 100)
        axis_x = []
        axis_y = []
        colors = []
        labels = []
        cmap   = {0:'b', 1:'r', 2:'g'}
        points = {0:{'x':[],'y':[]},
                  1:{'x':[],'y':[]},
                  2:{'x':[],'y':[]}}
        for position in positions:
            for velocity in velocities:
                action_ = self.agent.cost_to_go_policy((position, velocity))
                points[action_]['x'].append(position)
                points[action_]['y'].append(velocity)
        for action_ in points:
            ax.scatter(points[action_]['x'],
                       points[action_]['y'],
                       marker='*',
                       color=cmap[action_],
                       label=ACTIONS[action_],
                       s=20)    
            
        ax.legend()
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title('Episode %d' % (episode))

if __name__ == '__main__':  
    print("IIT-6051 Reinforcement Learning HW3")
    mountainCar = MountainCar(is_show=False, n_step=4, episodes=9000)
    agent       = FunctionApproximator(alpha=0.5)
    mountainCar.set_agent(agent=agent)
    mountainCar.fit()
    
    # with open(f'{SAVE_PATH}/test.pickle',"rb") as f:
    #     mountainCar = pickle.load(f)
    # mountainCar.show()
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(40, 10))
    # plot_episodes = [100]
    # axes = [fig.add_subplot(1, len(plot_episodes), i+1, projection='3d') for i in range(len(plot_episodes))]
    # num_tiles = 8
    # for idx, ep in enumerate(plot_episodes):
    #     print_cost(mountainCar.agent, episode=ep, ax=axes[idx])
    # plt.show()
    
    
    # for epi in SAVE_EPISODE:
    #     agent.set_weight(np.copy(np.load(f'{SAVE_PATH}/0.0375_8_1_{epi}.npy')))
    #     mountainCar.show()