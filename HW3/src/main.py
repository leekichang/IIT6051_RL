import pickle
import numpy as np
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from FunctionApproximator import *
from MountainCar import MountainCar

def Q1(episodes:list):
    fig = plt.figure(figsize=(15, 5))
    axes = [fig.add_subplot(1, len(episodes), i+1, projection='3d')\
             for i in range(len(episodes))]
    for idx, ep in enumerate(episodes):
        with open(f'{SAVE_PATH}/0.3_1_{ep}.pickle', 'rb') as f:
            mountainCar = pickle.load(f)
        mountainCar.plot_cost(ep, axes[idx])
    # plt.tight_layout()
    plt.savefig('../images/Q1.png')
    plt.close()

def Q2():
    runs = 10
    episodes = 500
    n_tiles = 8
    alphas = [0.1, 0.2, 0.5]
    
    mountainCars = [MountainCar(n_step=1) for _ in range(len(alphas))]
    
    steps = np.zeros((len(alphas), episodes))
    
    for run in range(runs):
        vfs = [FunctionApproximator(n_tiles=n_tiles, alpha=alpha) for alpha in alphas]
        for idx, v in enumerate(vfs):
            mountainCars[idx].set_agent(v)
        for idx in range(len(vfs)):
            for episode in tqdm(range(episodes)):
                step = mountainCars[idx].sarsa()
                steps[idx, episode] += step
    steps /= runs

    for i in range(0, len(alphas)):
        plt.plot(steps[i], label='alpha = '+str(alphas[i])+'/'+str(n_tiles))
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()

    plt.savefig('../images/Q2.png')
    plt.close()

def Q2_animation(episodes:list, alpha=0.3, n_step=1):
    runs = 3
    for episode in episodes:
        print(f"EPISODE: {episode}")
        with open(f'{SAVE_PATH}/{alpha}_{n_step}_{episode}.pickle',"rb") as f:
            mountainCar = pickle.load(f)
        mountainCar.show(runs)
        mountainCar.env.close()

def Q3(episodes:list):
    fig = plt.figure(figsize=(10, 10))
    axes = [fig.add_subplot(1, len(episodes), i+1)\
             for i in range(len(episodes))]
    for idx, ep in enumerate(episodes):
        with open(f'{SAVE_PATH}/0.5_4_{ep}.pickle', 'rb') as f:
            mountainCar = pickle.load(f)
        mountainCar.plot_policy(ep, axes[idx])
    plt.show()

def Q3_test(episode:list):
    plt.figure(figsize=(5, 5))
    cmap   = {0:'b', 1:'r', 2:'g'}
    points = {0:{'x':[],'y':[]},
              1:{'x':[],'y':[]},
              2:{'x':[],'y':[]}}
    with open(f'{SAVE_PATH}/0.5_4_{episode}.pickle', 'rb') as f:
        mountainCar = pickle.load(f)
    # import gym
    # mountainCar.env = gym.make('MountainCar-v0', render_mode="human")
    mountainCar.env.reset()
    mountainCar.currState, _ = mountainCar.env.reset()
    action = mountainCar.agent.getAction(mountainCar.currState)
    for _ in tqdm(range(100)):
        is_terminal = False
        mountainCar.env.reset()
        mountainCar.env.state = (np.random.uniform(-1.2, 0.6), np.random.uniform(-0.07, 0.07))
        while not is_terminal:
            points[action]['x'].append(mountainCar.currState[0])
            points[action]['y'].append(mountainCar.currState[1])
            mountainCar.currState, action, _, is_terminal = mountainCar.step(action)
    for action_ in points:
        plt.scatter(points[action_]['x'],
                   points[action_]['y'],
                   marker='x',
                   color=cmap[action_],
                   label=ACTIONS[action_],
                   s=50,
                   alpha=0.2)   
    plt.legend()
    plt.xlim([-1.3,0.7])
    plt.ylim([-0.08,0.08])
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.show()


def Exploration1():
    runs = 10
    episodes = 500
    n_tiles = 8
    alphas = [0.5, 0.3]
    n_steps = [1, 8]
    
    mountainCars = [MountainCar(n_step=n_steps[idx]) for idx in range(len(n_steps))]
    
    steps = np.zeros((len(alphas), episodes))
    for run in range(runs):
        vfs = [FunctionApproximator(n_tiles=n_tiles, alpha=alpha) for alpha in alphas]
        for idx, v in enumerate(vfs):
            mountainCars[idx].set_agent(v)
        for idx in range(len(vfs)):
            for episode in tqdm(range(episodes)):
                step = mountainCars[idx].sarsa()
                steps[idx, episode] += step

    steps /= runs

    for i in range(0, len(alphas)):
        plt.plot(steps[i], label='n = %.01f' % (n_steps[i]))
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()

    #plt.show()
    plt.savefig('../images/figure_10_3.png')
    plt.close()

def Exploration2():
    alphas = np.arange(0.25, 1.75, 0.25)
    n_steps = np.power(2, np.arange(0, 5))
    episodes = 50
    runs = 5
    max_steps = 300
    
    steps = np.zeros((len(n_steps), len(alphas)))
    for run in range(runs):
        for n_step_index, n_step in enumerate(n_steps):
            for alpha_index, alpha in enumerate(alphas):
                if (n_step == 8 and alpha > 1) or \
                        (n_step == 16 and alpha > 0.75):
                    steps[n_step_index, alpha_index] += max_steps * episodes
                    continue
                mountainCar = MountainCar(n_step=n_step)
                agent = FunctionApproximator(n_tiles=8, alpha=alpha)
                mountainCar.set_agent(agent)
                for episode in tqdm(range(episodes)):
                    step = mountainCar.sarsa()
                    steps[n_step_index, alpha_index] += step
    steps /= runs * episodes

    for i in range(0, len(n_steps)):
        plt.plot(alphas, steps[i, :], label='n = '+str(n_steps[i]))
    plt.xlabel('alpha * number of tilings(8)')
    plt.ylabel('Steps per episode')
    plt.ylim([220, max_steps])
    plt.legend()

    plt.savefig('../images/figure_10_4.png')
    plt.close()

if __name__ == '__main__':
    Exploration1()

