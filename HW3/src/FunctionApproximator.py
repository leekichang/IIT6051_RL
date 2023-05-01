import random
import numpy as np
from IHT import *
from config import *

np.random.seed(2022314416)

class FunctionApproximator:
    def __init__(self,
                 n_tiles   = N_TILES,
                 eps       = EPS,
                 alpha     = ALPHAS[0],
                 gamma     = GAMMA,
                 threshold = THRESHOLD):
        
        self.n_tiles    = n_tiles
        self.dim        = n_tiles ** 4
        self.hash_table = IHT(sizeval=self.dim)
        self.weight     = np.zeros(self.dim) # initialize with zeros
        self.eps        = eps
        self.alpha      = alpha / self.n_tiles
        self.gamma      = gamma
        self.threshold  = threshold
        pass
    
    def set_len(self, lens):
        self.lens = lens
    
    def set_weight(self, weight):
        self.weight = weight

    def getActiveTiles(self, state, action):
        scales = self.n_tiles / np.array(self.lens)
        
        active_tiles = tiles(self.hash_table, self.n_tiles,
                         [scales[0]*state[0], scales[1]*state[1]],
                         [action])
        return active_tiles
    
    def getValue(self, state, action):
        if state[0] == POSITION_MAX:
            return 0
        active_tiles = self.getActiveTiles(state, action)
        return np.sum(self.weight[active_tiles])

    def learn(self, state, action, target):
        active_tiles = self.getActiveTiles(state, action)
        estimation   = np.sum(self.weight[active_tiles])
        delta        = self.alpha * (target - estimation)
        for tile in active_tiles:
            self.weight[tile] += delta    
    
    def calcValues(self, state):
        result = []
        for action in list(ACTIONS.keys()):
            result.append(self.getValue(state, action))
        return result
    
    def getAction(self, state):
        """        
        Epsilon-Greedy Policy
        """
        if np.random.binomial(1, self.eps) == 1:
            return np.random.choice(list(ACTIONS.keys()))
        values = self.calcValues(state)
        action = np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])
        return action
        
    def cost_to_go(self, state):
        costs = []
        for action in list(ACTIONS.keys()):
            costs.append(self.getValue(state, action))
        return -np.max(costs)
    
    def cost_to_go_policy(self, state):
        costs = []
        for action in list(ACTIONS.keys()):
            costs.append(self.getValue(state, action))
        return np.argmax(costs)

    def save_weight(self, n_steps, episode, is_show):
        np.save(f'{SAVE_PATH}/{self.alpha}_{self.n_tiles}_{n_steps}_{episode}_{is_show}.npy', self.weight)

def get_rms(data):
    return np.sqrt(np.mean(data**2))

if __name__ == '__main__':
    pass
    