import numpy as np
N_TILES      = 8
EPS          = 0.0
ALPHAS       = np.array([0.1, 0.2, 0.5])/8
GAMMA        = 1
THRESHOLD    = 1e-6
POSITION_MAX = 0.6
EPISODES     = 9000
SAVE_EPISODE = [10, 50, 100, 500, 1000, 5000, 9000]
SAVE_PATH    = '../saved_weights'

ACTIONS   = {0:"LEFT",
             1:"NOTHING",
             2:"RIGHT"}