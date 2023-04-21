import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2022314416)
IMG_PATH = "../images/Q3"
os.makedirs(IMG_PATH, exist_ok=True)

class WindyGridWorld:
    def __init__(self,
                 H,
                 W,
                 WIND,
                 eps=0.1,
                 alpha=0.5,
                 reward=-1.0,
                 init_state=[3,0],
                 goal=[3,7]
                 ):
        self.H, self.W   = H,W
        self.wind        = WIND
        self.actions     = {"UP":0,"DOWN":1,"LEFT":2,"RIGHT":3}
        self.actions_lst = [0, 1, 2, 3]
        self.eps         = eps
        self.alpha       = alpha
        self.reward      = reward
        self.init_state  = init_state
        self.goal        = goal

    def move(self, state, action):
        i, j = state
        if action == self.actions["UP"]:
            return [max(i-1-self.wind[j], 0), j]
        elif action == self.actions["DOWN"]:
            return [max(min(i+1-self.wind[j], self.H - 1), 0), j]
        elif action == self.actions["LEFT"]:
            return [max(i - self.wind[j], 0), max(j - 1, 0)]
        elif action == self.actions["RIGHT"]:
            return [max(i - self.wind[j], 0), min(j + 1, self.W - 1)]
    
    def episode(self, q_value):
        time = 0
        state = self.init_state
        if np.random.binomial(1, self.eps) == 1:
            action = np.random.choice(self.actions_lst)
        else:
            values_ = q_value[state[0], state[1], :]
            action  = np.random.choice([action_ for action_, value_ \
                                        in enumerate(values_)\
                                        if value_ == np.max(values_)])
        
        while state != self.goal:
            next_state = self.move(state, action)
            if np.random.binomial(1, self.eps) == 1:
                next_action = np.random.choice(self.actions_lst)
            else:
                values_ = q_value[next_state[0], next_state[1], :]
                next_action = np.random.choice([action_ for action_,\
                                                value_ in enumerate(values_)\
                                                 if value_ == np.max(values_)])
            q_value[state[0], state[1], action] += \
                self.alpha * (self.reward + q_value[next_state[0],\
                            next_state[1], next_action] -\
                            q_value[state[0], state[1], action])
            state = next_state
            action = next_action
            time += 1
        return time

if __name__ == '__main__':
    H, W = 7, 10
    WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    ww = WindyGridWorld(H, W, WIND)
    q_value = np.zeros((H, W, 4))
    episode_limit = 170
    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(ww.episode(q_value))
        # time = episode(q_value)
        # episodes.extend([ep] * time)
        ep += 1

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig(f'{IMG_PATH}/Q3.png')
    plt.close()

    # display the optimal policy
    # optimal_policy = []
    # for i in range(0, H):
    #     optimal_policy.append([])
    #     for j in range(0, W):
    #         if [i, j] == ww.goal:
    #             optimal_policy[-1].append('G')
    #             continue
    #         bestAction = np.argmax(q_value[i, j, :])
    #         if bestAction == ww.actions["UP"]:
    #             optimal_policy[-1].append('U')
    #         elif bestAction == ww.actions["DOWN"]:
    #             optimal_policy[-1].append('D')
    #         elif bestAction == ww.actions["LEFT"]:
    #             optimal_policy[-1].append('L')
    #         elif bestAction == ww.actions["RIGHT"]:
    #             optimal_policy[-1].append('R')
    # print('Optimal policy is:')
    # for row in optimal_policy:
    #     print(row)
    # print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))

    