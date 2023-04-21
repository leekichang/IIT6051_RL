import os
import utils
import matplotlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from RandomWalk import RandomWalk

np.random.seed(2022314416)
IMG_PATH = '../images/Q2'
os.makedirs(IMG_PATH, exist_ok=True)

if __name__ == '__main__':
    steps = np.power(2, np.arange(0, 10)) # steps = 1~512
    alphas = np.arange(0, 1.1, 0.1)       # alpha = 0~1, 0.1 간격
    episodes = 10                         # 10 episode의 평균
    runs = 10                            # 100번 반복
    errors = np.zeros((len(steps), len(alphas))) # errors 저장할 buffer
    N_STATE = 19
    for step_ind, step in enumerate(tqdm(steps)):
        rw = RandomWalk(N=N_STATE, n_step=step, is_Q1=True)
        for alpha_ind, alpha in enumerate(alphas):
            for run in range(runs):
                value = np.copy(rw.init_values)#np.zeros(N_STATE+2)
                for ep in range(0, episodes):
                    rw.TD(value, n=step, alpha=alpha)
                    # calculate the RMS error
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(\
                        np.power(value - rw.TRUE_VALUE, 2)) / rw.N)
    errors /= episodes * runs
    for i in range(0, len(steps)):
        plt.plot(alphas, errors[i, :], label=f'n = {steps[i]}')
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.22, 0.40])
    plt.legend()
    plt.savefig(f'{IMG_PATH}/Q2_w1.png')
    plt.close()