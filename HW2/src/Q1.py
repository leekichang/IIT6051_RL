import os
import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from RandomWalk import RandomWalk
np.random.seed(2022314416)
IMG_PATH = '../images/Q1'
os.makedirs(IMG_PATH, exist_ok=True)

def compute_state_value(rw):
    episodes = [0, 1, 10, 25, 50, 100]
    current_values = np.copy(rw.init_values)
    plt.figure(1)
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot([f"{chr(65+i)}" for i in range(rw.N)], current_values[1:rw.N+1], label=f"{str(i)} Episodes")
            print(f"i:{i}, curr_val:{current_values}")
            print(current_values)
        rw.TD(value=current_values)

    plt.plot([f"{chr(65+i)}" for i in range(rw.N)], rw.TRUE_VALUE[1:rw.N+1], 'r--',label='true values')
    plt.xlabel('State')
    plt.ylabel('Estimated Value')
    plt.legend()

def rms_error(rw):
    # Same alpha value can appear in both arrays
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)
        if i < len(td_alphas):
            method = 'TD'
            linestyle = 'solid'
        else:
            method = 'MC'
            linestyle = 'dashdot'
        for r in tqdm(range(runs)):
            errors = []
            current_values = np.copy(rw.init_values)
            for i in range(0, episodes):
                errors.append(np.sqrt(np.sum(np.power(rw.TRUE_VALUE - current_values, 2)) / 5.0))
                if method == 'TD':
                    rw.TD(value=current_values, n=rw.n_step, alpha=alpha)
                else:
                    rw.MonteCarlo(values=current_values,alpha=alpha)
            total_errors += np.asarray(errors)
        total_errors /= runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', $\\alpha$ = %.02f' % (alpha))
    plt.xlabel('Walks/Episodes')
    plt.ylabel('Empirical RMS error, averaged over states')
    plt.legend()

if __name__ == '__main__':
    args = utils.parse_args()
    rw = RandomWalk(N=args.n_state, n_step=args.n_step, GAMMA=1)
    plt.figure()
    compute_state_value(rw)
    plt.savefig(f'{IMG_PATH}/Q1_{args.n_state}state_{args.n_step}step_val.png')
    plt.close()
    plt.figure()
    rms_error(rw)
    plt.savefig(f'{IMG_PATH}/Q1_{args.n_state}state_{args.n_step}step_rms.png')
    plt.close()
