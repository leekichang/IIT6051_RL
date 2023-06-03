import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    os.makedirs('./figures', exist_ok=True)
    models = ['q_learning', 'REINFORCE', 'DQN']
    #### BEST RESULT ####
    FIGSIZE=(6,3)
    plt.figure(figsize=FIGSIZE)
    for model in models:
        e = []
        rs = []
        for i in range(10):
            rs.append(np.load(f'./results/{model}_{i}.npy'))
        epis = [len(epi) for epi in rs]
        max_epi = max(epis)
        min_epi = min(epis)
        episodes = np.arange(max_epi)
        q_learning = np.zeros(max_epi)
        r = np.load(f'./results/{model}_{epis.index(min_epi)}.npy')
        plt.plot(r, label=f'{model}')
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'./figures/best_scores.png')
    plt.close()
    
    plt.figure(figsize=FIGSIZE)
    for model in models:
        e = []
        rs = []
        for i in range(10):
            rs.append(np.load(f'./results/{model}_{i}.npy'))
        epis = [len(epi) for epi in rs]
        max_epi = max(epis)
        min_epi = min(epis)
        episodes = np.arange(max_epi)
        q_learning = np.zeros(max_epi)
        r = np.load(f'./results/{model}_{epis.index(max_epi)}.npy')
        plt.plot(r, label=f'{model}')
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'./figures/worst_scores.png')
    plt.close()
    
    plt.figure(figsize=FIGSIZE)
    for model in models:
        e = []
        rs = []
        for i in range(9):
            rs.append(np.load(f'./results/{model}_{i}.npy'))
        epis = [len(epi) for epi in rs]
        max_epi = max(epis)
        min_epi = min(epis)
        med_epi = np.median(epis)
        episodes = np.arange(max_epi)
        q_learning = np.zeros(max_epi)
        r = np.load(f'./results/{model}_{epis.index(med_epi)}.npy')
        plt.plot(r, label=f'{model}')
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'./figures/median_scores.png')
    plt.close()
    
    plt.figure(figsize=FIGSIZE)
    for model in models:
        e = []
        rs = []
        for i in range(10):
            rs.append(np.load(f'./results/{model}_{i}.npy'))
        epis = [len(epi) for epi in rs]
        max_epi = max(epis)
        episodes = np.arange(max_epi)
        q_learning = np.zeros(max_epi)
        for r in rs:
            q_learning = q_learning + np.concatenate([r,np.zeros(max_epi-len(r))])/10
        plt.plot(q_learning, label=f'{model}')
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'./figures/avg_scores.png')
    plt.close()
    
    result_dict = {model:{'max':0, 'min':0, 'med':0, 'avg':0, 'std':0} for model in models}
    for model in models:
        rs = []
        for i in range(10):
            rs.append(np.load(f'./results/{model}_{i}.npy'))
        epis = [len(epi) for epi in rs]
        max_epi = max(epis)
        min_epi = min(epis)
        med_epi = np.median(epis)
        avg_epi = np.average(epis)
        std     = np.std(epis)
        result_dict[model]['max'] = max_epi
        result_dict[model]['min'] = min_epi
        result_dict[model]['med'] = med_epi
        result_dict[model]['avg'] = avg_epi
        result_dict[model]['std'] = std
    print(result_dict)
    
    
    for model in models:
        e = []
        rs, ts = [], []
        for i in range(10):
            rs.append(len(np.load(f'./results/{model}_{i}.npy')))
        ts = np.load(f'./results/{model}_time.npy')
        #print(np.shape(ts), np.shape(rs))
        print(model)
        print(f'{np.average(np.array(ts)/np.array(rs)) * 1000:.2f} msec')
        print(f'{np.std(np.array(ts)/np.array(rs)):.4f}')
        print()
            
    plt.figure(figsize=FIGSIZE)
    mem_sizes = [1000, 2000, 3000, 4000, 5000]
    for mem in mem_sizes:
        a = np.load(f'./results/DQN_mean_memsize{mem}.npy')
        plt.plot(a, label=f'{mem}', linewidth=2)
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'./figures/avg_scores_memsize.png')
    plt.close()
    
    plt.figure(figsize=FIGSIZE)
    mem_sizes = [12, 24, 36, 48, 60]
    for mem in mem_sizes:
        a = np.load(f'./results/DQN_mean_hidden_{mem}.npy')
        plt.plot(a, label=f'{mem}', linewidth=2)
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'./figures/avg_scores_hidden.png')
    plt.close()