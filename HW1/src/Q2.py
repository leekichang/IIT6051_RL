import numpy as np

def print_val(val):
	for v in val:
		print(f'{v:>5.2f}', end=' ')
	print()
	
#코너에 있는 state들은 self-edge가 있는 case로 볼 수 있음
with open('./probs.csv', 'r', encoding='utf-8') as f:
	lines = f.readlines()

probs = np.ndarray((16, 16))

for idx, line in enumerate(lines):
	for jdx, p in enumerate(line.strip().split(',')):
		probs[idx][jdx] = float(p)

reward = -0.1*np.ones(16)
reward[3] =  1
reward[7] = -1

GAMMA = 1.0

I = np.eye(16)

inv_mat = np.linalg.inv(I-GAMMA*probs)
values = np.matmul(inv_mat, reward)
#for idx, val in enumerate(values):
#	print(f'{val:<3.2f}')
print(values.reshape(4, 4))

i = 0
state_values = np.zeros(reward.shape)
state_values[3] += 1
state_values[7] += -1
while True:
    if i==0 or i == 1 or i == 5 or i == 10:
        print(f'Iteration {i:>02}: ')
        print(state_values.reshape(4, 4))
        print()
    
    # print(np.broadcast_to(reward, (16,16)).transpose())
    # print(probs)
    mask = np.ones((16,16))
    mask[probs==0] = 0
    print(probs.shape, probs[mask==1].shape)
    print(probs[mask==1])
    # print(np.multiply(np.broadcast_to(reward, (16,16)), mask) + np.multiply((GAMMA*probs), np.broadcast_to(state_values, (16,16))))
    # print((np.multiply(np.broadcast_to(reward, (16,16)), mask) + np.multiply((GAMMA*probs), np.broadcast_to(state_values, (16,16))))[mask==1].shape)
    new_state_values = np.max((np.multiply(np.broadcast_to(reward, (16,16)), mask) + np.multiply((GAMMA*probs), np.broadcast_to(state_values, (16,16))))[mask==1], axis=1)
    
    i += 1
    if np.abs(state_values-new_state_values).sum() < 1e-3:
	    state_values = new_state_values
	    state_values[3] = 1
	    state_values[7] = -1
	    break
    state_values = new_state_values
    state_values[3] = 1
    state_values[7] = -1
    
    
print(f'Iteration {i:>02}:')
print(state_values.reshape(4, 4))
print()





