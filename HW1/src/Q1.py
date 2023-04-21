import numpy as np

def print_val(val):
	for v in val:
		print(f'{v:>5.2f}', end=' ')
	print()

GAMMAS = [1]#, 0.9, 1]
states = {0:'sleep',
		  1:'pass',
		  2:'facebook',
		  3:'pub',
		  4:'class 1',
		  5:'class 2',
		  6:'class 3'}

with open('./rewards.txt', 'r') as f: # R
	reward = f.readlines()

reward = np.array(reward, dtype=np.float64)

with open('./policy.txt', 'r') as f: # P
	lines = f.readlines()

probs = []

# Value

for line in lines:
	tmp = line.strip().split(' ')
	probs.append(np.array(tmp, dtype=np.float64))

probs = np.array(probs)

I = np.eye(7)
for GAMMA in GAMMAS:
	inv_mat = np.linalg.inv(I-GAMMA*probs) # (I-gamma*P)
	values = np.matmul(inv_mat,reward)    # (I-gamma*P)^-1 x R
	print(f'GAMMA: {GAMMA}')
	for idx, val in enumerate(values):
		print(f'{states[idx]:>8}: {val:<3.2f}')

	print()
	print()

	# action values
	action_value = np.zeros(probs.shape)
	for idx, state in enumerate(states.keys()):
		for i in range(len(list(states.keys()))):
			if probs[idx][i] != 0:
				action_value[idx][i] = reward[idx] + GAMMA*probs[idx][i]*values[i]
	for idx, vec in enumerate(action_value):
		print(f'{states[idx]:>10}:  ', end='')
		for element in vec:
			print(f'{element:^6.2f} ', end='')
		print()

	print()
	print()

	state_values 	 = np.zeros(values.shape)	
	i = 0
	while True:
		print(f'Iteration {i:>02}: ', end='')
		print_val(state_values)
		new_state_values = reward + np.matmul((GAMMA*probs), state_values)
		i += 1
		if np.power(np.power(state_values-new_state_values, 2), 0.5).sum() < 1e-3:
			state_values = new_state_values
			break
		state_values = new_state_values
		
	print(f'Iteration {i:>02}: ', end='')
	print_val(state_values)
	print()