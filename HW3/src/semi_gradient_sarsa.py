import numpy as np
from FunctionApproximator import *

class Agent:
    def __init__(self, value_fucnction, n):
        self.n = n
        self.value_fucntion = value_fucnction
        
    def 
    
def semi_gradient_n_step_sarsa(value_function, n=1):
    # start at a random position around the bottom of the valley
    current_position = np.random.uniform(-0.6, -0.4)
    # initial velocity is 0
    current_velocity = 0.0
    # get initial action
    current_action = get_action(current_position, current_velocity, value_function)

    # track previous position, velocity, action and reward
    positions = [current_position]
    velocities = [current_velocity]
    actions = [current_action]
    rewards = [0.0]

    # track the time
    time = 0

    # the length of this episode
    T = float('inf')
    while True:
        # go to next time step
        time += 1

        if time < T:
            # take current action and go to the new state
            new_position, new_velocity, reward = step(current_position, current_velocity, current_action)
            # choose new action
            new_action = get_action(new_position, new_velocity, value_function)

            # track new state and action
            positions.append(new_position)
            velocities.append(new_velocity)
            actions.append(new_action)
            rewards.append(reward)

            if new_position == POSITION_MAX:
                T = time

        # get the time of the state to update
        update_time = time - n
        if update_time >= 0:
            returns = 0.0
            # calculate corresponding rewards
            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += rewards[t]
            # add estimated state action value to the return
            if update_time + n <= T:
                returns += value_function.value(positions[update_time + n],
                                                velocities[update_time + n],
                                                actions[update_time + n])
            # update the state value function
            if positions[update_time] != POSITION_MAX:
                value_function.learn(positions[update_time], velocities[update_time], actions[update_time], returns)
        if update_time == T - 1:
            break
        current_position = new_position
        current_velocity = new_velocity
        current_action = new_action

    return time
