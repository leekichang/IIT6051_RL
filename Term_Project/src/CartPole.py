import gym

class CartPole:
    def __init__(self, is_show=False):
        self.is_show = is_show
        if is_show:
            self.env = gym.make('CartPole-v1', render_mode = 'human')
        else:
            self.env = gym.make('CartPole-v1', render_mode = None)
    
        
if __name__ == '__main__':
    CP = CartPole(is_show=True)
    CP.env.reset()
    
    
    
# import gym
# import numpy as np

# # Create the CartPole environment
# env = gym.make('CartPole-v1', render_mode='human')

# # Set the hyperparameters
# learning_rate = 0.01
# discount_factor = 0.99
# num_episodes = 1000

# # Initialize the weight matrix
# num_states = env.observation_space.shape[0]
# num_actions = env.action_space.n
# weights = np.random.rand(num_states, num_actions)

# # Function to choose an action based on the policy
# def choose_action(state):
#     action_probs = softmax(np.dot(state, weights))
#     action = np.random.choice(range(num_actions), p=action_probs)
#     return action

# # Function to update the weights based on the policy gradient update rule
# def update_weights(states, actions, rewards):
#     total_rewards = np.sum(rewards)
#     for t in range(len(states)):
#         state = states[t]
#         action = actions[t]
#         gradient = np.reshape(state, (num_states, 1))
#         gradient[action] -= 1
#         weights[:, action] += learning_rate * gradient * total_rewards

# # Softmax function for action selection
# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()

# # Training loop
# for episode in range(num_episodes):
#     state = env.reset()
#     done = False
#     states, actions, rewards = [], [], []

#     while not done:
#         action = choose_action(state)
#         next_state, reward, done, _ = env.step(action)
#         states.append(state)
#         actions.append(action)
#         rewards.append(reward)
#         state = next_state

#     update_weights(states, actions, rewards)

#     if episode % 100 == 0:
#         print("Episode:", episode)

# # Evaluate the learned policy
# num_eval_episodes = 100
# total_rewards = 0

# for _ in range(num_eval_episodes):
#     state = env.reset()
#     done = False

#     while not done:
#         action = choose_action(state)
#         state, reward, done, _ = env.step(action)
#         total_rewards += reward

# average_reward = total_rewards / num_eval_episodes
# print("Average reward:", average_reward)
