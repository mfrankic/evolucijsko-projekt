import gymnasium as gym
import numpy as np

# Load the saved q_states dictionary
q_states = np.load("same_seed/q_states.npy", allow_pickle=True).item() 

# Define a function for discretizing states similar to the training script
def discretize_state(state):
    discrete_state = (min(2, max(-2, int((state[0]) / 0.05))), \
                        min(2, max(-2, int((state[1]) / 0.1))), \
                        min(2, max(-2, int((state[2]) / 0.1))), \
                        min(2, max(-2, int((state[3]) / 0.1))), \
                        min(2, max(-2, int((state[4]) / 0.1))), \
                        min(2, max(-2, int((state[5]) / 0.1))), \
                        int(state[6]), \
                        int(state[7]))
    return discrete_state

# Define a function to select the best action for a given state
def best_action(qstates_dict, state, env_actions):
    qvals = [qstates_dict[state + (action, )] for action in range(env_actions)]
    return np.argmax(qvals)

env = gym.make("LunarLander-v2")
env.reset(seed=35)

# This is the maximum number of timesteps per episode. Set it as per your requirements.
MAX_TIMESTEPS_PER_EPISODE = 1000

# Loop over episodes
for episode in range(1000):
    # Reset the environment and initialize variables
    state = discretize_state(env.reset(seed=35)[0])
    total_reward = 0

    for t in range(MAX_TIMESTEPS_PER_EPISODE):
        # Choose the action with the highest Q-value for the current state
        action = best_action(q_states, state, env.action_space.n)

        # Execute the action and get the next state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(next_state)

        total_reward += reward

        # If the episode is finished, we leave the for loop
        if done:
            break

        state = next_state
    
    # append the score for the episode to a file
    with open('../../data/lunar_lander_q_scores.csv', 'a') as f:
        f.write(f'{episode + 1},{total_reward}\n')

    print("\nEpisode {} finished after {} timesteps".format(episode + 1, t + 1))
    print("Episode {}: Total Reward = {}".format(episode + 1, total_reward))

env.close()
