import numpy as np
import gymnasium as gym

# Initialize your environment
env = gym.make("CartPole-v1", render_mode='human')

DISCRETE_OS_SIZE = [50, 50, 100, 100] #our dimensions
real_observation_space = np.array([env.observation_space.high[0], env.observation_space.high[1], env.observation_space.high[2], 3.5]) #disregarding cart data
discrete_os_win_size = (real_observation_space * 2 / DISCRETE_OS_SIZE) #step-size inside our discrete observation space
# Define your function to get discrete state
# def get_discrete_state(state):
#     trimmed_state = np.array([state[2], state[3]])
#     discrete_state = (trimmed_state + real_observation_space) / discrete_os_win_size
#     return tuple(discrete_state.astype(int))

def get_discrete_state(state):
    discrete_state = (state + real_observation_space) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

# Load the saved q_table from a file
q_table = np.load('q_table.npy')

# Run some episodes using the learned Q-table
for episode in range(1):  # Let's play 5 episodes
    done = False
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    
    total_reward = 0
    while not done:
        # Choose action according to the learned policy
        action = np.argmax(q_table[discrete_state])

        # Perform the action
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        new_discrete_state = get_discrete_state(new_state)
        
        total_reward += reward

        # Render the environment to visualize the playing
        env.render()

        # Update the current state
        discrete_state = new_discrete_state
        
    print(f'Episode: {episode+1}, Total reward: {total_reward}')
