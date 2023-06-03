import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")
env.reset()

DISCRETE_OS_SIZE = [10] * 8
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    discrete_state = np.minimum(discrete_state, np.array(DISCRETE_OS_SIZE) - 1)  # add this line
    return tuple(discrete_state.astype(int))

q_table = np.load("q_table.npy")

for episode in range(1):
    done = False
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    
    total_reward = 0
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        new_discrete_state = get_discrete_state(new_state)
        
        total_reward += reward
        discrete_state = new_discrete_state
        
    print(f'Episode: {episode+1}, Total reward: {total_reward}')
