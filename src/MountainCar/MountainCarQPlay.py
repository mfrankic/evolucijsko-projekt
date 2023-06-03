import gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode='human')

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

filename_q_table = "q_table_mc.npy"

def get_discrete_table(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

q_table = np.load(filename_q_table)

for episode in range(1):
    done = False
    obs, info = env.reset()
    discrete_state = get_discrete_table(obs)

    total_reward = 0

    while not done:
        action = np.argmax(q_table[discrete_state])

        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        new_discrete_state = get_discrete_table(observation)

        total_reward += reward

        env.render()

        discrete_state = new_discrete_state

    print(f'Episode: {episode+1}, Total reward: {total_reward}')