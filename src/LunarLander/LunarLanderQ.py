import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v2")
env.reset()

def print_observation_space(env):
    print(f"Observation space high: {env.observation_space.high}")
    print(f"Observation space low: {env.observation_space.low}")
    print(f"Number of actions in the action space: {env.action_space.n}")

DISCRETE_OS_SIZE = [12] * 8
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    discrete_state = np.minimum(discrete_state, np.array(DISCRETE_OS_SIZE) - 1)
    return tuple(discrete_state.astype(int))

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

LEARNING_RATE = 0.5
DISCOUNT = 0.95
EPISODES = 40000
LOG_FREQUENCY = 100
epsilon = 0.1
START_DECAY = 1
END_DECAY = EPISODES // 1.5
epsilon_decay_by = epsilon / (END_DECAY - START_DECAY)

total_reward = 0
for episode in range(EPISODES):
    if (episode + 1) % LOG_FREQUENCY == 0:
        print(f"Episode {episode + 1}")
        print(f"Total reward: {total_reward}")

    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False

    total_reward = 0
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        new_discrete_state = get_discrete_state(new_state)
        total_reward += reward

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        elif reward == 100:
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    if END_DECAY >= episode >= START_DECAY:
        epsilon -= epsilon_decay_by

np.save("q_table.npy", q_table)
