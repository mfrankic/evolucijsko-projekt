import gym
import numpy as np

#env = gym.make("MountainCar-v0", render_mode='human')
env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 100

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

filename_q_table = "q_table_mc.npy"
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_table(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(EPISODES):
    done = False
    obs, info = env.reset()
    discrete_state = get_discrete_table(obs)

    # if episode % SHOW_EVERY == 0:
    #    print(episode)

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        new_discrete_state = get_discrete_table(observation)

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])

            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[discrete_state + (action, )] = new_q

        elif observation[0] >= env.goal_position:
            #print(f"CILJ {episode}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

np.save(filename_q_table, q_table)
