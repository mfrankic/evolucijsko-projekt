import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")
env.reset()

DISCRETE_OS_SIZE = [20, 20, 50, 50]
real_observation_space = np.array([env.observation_space.high[0], env.observation_space.high[1], env.observation_space.high[2], 3.5])
discrete_os_win_size = (real_observation_space * 2 / DISCRETE_OS_SIZE)

def get_discrete_state(state):
    discrete_state = (state + real_observation_space) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

q_table = np.random.uniform(low=0, high=1, size =(DISCRETE_OS_SIZE + [env.action_space.n]))

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000
LOG_FREQUENCY = 500
epsilon = 0.3
START_DECAY = 1
END_DECAY = EPISODES // 1.5
epsilon_decay_by = epsilon / (END_DECAY - START_DECAY)

total_reward = 0
for episode in range(EPISODES):
    #Just some logging info
    if (episode + 1) % LOG_FREQUENCY == 0:
        print(f"Episode {episode + 1}")
        print(f"Total reward: {total_reward}")

    #Resetting the environment as well as getting state 0
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False

    total_reward = 0
    #One iteration of the environment
    while not done:

        #Using epsilon to introduce exploration
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,2)

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        cart_position = new_state[0]
        pole_angle = new_state[2]
        reward = reward - (np.abs(cart_position)/4.8)**2 - (np.abs(pole_angle * 1.2)/0.418)**0.5
        
        new_discrete_state = get_discrete_state(new_state)
        total_reward += reward

        # Adjusting the values in our Q-table according to the Q-learning formula
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        discrete_state = new_discrete_state

    #Decay epsilon
    if END_DECAY >= episode >= START_DECAY:
        epsilon -= epsilon_decay_by
    
np.save("q_table.npy", q_table)
