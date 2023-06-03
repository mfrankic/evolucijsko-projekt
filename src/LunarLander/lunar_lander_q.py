import gymnasium as gym
import collections
import numpy as np
import random

env = gym.make("LunarLander-v2")
env.reset()

EPISODES = 50000
GAMMA = 0.99
LEARNING_RATE = 0.1
MIN_EPSILON = 0.01
DECAY_EPSILON = 0.996

PRINT_FREQUENCY = 100

q_states = collections.defaultdict(float)   # note that the first insertion of a key initializes its value to 0.0
return_per_ep = [0.0]
epsilon = 1.0
num_actions = env.action_space.n

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

def epsilon_greedy(q_func, state, eps, env_actions):
    prob = np.random.random()

    if prob < eps:
        return random.choice(range(env_actions))
    else:
        qvals = [q_func[state + (action, )] for action in range(env_actions)]
        return np.argmax(qvals)
      
def greedy(qstates_dict, state, env_actions):
    qvals = [qstates_dict[state + (action, )] for action in range(env_actions)]
    return max(qvals)

def decay_epsilon(curr_eps, exploration_final_eps):
    if curr_eps < exploration_final_eps:
        return curr_eps
    
    return curr_eps * DECAY_EPSILON

for i in range(EPISODES):
    t = 0

    # Initial episode state: S
    curr_state = discretize_state(env.reset()[0])
    
    while True:
        action = epsilon_greedy(q_states, curr_state, epsilon, num_actions)

        qstate = curr_state + (action, )

        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(observation)

        # Policy evaluation step
        if not done:
            q_states[qstate] += LEARNING_RATE * (reward + GAMMA * greedy(q_states, next_state, num_actions) - q_states[qstate])
        else:
            q_states[qstate] += LEARNING_RATE * (reward - q_states[qstate])

        return_per_ep[-1] += reward

        if done:
            if (i + 1) % PRINT_FREQUENCY == 0:
                print("\nEpisode finished after {} timesteps".format(t + 1))
                print("Episode {}: Total Return = {}".format(i + 1, return_per_ep[-1]))
                print("Total keys in q_states dictionary = {}".format(len(q_states)))

            if (i + 1) % 100 == 0:
                mean_100ep_reward = round(np.mean(return_per_ep[-101:-1]), 1)
                print("Last 100 episodes mean reward: {}".format(mean_100ep_reward))

            epsilon = decay_epsilon(epsilon, MIN_EPSILON)
            return_per_ep.append(0.0)

            break

        curr_state = next_state
        t += 1
    
# save q_states dictionary to file
np.save("q_states.npy", q_states)
