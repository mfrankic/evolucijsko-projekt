import numpy as np
import gymnasium as gym
import pickle
import multiprocessing as mp
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from es import OpenES

class LunarLanderSolver:
    def __init__(self, num_params):
        self.es = OpenES(num_params)
        self.wind = False

    def get_action(self, params, state):
        """Compute action using a simple linear policy."""
        params = np.reshape(params, (4, 8))
        return np.argmax(np.matmul(params, state))  # action is now chosen from a discrete set of 4 actions.

    def get_reward(self, params):
        """Run one episode with the given parameters and return the total reward."""
        env = gym.make('LunarLander-v2', enable_wind=self.wind)
        state, _ = env.reset()
        total_reward = 0.0
        for _ in range(1000):  # Run for a maximum of 1000 steps
            action = self.get_action(params, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
        return total_reward

    def train(self, num_iterations, wind=False):
        """Train the policy for the given number of iterations."""
        self.wind = wind
        
        with mp.Pool() as pool:
            for iter in range(num_iterations):
                params_list = self.es.ask()
                reward_list = pool.map(self.get_reward, params_list)
                self.es.tell(reward_list)
                if (iter + 1) % 10 == 0:
                    print(f'Iteration: {iter + 1}, Reward: {np.mean(reward_list)}')
                    # append the iteration number and reward to a csv file
                    with open('../../data/lunar_lander_iteration_reward.csv', 'a') as f:
                        f.write(f'{iter + 1},{np.mean(reward_list)}\n')
        return self.es.result()

    def save_weights(self, filename):
        """Save the best parameters to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.es.best_param(), f)

    def load_weights(self, filename):
        """Load the best parameters from a file."""
        with open(filename, 'rb') as f:
            self.es.set_mu(pickle.load(f))
        return self.es.mu
            
    def play(self, params, render=True, wind=False):
        """Use the trained policy to play the game."""
        if render:
            env = gym.make('LunarLander-v2', render_mode='human', enable_wind=wind)
        else:
            env = gym.make('LunarLander-v2', enable_wind=wind)

        state, _ = env.reset()
        
        total_reward = 0
        for _ in range(1000):
            action = self.get_action(params, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
              
        print(f'Total reward: {total_reward}')
        env.close()
        return total_reward
