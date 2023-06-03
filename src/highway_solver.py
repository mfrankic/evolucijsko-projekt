import gymnasium as gym
import numpy as np
import multiprocessing as mp
import pickle
import warnings

from es import OpenES

class HighwaySolver:
    def __init__(self, num_params):
        self.es = OpenES(num_params, popsize=8)

    def get_action(self, params, state):
        """Compute action using a simple linear policy."""
        q_values = np.matmul(params.reshape(5, 5*5), state.flatten())
        return np.argmax(q_values)

    def get_reward(self, params):
        """Run one episode with the given parameters and return the total reward."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            env = gym.make('highway-v0')
        state, _ = env.reset()
        state = state.flatten()
        total_reward = 0.0
        for _ in range(40):  # Run for a maximum of 500 steps
            action = self.get_action(params, state)
            state, reward, terminated, truncated, _ = env.step(action)
            state = state.flatten()
            done = terminated or truncated
            total_reward += reward
            if done:
                break
        return total_reward

    def train(self, num_iterations):
        """Train the policy for the given number of iterations."""
        with mp.Pool() as pool:
            for iter in range(num_iterations):
                params_list = self.es.ask()
                # Parallelize the computation of rewards
                reward_list = pool.map(self.get_reward, params_list)
                self.es.tell(reward_list)
                if (iter + 1) % 1 == 0:
                    print(f'Iteration: {iter + 1}, Reward: {max(reward_list)}')
        return self.es.result()

    def save_weights(self, filename):
        """Save the weights of the model to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.es.best_param(), f)

    def load_weights(self, filename):
        """Load the weights of the model from a file."""
        with open(filename, 'rb') as f:
            self.es.set_mu(pickle.load(f))
        return self.es.mu
      
    def play(self, params, render=True):
        """Use the trained policy to play the game."""
        if render:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                env = gym.make('highway-v0', render_mode='human')
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                env = gym.make('highway-v0')
            
        state, _ = env.reset()
        
        total_reward = 0
        for _ in range(500):
            action = self.get_action(params, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
              
        print(f'Total reward: {total_reward}')
        env.close()
        return total_reward
