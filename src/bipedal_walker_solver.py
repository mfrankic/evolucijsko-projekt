import gymnasium as gym
import numpy as np
import pickle
import multiprocessing as mp

from es import OpenES

class BipedalWalkerSolver:
    def __init__(self, num_params):
        self.es = OpenES(num_params)

    def get_action(self, params, state):
        """Compute action using a simple linear policy."""
        params_reshaped = params.reshape(4, 24)
        return np.tanh(np.matmul(params_reshaped, state))  # returns a 4-dimensional vector

    def get_reward(self, params):
        """Run one episode with the given parameters and return the total reward."""
        env = gym.make('BipedalWalker-v3', render_mode='human')
        state, _ = env.reset()
        total_reward = 0.0
        no_change_counter = 0
        previous_state = state
        
        for _ in range(2000):  # Run for a maximum of 2000 steps
            action = self.get_action(params, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Stop the simulation if agent is stuck
            if np.allclose(previous_state, state, atol=1e-4):
                no_change_counter += 1
                if no_change_counter > 50:
                    break
            else:
                no_change_counter = 0

            previous_state = state
            
            if done:
                break
        return total_reward

    def train(self, num_iterations):
        """Train the policy for the given number of iterations."""
        with mp.Pool(1) as pool:
            for iter in range(num_iterations):
                params_list = self.es.ask()
                reward_list = pool.map(self.get_reward, params_list)
                self.es.tell(reward_list)
                if (iter + 1) % 2 == 0:
                    print(f'Iteration: {iter + 1}, Reward: {max(reward_list)}')
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
            
    def play(self, params):
        """Use the trained policy to play the game."""
        env = gym.make('BipedalWalker-v3', render_mode='human')
        state, _ = env.reset()
        
        total_reward = 0
        for _ in range(2000):
            action = self.get_action(params, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
              
        print(f'Total reward: {total_reward}')
        env.close()
