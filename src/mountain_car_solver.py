import numpy as np
import gymnasium as gym
import pickle
import multiprocessing as mp

from es import OpenES

class MountainCarSolver:
    def __init__(self, num_params):
        self.es = OpenES(num_params, sigma_init=0.5, learning_rate=0.03, learning_rate_decay=0.999, forget_best=False)
        
    def get_action(self, params, state):
        """Compute action using a simple linear policy."""
        action = np.matmul(params, state)
        if action < -0.5:
            return 0  # push left
        elif action > 0.5:
            return 2  # push right
        else:
            return 1  # do nothing

    def get_reward(self, params):
        """Run one episode with the given parameters and return the total reward."""
        env = gym.make('MountainCar-v0')
        state, _ = env.reset()
        total_reward = 0.0
        for _ in range(200):  # Run for a maximum of 200 steps
            action = self.get_action(params, state)
            state, reward, terminated, truncated, _ = env.step(action)
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
                reward_list = pool.map(self.get_reward, params_list)
                self.es.tell(reward_list)
                if (iter + 1) % 100 == 0:
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
            
    def play(self, params):
        """Use the trained policy to play the game."""
        env = gym.make('MountainCar-v0', render_mode='human')
        state, _ = env.reset()
        
        total_reward = 0
        for _ in range(200):
            action = self.get_action(params, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
              
        print(f'Total reward: {total_reward}')
        env.close()
