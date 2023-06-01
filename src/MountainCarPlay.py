import gymnasium as gym
import pickle
import numpy as np

class Agent:
    def __init__(self, env, weights):
        self.env: gym.Env = env
        self.weights = weights

    def get_action(self, state):
        action = np.dot(self.weights, state)
        action = np.tanh(action)  # squash outputs to be between -1 and 1
        action = int((action + 1) * 1.5)  # scale to 0, 1, or 2
        return action

def play_game(env, agent):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = agent.get_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f'Total reward: {total_reward}')

def main():
    env = gym.make('MountainCar-v0', render_mode='human')

    # Load the weights from the file
    with open('best_agent_mc.pkl', 'rb') as f:
        weights = pickle.load(f)

    agent = Agent(env, weights)

    play_game(env, agent)
    
    env.close()

if __name__ == "__main__":
    main()
