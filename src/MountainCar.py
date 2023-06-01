import gymnasium as gym
import numpy as np
import multiprocessing as mp
import pickle
import time

# Agent class
class Agent:
    def __init__(self, env):
        self.env: gym.Env = env
        self.weights = np.random.rand(2) * 2 - 1  # MountainCar-v0 has 2 states

    def get_action(self, state):
        action = np.dot(self.weights, state)
        action = np.tanh(action)  # squash outputs to be between -1 and 1
        action = int((action + 1) * 1.5)  # scale to 0, 1, or 2
        return action

    def get_fitness(self):
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        max_position = -np.inf
        while not done:
            action = self.get_action(state)
            state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            position, velocity = state
            if position > max_position:
                max_position = position
                total_reward += 0.5
            if velocity > 0.1 and action == 0:
                total_reward -= 0.5
            if velocity < -0.1 and action == 2:
                total_reward -= 0.5
            if position > -0.55 and position < -0.4 and action == 1:
                total_reward -= 0.5
            
            total_reward += reward
        return total_reward

    def mutate(self, sigma):
        self.weights += sigma * np.random.randn(2)  # MountainCar-v0 has 2 states

# Create a global env variable so we don't need to pass it around
env = gym.make('MountainCar-v0')  # Change the environment here

# Define the fitness function at the top level so it can be pickled
def get_fitness(agent):
    return agent.get_fitness()

# Evolution strategy
def evolution_strategy(sigma, population_size, generation_count):
    agent = Agent(env)

    for generation in range(generation_count):
        pool = mp.Pool(processes=mp.cpu_count())
        new_agents = []
        for _ in range(population_size):
            new_agent = Agent(env)
            new_agent.weights = np.copy(agent.weights)
            new_agent.mutate(sigma)
            new_agents.append(new_agent)
        fitnesses = pool.map(get_fitness, new_agents)

        max_fitness = max(fitnesses)
        max_index = fitnesses.index(max_fitness)

        agent = new_agents[max_index]
        print(f'Generation {generation}, max fitness: {max_fitness}')
        
        if max_fitness > -200:
            sigma = 0.3
        if max_fitness > -170:
            sigma = 0.2
        if max_fitness > -150:
            sigma = 0.1
        if max_fitness > -120:
            sigma = 0.05
        if max_fitness > -90:
            sigma = 0.01

    return agent

def main():
    sigma = 0.4
    population_size = 100
    generation_count = 500

    best_agent = evolution_strategy(sigma, population_size, generation_count)

    # Save the weights of the best agent to a file
    with open('best_agent_mc.pkl', 'wb') as f:
        pickle.dump(best_agent.weights, f)

    env.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f'Time taken: {time.time() - start_time}')
