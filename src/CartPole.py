import gymnasium as gym
import numpy as np
import multiprocessing as mp
import pickle
import time

# Agent class
class Agent:
    def __init__(self, env):
        self.env: gym.Env = env
        self.weights = np.random.rand(4) * 2 - 1

    def get_action(self, state):
        action = 1 if np.dot(self.weights, state) > 0 else 0
        return action

    def get_fitness(self):
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action = self.get_action(state)
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
        return total_reward

    def mutate(self, sigma):
        self.weights += sigma * np.random.randn(4)

# Create a global env variable so we don't need to pass it around
env = gym.make('CartPole-v1')

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

    return agent

def main():
    sigma = 0.25
    population_size = 50
    generation_count = 100

    best_agent = evolution_strategy(sigma, population_size, generation_count)

    # Save the weights of the best agent to a file
    with open('best_agent.pkl', 'wb') as f:
        pickle.dump(best_agent.weights, f)

    env.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f'Time taken: {time.time() - start_time}')
