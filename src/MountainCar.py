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
            # if position > max_position + 0.1:
            #     max_position = position
            #     total_reward += 1
            
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
    end_decay = generation_count // 1.5
    decay_by = sigma / (end_decay - 1)
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
        print(f'Generation {generation+1}, max fitness: {max_fitness}')
        
        if end_decay >= generation >= 1:
            sigma -= decay_by

    return agent

def main():
    sigma = 0.5
    population_size = 300
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
