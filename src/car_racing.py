import random
import _pickle as pickle
import numpy as np
from EvolutionStrategy import EvolutionStrategy
import gym
from car_racing_model import Model


class Agent:
    AGENT_HISTORY_LENGTH = 1
    NUM_OF_ACTIONS = 3
    POPULATION_SIZE = 5
    EPS_AVG = 1
    SIGMA = 0.1
    LEARNING_RATE = 0.03
    INITIAL_EXPLORATION = 0.1
    FINAL_EXPLORATION = 0.0
    EXPLORATION_DEC_STEPS = 100000
    NEGATIVE_REWARD_LIMIT = 20

    def __init__(self, render=False):
        self.model = Model()
        self.env: gym.Env = gym.make('CarRacing-v2', render_mode='human') if render else gym.make('CarRacing-v2')
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE, num_threads = -1)
        self.exploration = self.INITIAL_EXPLORATION

    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))
        return np.argmax(prediction) - 1

    def load(self, filename='car_racing_weights.pkl'):
        with open(filename,'rb') as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()

    def get_observation(self):
        return np.array(self.env.state).flatten() / 255

    def save(self, filename='car_racing_weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.es.get_weights(), fp)

    def play(self, episodes):
        self.model.set_weights(self.es.weights)
        for episode in range(episodes):
            self.env.reset()
            observation = self.get_observation()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            score = 0
            negative_counter = 0
            while not done:
                action = self.get_predicted_action(sequence)
                _, reward, terminated, truncated, _ = self.env.step([action, 1, 0])
                done = terminated or truncated
                
                # if reward is negative, increment counter
                if reward < 0:
                    negative_counter += 1
                else:
                    negative_counter = 0

                # if there have been NEGATIVE_REWARD_LIMIT negative rewards, end the episode
                if negative_counter >= self.NEGATIVE_REWARD_LIMIT:
                    score -= 20
                    done = True
                
                observation = self.get_observation()
                sequence = sequence[1:]
                sequence.append(observation)
                score += reward
                print("score: %d" % score)

    def train(self, iterations):
        self.es.run(iterations, print_step=1)

    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)
        for episode in range(self.EPS_AVG):
            self.env.reset()
            observation = self.get_observation()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            negative_counter = 0
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration - self.INITIAL_EXPLORATION/self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action = random.uniform(-1, 1)
                else:
                    action = self.get_predicted_action(sequence)
                _, reward, terminated, truncated, _ = self.env.step([action, 1, 0])
                done = terminated or truncated
                
                # if reward is negative, increment counter
                if reward < 0:
                    negative_counter += 1
                else:
                    negative_counter = 0

                # if there have been NEGATIVE_REWARD_LIMIT negative rewards, end the episode
                if negative_counter >= self.NEGATIVE_REWARD_LIMIT:
                    total_reward -= 20
                    done = True
                
                total_reward += reward
                observation = self.get_observation()
                sequence = sequence[1:]
                sequence.append(observation)

        return total_reward/self.EPS_AVG
