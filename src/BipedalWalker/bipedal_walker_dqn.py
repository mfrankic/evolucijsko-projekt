import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque
import random

# Hyperparameters
GAMMA = 0.99
LR_ACTOR = 0.001
LR_CRITIC = 0.002
TAU = 0.005
BUFFER_CAPACITY = 10000
BATCH_SIZE = 64
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1

class Actor(tf.keras.Model):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.out = Dense(num_actions, activation='tanh')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.out = Dense(1)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


class Agent:
    def __init__(self, num_actions, num_states):
        self.num_actions = num_actions
        self.num_states = num_states
        self.actor = Actor(self.num_actions)
        self.critic = Critic()
        self.target_actor = Actor(self.num_actions)
        self.target_critic = Critic()
        self.buffer = deque(maxlen=BUFFER_CAPACITY)
        self.actor_optimizer = Adam(LR_ACTOR)
        self.critic_optimizer = Adam(LR_CRITIC)
        self.epsilon = EPSILON

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def get_action(self, state):
        state = np.reshape(state, [1, self.num_states])
        sampled_actions = tf.squeeze(self.actor(state)).numpy()
        noise = self.epsilon * np.random.normal(size=self.num_actions)
        sampled_actions = sampled_actions + noise
        return np.clip(sampled_actions, -1, 1)

    def update(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) < BATCH_SIZE:
            return

        samples = random.sample(self.buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*samples))
        states = np.array(states).reshape(-1, self.num_states)
        next_states = np.array(next_states).reshape(-1, self.num_states)
        dones = dones.astype(int).reshape(-1, 1)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            target_actions = self.target_actor(next_states)
            y = reward + GAMMA * self.target_critic(
                np.concatenate([next_states, target_actions], axis=1)) * (1 - dones)
            critic_value = self.critic(np.concatenate([states, actions], axis=1))
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

            actor_part = self.critic(
                np.concatenate([states, self.actor(states)], axis=1))
            actor_loss = -tf.math.reduce_mean(actor_part)

        grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

        self.update_target(self.target_actor.variables, self.actor.variables, TAU)
        self.update_target(self.target_critic.variables, self.critic.variables, TAU)

        self.epsilon = max(self.epsilon * EPSILON_DECAY, MIN_EPSILON)

    @staticmethod
    def update_target(target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))


if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3")
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    agent = Agent(num_actions, num_states)
    num_episodes = 1000

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        print(f"Episode: {ep+1}, Reward: {total_reward}")
