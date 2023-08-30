from collections import deque
import numpy as np
import torch
import torch.nn as nn
#from torch.autograd import Variable
import torch.nn.functional as F

import gymnasium as gym
#import argparse
import os

import time
from TwinDelayed import Actor, Critic, ReplayBuffer, TD3
from collections import deque

env = gym.make('BipedalWalker-v3', hardcore=True)

state, _ = env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])
        
agent = TD3(state_dim, action_dim, max_action)
    
# load(agent = agent, filename='checkpnt, directory = 'dir_chkpoint')
def load(agent, filename, directory):
    agent.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    agent.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
    agent.actor_target.load_state_dict(torch.load('%s/%s_actor_t.pth' % (directory, filename)))
    agent.critic_target.load_state_dict(torch.load('%s/%s_critic_t.pth' % (directory, filename)))
    
    return agent

def play(env, agent, n_episodes):
    state, _ = env.reset()
    
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()        
        score = 0
        
        time_start = time.time()
        
        while True:
            action = agent.select_action(np.array(state))
            # env.render()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
            if done:
                break 

        s = (int)(time.time() - time_start)
        
        scores_deque.append(score)
        scores.append(score)
        
        with open('../../data/bipedal_walker_td3_scores.csv', 'a') as f:
            f.write(f'{i_episode},{score}\n')
        
        print('Episode {}\tAverage Score: {:.2f},\tScore: {:.2f} \tTime: {:02}:{:02}:{:02}'\
                  .format(i_episode, np.mean(scores_deque), score, s//3600, s%3600//60, s%60))

load(agent=agent, filename='chpnt_88seed_300-5sc_9h44m', directory = 'dir_chk')

play(env=env, agent=agent, n_episodes=1000)
