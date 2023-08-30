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

start_timestep=1e4

std_noise=0.1

env = gym.make('BipedalWalker-v3', hardcore=True)

state, _ = env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

agent = TD3(state_dim, action_dim, max_action)

# save(agent = agent, filename='checkpnt, directory = 'dir_chkpoint')     
def save(agent, filename, directory):
    torch.save(agent.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(agent.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
    torch.save(agent.actor_target.state_dict(), '%s/%s_actor_t.pth' % (directory, filename))
    torch.save(agent.critic_target.state_dict(), '%s/%s_critic_t.pth' % (directory, filename))
    
# Twin Delayed Deep Deterministic (TD3) policy gradient algorithm
def twin_ddd_train(n_episodes=3000, save_every=10):

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []    

    time_start = time.time()                    # Init start time
    replay_buf = ReplayBuffer()                 # Init ReplayBuffer
    
    timestep_after_last_save = 0
    total_timesteps = 0
    
    low = env.action_space.low
    high = env.action_space.high
    
    print('Low in action space: ', low, ', High: ', high, ', Action_dim: ', action_dim)
            
    for i_episode in range(1, n_episodes+1):
        
        timestep = 0
        total_reward = 0
        
        # Reset environment
        state, _ = env.reset()
        done = False
        
        no_change_counter = 0
        previous_state = state
        
        while True:
            
            # Select action randomly or according to policy
            if total_timesteps < start_timestep:
                action = env.action_space.sample()
            else:
                action = agent.select_action(np.array(state))
                if std_noise != 0: 
                    shift_action = np.random.normal(0, std_noise, size=action_dim)
                    action = (action + shift_action).clip(low, high)
            
            # Perform action
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            done_bool = 0 if timestep + 1 == env._max_episode_steps else float(done)
            total_reward += reward                          # full episode reward

            # Store every timestep in replay buffer
            replay_buf.add((state, new_state, action, reward, done_bool))
            state = new_state

            timestep += 1     
            total_timesteps += 1
            timestep_after_last_save += 1
            
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

        scores_deque.append(total_reward)
        scores_array.append(total_reward)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
            
        if (i_episode % 10) == 0:
            # train_by_episode(time_start, i_episode) 
            s = (int)(time.time() - time_start)
            print('Ep. {}, Timestep {},  Ep.Timesteps {}, Score: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02} '\
                    .format(i_episode, total_timesteps, timestep, \
                            total_reward, avg_score, s//3600, s%3600//60, s%60))
            
            # append the iteration number and reward to a csv file
            with open('../../data/bipedal_walker_hardcore_td3_iteration_reward.csv', 'a') as f:
                f.write(f'{i_episode},{np.mean(scores_array[-10:])}\n')

        agent.train(replay_buf, timestep)

        # Save episode if more than save_every=5000 timesteps
        if timestep_after_last_save >= save_every:
            timestep_after_last_save %= save_every
            save(agent, 'checkpnt_seed_88', 'dir_chk_hard')
        
        if len(scores_deque) == 100 and np.mean(scores_deque) >= 300.5:
            print('Environment solved with Average Score: ',  np.mean(scores_deque))
            break

    time_end = time.time()
    training_time = (time_end - time_start) / 60
    
    with open('../../data/bipedal_walker_hardcore_td3_training_time.csv', 'w') as f:
        f.write(f'{training_time},{i_episode}')
    
    return scores_array, avg_scores_array

scores, avg_scores = twin_ddd_train()

save(agent, 'chpnt_88seed_300-5sc_9h44m', 'dir_chk_hard')
