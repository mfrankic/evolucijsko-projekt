import ale_py
import shimmy
import gymnasium as gym

env = gym.make("ALE/MsPacman-v5", render_mode="human")
observation, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

env.close()
