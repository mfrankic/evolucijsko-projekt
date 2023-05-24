from flappy import *

agent = Agent()

# the pre-trained weights are saved into 'weights.pkl' which you can use.
agent.load('weights.pkl')

# play one episode
agent.play(50)
