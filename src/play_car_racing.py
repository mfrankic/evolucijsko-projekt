from car_racing import *

agent = Agent(render=True)

# the pre-trained weights are saved into 'weights.pkl' which you can use.
agent.load('car_racing_weights.pkl')

# play one episode
agent.play(50)
