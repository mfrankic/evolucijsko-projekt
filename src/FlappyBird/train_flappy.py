from flappy import *

agent = Agent()

agent.load('weights.pkl')

agent.train(200)

agent.save()
