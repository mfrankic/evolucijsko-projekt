from bipedal_walker_solver import BipedalWalkerSolver

solver = BipedalWalkerSolver(num_params=24*4)

solver.load_weights('bipedal_walker_weights_2.pkl')
best_params, best_reward, curr_reward, sigma = solver.train(num_iterations=1)

# solver.save_weights('bipedal_walker_weights_2.pkl')

solver.play(best_params)
