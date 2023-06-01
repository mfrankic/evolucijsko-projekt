from cart_pole_solver import CartPoleSolver

# Assume a simple linear policy with four parameters corresponding to the four state variables
solver = CartPoleSolver(num_params=4)

# Train the model
best_params, best_reward, curr_reward, sigma = solver.train(num_iterations=50)

# Save the weights
solver.save_weights('cart_pole_weights.pkl')

# Load the weights and play the game
solver.load_weights('cart_pole_weights.pkl')
solver.play(best_params)
