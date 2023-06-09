import sys

from mountain_car_solver import MountainCarSolver

# Assume a simple linear policy with four parameters corresponding to the four state variables
solver = MountainCarSolver(num_params=3)

def main(action):
    if action == 'train':
        # Train the model
        best_params, best_reward, curr_reward, sigma = solver.train(num_iterations=2000)

        # Save the weights
        solver.save_weights('mountain_car_weights.pkl')

        # Load the weights and play the game
        solver.load_weights('mountain_car_weights.pkl')
        solver.play(best_params)
    elif action == 'play':
        # Load the weights and play the game
        best_params = solver.load_weights('mountain_car_weights.pkl')
        solver.play(best_params)
    else:
        print('Invalid action')
        
if __name__ == '__main__':
    main(sys.argv[1])
