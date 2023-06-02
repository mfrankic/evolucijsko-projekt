import sys

from bipedal_walker_solver import BipedalWalkerSolver

solver = BipedalWalkerSolver(num_params=24*4)

def main(action):
    if action == 'train':
        # Train the model
        best_params, best_reward, curr_reward, sigma = solver.train(num_iterations=100)

        # Save the weights
        solver.save_weights('bipedal_walker_weights_3.pkl')

        # Load the weights and play the game
        solver.load_weights('bipedal_walker_weights_3.pkl')
        solver.play(best_params)
    elif action == 'play':
        # Load the weights and play the game
        best_params = solver.load_weights('bipedal_walker_weights_3.pkl')
        solver.play(best_params)
    else:
        print('Invalid action')

if __name__ == '__main__':
    main(sys.argv[1])
