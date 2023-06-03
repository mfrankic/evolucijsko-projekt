import sys
import time

from lunar_lander_solver import LunarLanderSolver

# Assume a simple linear policy with four parameters corresponding to the four state variables
solver = LunarLanderSolver(num_params=4)

def main(action, get_data=False):
    if action == 'train':
        # Train the model
        iterations = 100
        start = time.time()
        best_params, best_reward, curr_reward, sigma = solver.train(num_iterations=iterations)
        end = time.time()
        
        training_time = end - start
        
        # save the training time and number of iterations to a csv file
        with open('../../data/lunar_lander_training_time.csv', 'w') as f:
            f.write(f'{training_time},{iterations}')

        # Save the weights
        solver.save_weights('lunar_lander_weights.pkl')

        # Load the weights and play the game
        solver.load_weights('lunar_lander_weights.pkl')
        solver.play(best_params)
    elif action == 'play':
        # Load the weights and play the game
        best_params = solver.load_weights('lunar_lander_weights.pkl')
        
        if get_data:
            for i in range(1, 1001):
                score = solver.play(best_params, render=False)
                # save the score to a csv file
                with open('../../data/lunar_lander_scores.csv', 'a') as f:
                    f.write(f'{i},{score}\n')
        else:
            solver.play(best_params)
    else:
        print('Invalid action')

if __name__ == '__main__':
    action = sys.argv[1]
    get_data = sys.argv[2] == '--get-data' if len(sys.argv) > 2 else False
    main(action, get_data)
