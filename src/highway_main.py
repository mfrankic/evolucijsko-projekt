import sys
import time

from highway_solver import HighwaySolver


solver = HighwaySolver(num_params=5*5*5)

def main(action):
    if action == 'train':
        # Train the model
        iterations = 10
        start = time.time()
        best_params, best_reward, curr_reward, sigma = solver.train(num_iterations=iterations)
        end = time.time()
        
        training_time = end - start
        
        # save the training time and number of iterations to a csv file
        with open('../data/highway_training_time.csv', 'w') as f:
            f.write(f'{training_time},{iterations}')

        # Save the weights
        solver.save_weights('highway_weights.pkl')

        # Load the weights and play the game
        solver.load_weights('highway_weights.pkl')
        solver.play(best_params)
    elif action == 'play':
        # Load the weights and play the game
        best_params = solver.load_weights('highway_weights.pkl')
        for i in range(1, 1001):
            score = solver.play(best_params)
            # save the score to a csv file
            with open('../data/highway_scores.csv', 'a') as f:
                f.write(f'{i},{score}\n')
    else:
        print('Invalid action')

if __name__ == '__main__':
    main(sys.argv[1])
