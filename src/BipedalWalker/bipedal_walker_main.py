import sys
import time

from bipedal_walker_solver import BipedalWalkerSolver


solver = BipedalWalkerSolver(num_params=24*4)

def main(action, get_data=False):
    if action == 'train':
        # Train the model
        iterations = 4000
        start = time.time() / 60
        best_params, best_reward, curr_reward, sigma = solver.train(num_iterations=iterations)
        end = time.time() / 60
        
        training_time = end - start
        
        # save the training time and number of iterations to a csv file
        with open('../../data/bipedal_walker_training_time.csv', 'w') as f:
            f.write(f'{training_time},{iterations}')

        # Save the weights
        solver.save_weights('bipedal_walker_weights.pkl')

        # Load the weights and play the game
        solver.load_weights('bipedal_walker_weights.pkl')
        solver.play(best_params)
    elif action == 'play':
        # Load the weights and play the game
        best_params = solver.load_weights('bipedal_walker_weights.pkl')
        
        if get_data:
            for i in range(1, 1001):
                score = solver.play(best_params, render=False)
                # save the score to a csv file
                with open('../../data/bipedal_walker_scores.csv', 'a') as f:
                    f.write(f'{i},{score}\n')
        else:
            solver.play(best_params)
    else:
        print('Invalid action')

if __name__ == '__main__':
    action = sys.argv[1]
    get_data = sys.argv[2] == '--get-data' if len(sys.argv) > 2 else False
    main(action, get_data)
