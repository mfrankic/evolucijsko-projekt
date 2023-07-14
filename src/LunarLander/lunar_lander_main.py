import sys
import time

from lunar_lander_solver import LunarLanderSolver

# Assume a simple linear policy with four parameters corresponding to the four state variables
solver = LunarLanderSolver(num_params=4*8)

def main(action, wind=False, get_data=False, opposite=False):
    if action == 'train':
        # Train the model
        iterations = 4000
        start = time.time()
        best_params, best_reward, curr_reward, sigma = solver.train(num_iterations=iterations, wind=wind)
        end = time.time()
        
        training_time = end - start
        
        # save the training time and number of iterations to a csv file
        with open('../../data/lunar_lander_training_time.csv', 'w') as f:
            f.write(f'{training_time},{iterations}')

        # Save the weights
        solver.save_weights('lunar_lander_wind_weights.pkl') if wind else solver.save_weights('lunar_lander_weights.pkl')

        # Load the weights and play the game
        solver.load_weights('lunar_lander_wind_weights.pkl') if wind else solver.load_weights('lunar_lander_weights.pkl')
        solver.play(best_params, wind=wind)
    elif action == 'play':
        # Load the weights and play the game
        best_params = solver.load_weights('lunar_lander_wind_weights.pkl') if opposite ^ wind else solver.load_weights('lunar_lander_weights.pkl')
        
        if get_data:
            for i in range(1, 1001):
                score = solver.play(best_params, render=False, wind=wind)
                # save the score to a csv file
                with open('../../data/lunar_lander_scores.csv', 'a') as f:
                    f.write(f'{i},{score}\n')
        else:
            solver.play(best_params, wind=wind)
    else:
        print('Invalid action')

if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'help'
    args = sys.argv[2:]
    
    wind = '--wind' in args
    opposite = '--opposite' in args
    get_data = '--get-data' in args

    # write help message
    help_message = 'Usage: python lunar_lander_main.py <action> [--wind] [--opposite] [--get-data]\n\n'
    additional_info = 'action: train or play\n\n'
    additional_info += '--wind: play the game with wind\n'
    additional_info += '--opposite: play the game with weights trained on the opposite version\n'
    additional_info += '--get-data: play the game and get the data for 1000 iterations\n'
    additional_info += '\nNote: --opposite and --get-data can only be used with action play\n'
    
    if action == 'help':
        print(help_message + additional_info)
        exit()

    main(action, wind, get_data, opposite)
