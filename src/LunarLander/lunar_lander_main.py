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
        best_params = solver.load_weights('lunar_lander_wind_weights.pkl') if ((not opposite) and wind) else solver.load_weights('lunar_lander_weights.pkl')
        
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
    action = sys.argv[1]
    wind = sys.argv[2] == '--wind' if len(sys.argv) > 2 else False
    get_data = sys.argv[3] == '--get-data' if len(sys.argv) > 3 else False
    opposite = False
    if action == 'play':
        if wind and get_data:
            opposite = sys.argv[4] == '--opposite' if len(sys.argv) > 4 else False
        elif ((not wind) and get_data) or (wind and (not get_data)):
            opposite = sys.argv[3] == '--opposite' if len(sys.argv) > 3 else False
        else:
            opposite = sys.argv[2] == '--opposite' if len(sys.argv) > 2 else False

    main(action, wind, get_data, opposite)
