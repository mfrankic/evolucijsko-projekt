import sys
import time

from bipedal_walker_solver import BipedalWalkerSolver


solver = BipedalWalkerSolver(num_params=24*4)

def main(action, hardcore=False, get_data=False, opposite=False):
    if action == 'train':
        # Train the model
        iterations = 16000
        start = time.time() / 60
        best_params, best_reward, curr_reward, sigma = solver.train(num_iterations=iterations, hardcore=hardcore)
        end = time.time() / 60
        
        training_time = end - start
        
        # save the training time and number of iterations to a csv file
        with open('../../data/bipedal_walker_training_time.csv', 'w') as f:
            f.write(f'{training_time},{iterations}')

        # Save the weights
        solver.save_weights('bipedal_walker_hardcore_weights.pkl') if hardcore else solver.save_weights('bipedal_walker_weights.pkl')

        # Load the weights and play the game
        solver.load_weights('bipedal_walker_hardcore_weights.pkl') if hardcore else solver.load_weights('bipedal_walker_weights.pkl')
        solver.play(best_params)
    elif action == 'play':
        # Load the weights and play the game
        best_params = solver.load_weights('bipedal_walker_hardcore_weights.pkl') if opposite ^ hardcore else solver.load_weights('bipedal_walker_weights.pkl')
        
        if get_data:
            for i in range(1, 1001):
                score = solver.play(best_params, render=False, hardcore=hardcore)
                # save the score to a csv file
                with open('../../data/bipedal_walker_scores.csv', 'a') as f:
                    f.write(f'{i},{score}\n')
        else:
            solver.play(best_params, hardcore=hardcore)
    else:
        print('Invalid action')

if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'help'
    args = sys.argv[2:]
    
    hardcore = '--hardcore' in args
    opposite = '--opposite' in args
    get_data = '--get-data' in args
    
    # write help message
    help_message = 'Usage: python bipedal_walker_main.py <action> [--hardcore] [--opposite] [--get-data]\n\n'
    additional_info = 'action: train or play\n\n'
    additional_info += '--hardcore: play the hardcore version of the game\n'
    additional_info += '--opposite: play the game with weights trained on the opposite version\n'
    additional_info += '--get-data: play the game and get the data for 1000 iterations\n'
    additional_info += '\nNote: --opposite and --get-data can only be used with action play\n'
    
    if action == 'help':
        print(help_message + additional_info)
        exit()
    
    main(action, hardcore, get_data, opposite)
