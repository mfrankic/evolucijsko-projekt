import sys
import time

from bipedal_walker_solver import BipedalWalkerSolver


solver = BipedalWalkerSolver(num_params=24*4)

def main(action, hardcore=False, get_data=False, opposite=False):
    if action == 'train':
        # Train the model
        iterations = 6000
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
        best_params = solver.load_weights('bipedal_walker_hardcore_weights.pkl') if ((not opposite) and hardcore) else solver.load_weights('bipedal_walker_weights.pkl')
        
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
    action = sys.argv[1]
    hardcore = sys.argv[2] == '--hardcore' if len(sys.argv) > 2 else False
    get_data = sys.argv[3] == '--get-data' if len(sys.argv) > 3 else False
    opposite = False
    if action == 'play':
        if hardcore and get_data:
            opposite = sys.argv[4] == '--opposite' if len(sys.argv) > 4 else False
        elif ((not hardcore) and get_data) or (hardcore and (not get_data)):
            opposite = sys.argv[3] == '--opposite' if len(sys.argv) > 3 else False
        else:
            opposite = sys.argv[2] == '--opposite' if len(sys.argv) > 2 else False
    
    main(action, hardcore, get_data, opposite)
