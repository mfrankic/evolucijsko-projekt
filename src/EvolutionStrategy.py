from __future__ import print_function
import numpy as np
import multiprocessing as mp

np.random.seed(0)


def worker_process(arg):
    get_reward_func, weights = arg
    return get_reward_func(weights)


class EvolutionStrategy(object):
    def __init__(self, weights, get_reward_func, population_size=50, sigma=0.1, learning_rate=0.03, decay=0.999,
                 num_threads=1):

        self.weights = weights
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads

    def _get_weights_try(self, w, p):
        return [w_i + self.SIGMA * p_i for w_i, p_i in zip(w, p)]

    def get_weights(self):
        return self.weights

    def _get_population(self):
        return [[np.random.randn(*w.shape) for w in self.weights] for _ in range(self.POPULATION_SIZE)]

    def _get_rewards(self, pool, population):
        worker_args = ((self.get_reward, self._get_weights_try(self.weights, p)) for p in population)
        if pool is not None:
            rewards = pool.map(worker_process, worker_args)
        else:
            rewards = [self.get_reward(weights_try) for weights_try in map(self._get_weights_try, [self.weights]*len(population), population)]
        return np.array(rewards)
        
    def _update_weights(self, rewards, population):
        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std
        update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
        self.weights = [w + update_factor * np.dot(np.array([p[index] for p in population]).T, rewards).T for index, w in enumerate(self.weights)]
        self.learning_rate *= self.decay

    def run(self, iterations, print_step=10):
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        for iteration in range(iterations):
            population = self._get_population()
            rewards = self._get_rewards(pool, population)

            self._update_weights(rewards, population)

            if (iteration + 1) % print_step == 0:
                print('iter %d. reward: %f' % (iteration + 1, self.get_reward(self.weights)))
        if pool is not None:
            pool.close()
            pool.join()
