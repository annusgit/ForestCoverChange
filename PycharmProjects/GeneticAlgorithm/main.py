

from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from scipy import optimize


def this_objective(population):
    return population ** 5 + (population - 28) ** 2 + np.tan(population)


class Population(object):

    def __init__(self, size, mutation_rate, num_parents, fittest, max_range): # N is the size of population
        self.max_range = max_range # max range of solution space
        self.fittest_count = fittest # number of the fittest to pick as parents for next generation
        # list of current population of solutions
        self.population = np.random.uniform(low=-self.max_range, high=self.max_range, size=size)
        self.size = size # size of population
        self.fittest = [0]*fittest # actual list of the fittest in current generation
        self.parents = [] # list of parents for next generation
        self.fitness_vals = [0]*size # numerical fitness of each subject in current generation
        self.pick_probs = [0]*size # probability to pick each parent from the self.fittest list of subjects
        self.mutation = mutation_rate # mutation rate to alter the genes of offsprings
        self.num_parents = num_parents # number of parents for each offspring
        self.epsilon = 1e-10 # to save division by zero
        self.correct = optimize.brentq(this_objective, -max_range, max_range)
        pass

    def fitness(self):
        # fitness is how close the guess was to the actual answer,
        # in our case the answer is a root to an equation, so we just see how far it puts the equation from zero!
        # this gives us how close we are to zero, the less the value, the more fit we are!
        objective = this_objective(self.population)
        fitness = abs(objective + self.epsilon) # a very simple equation whose root is 17
        self.fitness_vals = (1/fitness)
        # some conversions before going further
        self.max_range = abs(self.fitness_vals.max())
        self.fitness_vals = np.clip(self.fitness_vals, a_min=-self.max_range, a_max=self.max_range)
        self.fitness_vals /= self.max_range # val range [-1, 1]
        self.fitness_vals = (self.fitness_vals+1)/2
        # since these are to be used as probabilities, they should add to 1
        self.fitness_vals /= self.fitness_vals.sum()
        # finally pick the fittest
        indices_fittest = np.argsort(self.fitness_vals)[::-1] #[:self.fittest_count]
        self.fittest = self.population[indices_fittest]
        pass

    def select(self):
        # will generate a list of parents to yield the next generation
        # the fittest will have the highest probability to be picked for mating
        # print(self.fitness_vals.sum())
        self.parents = np.random.choice(a=self.population,
                                        size=self.num_parents*self.size,
                                        p=self.fitness_vals,
                                        replace=True)
        pass

    def reproduce(self):
        # we will reproduce by taking average of parents, you can do whatever you want
        # self.population = np.zeros(self.size) # reset the population to get new values
        # for i in range(0,self.size,self.num_parents):
        #     # self.population[i] = 0 # reset this population member
        #     for k in range(i, i+self.num_parents):
        #         self.population[i] += self.parents[k]
        #     # self.population[i] /= (k+1)
        half = int(1*self.size/2)
        self.population = np.hstack((self.fittest[:half], self.population[half:]))
        np.random.shuffle(self.population) # !
        pass

    def mutate(self):
        mutate = np.random.uniform(0, 1, self.size)
        select_to_mutate = mutate < self.mutation
        self.population += select_to_mutate*np.random.uniform(low=-self.population.mean(),
                                                              high=self.population.mean(),
                                                              size=self.size)
        pass

    def get_fittest(self):
        # will return the stats for the fittest in the current generations
        this_id = np.argmax(self.fitness_vals)
        return self.population[this_id]


def main():
    population = Population(size=2000, mutation_rate=0.01, num_parents=5, fittest=10, max_range=100000)
    generations = 2000000
    threshold = 1e-10
    # 1. calculate fitness,
    # 2. choose fittest as parents but with some randomness,
    # 3. reproduce to get new generation,
    # 4. mutate to introduce variation
    # 5. repeat...
    # print(population.population)
    for N in range(generations):
        population.fitness()
        # population.select()
        population.reproduce()
        population.mutate()
        approx = population.get_fittest()
        error = abs(approx-population.correct)
        if error < threshold:
            break
        if N % 1000 == 0:
            print('log: generation {}: fittest element at {}, '
                  'actual value at {}'.format(N, approx, population.correct))
    print('log: solution found at generation {}, solution {}, '
          'actual {}, error {}%'.format(N, approx, population.correct, 100*error))
    pass


if __name__ == '__main__':
    main()













