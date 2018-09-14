

from __future__ import print_function
from __future__ import division
import random
import string
import numpy as np
import matplotlib.pyplot as pl
from difflib import SequenceMatcher


def similar(a, b):
    # 'a' can be a huge list to be compared to a single target 'b'
    for i, member in enumerate(a):
        match = SequenceMatcher(None, member, b).ratio()
        if i == 0:
            all_comparisons = np.asarray(match, dtype=np.double)
        else:
            all_comparisons = np.hstack((all_comparisons, match))
    # print(all_comparisons)
    return np.asarray(all_comparisons, dtype=np.float)


target_sentence = 'To be or not to be'

class Population(object):

    def __init__(self, size, mutation_rate, num_parents, fittest, max_len, div_thresh=1e-10): # N is the size of population
        self.max_len = max_len # max possible length of sentence
        self.fittest_count = fittest # number of the fittest to pick as parents for next generation
        # list of current population of solutions
        self.population = [''.join(self.generate()
                                   for _ in range(self.max_len)) for _ in range(size)]
        self.population.append('to be or this to be')
        self.population = np.asarray(self.population)
        self.size = size # size of population
        self.fittest = [0]*fittest # actual list of the fittest in current generation
        self.parents = [] # list of parents for next generation
        self.fitness_vals = [0]*size # numerical fitness of each subject in current generation
        self.pick_probs = [0]*size # probability to pick each parent from the self.fittest list of subjects
        self.mutation = mutation_rate # mutation rate to alter the genes of offsprings
        self.num_parents = num_parents # number of parents for each offspring
        self.epsilon = 1e-10 # to save division by zero
        self.target = target_sentence
        pass

    def generate(self):
        return random.choice(string.letters+ string.ascii_lowercase + string.digits + string.whitespace)

    def fitness(self):
        # fitness is how close the guess was to the actual answer,
        # in our case the answer is a root to an equation, so we just see how far it puts the equation from zero!
        # this gives us how close we are to zero, the less the value, the more fit we are!
        self.fitness_vals = similar(self.population, self.target)**3
        # since these are to be used as probabilities, they should add to 1
        self.fitness_vals /= (self.fitness_vals.sum()+self.epsilon)
        # # finally pick the fittest by their indices
        self.fittest = self.fitness_vals.argsort()[::-1] #[:self.num_parents]
        pass

    def select(self):
        # will generate a list of parents to yield the next generation
        # the fittest will have the highest probability to be picked for mating
        # print(self.population)
        # print(self.fitness_vals)
        # print(self.fittest)
        self.parents = self.population[self.fittest]
        pass

    def reproduce(self):
        # we will reproduce by taking average of parents, you can do whatever you want
        # self.population = np.zeros(self.size) # reset the population to get new values
        new_population = []
        for _ in range(self.size):
            these_parents = np.random.choice(a=self.parents, size=2, replace=True, p=self.fitness_vals)
            child = these_parents[0][:self.max_len] + these_parents[1][self.max_len:]
            new_population.append(child)
        return np.asarray(new_population)

    def mutate(self):
        # for mutation, we shall randomly change a few numbers in the whole
        mutate = np.random.uniform(0, 1, self.size)
        select_to_mutate = mutate < self.mutation
        # print(np.count_nonzero(select_to_mutate)/len(self.population))
        these_to_mutate = np.nonzero(select_to_mutate)[0]
        for id_x in these_to_mutate:
            this_to_mutate = list(self.population[id_x])
            # print(this_to_mutate, self.generate())
            this_to_mutate[np.random.randint(self.max_len)] = self.generate()
            self.population[id_x] = str(this_to_mutate)
        pass

    def get_fittest(self):
        # will return the stats for the fittest in the current generations
        this_id = np.argmax(self.fitness_vals)
        return self.population[this_id]


def main():
    population = Population(size=10, mutation_rate=0.08, num_parents=5,
                            fittest=10, max_len=len(target_sentence),
                            div_thresh=1e-10)
    generations = 100000
    # 1. calculate fitness,
    # 2. choose fittest as parents but with some randomness,
    # 3. reproduce to get new generation,
    # 4. mutate to introduce variation
    # 5. repeat...
    for N in range(generations):
        population.fitness()
        # print(population.population)
        population.select()
        population.reproduce()
        population.mutate()
        approx = population.get_fittest()
        print(approx)
    pass


if __name__ == '__main__':
    main()













