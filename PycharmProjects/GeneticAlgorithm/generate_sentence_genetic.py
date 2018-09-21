

from __future__ import print_function
from __future__ import division
import time
import random
import string
import numpy as np
import matplotlib.pyplot as pl
from difflib import SequenceMatcher
target_sentence = 'i think evolution works!'


def similar(a, b, scale=100, exp=2):
    # 'a' can be a huge list to be compared to a single target 'b'
    # print(a)
    # exit()
    for i, member in enumerate(a):
        match = SequenceMatcher(None, member, b).ratio()
        # match = 0
        # for j in range(len(b)):
        #     if member[j] == b[j]:
        #         match += 1
        # match /= len(b)
        if i == 0:
            all_comparisons = np.asarray(match, dtype=np.double)
        else:
            all_comparisons = np.hstack((all_comparisons, match))
    # similarity is a value between 0.0 and 1.0, so we should scale them by, say 10
    fitness = scale*np.asarray(all_comparisons, dtype=np.float)
    fitness = fitness**exp
    fitness /= (scale**exp)
    fitness *= scale
    return fitness


class Population(object):

    def __init__(self, size, mutation_rate, max_len, fitness_scale=100,
                 fitness_exp=3, div_thresh=1e-10): # N is the size of population
        self.max_len = max_len # max possible length of sentence
        # list of current population of solutions
        self.size = size # size of population
        self.population = [self.generate_predictions() for _ in range(self.size)]
        # self.population.append('to be and not to by')
        self.population = np.asarray(self.population)
        self.fittest = [0]*size # ordered list of the fittest among the current population
        self.parents = [] # list of parents for next generation
        self.fitness_vals = [0]*size # ordered list of numerical fitness of each subject in the current generation
        self.pick_probs = [0]*size # probability to pick each parent from the self.fittest list of subjects
        self.mutation = mutation_rate # mutation rate to alter the genes of offsprings
        self.fitness_scale = fitness_scale
        self.fitness_exp = fitness_exp
        self.epsilon = div_thresh # to save division by zero
        self.target = target_sentence
        pass

    def generate_predictions(self):
        return ''.join(self.generate_character() for _ in range(self.max_len))

    def generate_character(self):
        return random.choice(' abcdefghijklmnopqrstuvwxyz!')
                             # '0123456789' +
                             # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def fitness(self):
        self.fitness_vals = similar(self.population, self.target,
                                    scale=self.fitness_scale, exp=self.fitness_exp)
        self.fittest = self.population[np.argmax(self.fitness_vals)]
        self.avg_fitness = self.fitness_vals.mean()
        pass

    def select(self):
        self.parents = []
        for i, member in enumerate(self.population):
            for f in range(np.floor(self.fitness_vals[i]).astype(np.int)):
                self.parents.append(member)
        pass

    def reproduce(self, one_parent_only=False):
        # we will reproduce by taking average of parents, you can do whatever you want
        if one_parent_only:
            return
        self.population = []
        for k in range(self.size):
            parent1 = np.random.choice(self.parents)
            parent2 = np.random.choice(self.parents)
            N = np.random.randint(self.max_len) # int(self.max_len/2) #
            child = parent1[:N] + parent2[N:]
            self.population.append(child)

    def mutate(self):
        for k in range(self.size):
            member = [x for x in self.population[k]]
            if np.random.randn() < self.mutation:
                for j in range(self.max_len):
                    if np.random.randn() < self.mutation:
                        member[j] = self.generate_character()
                # member[np.random.randint(self.max_len)] = self.generate_character()
            self.population[k] = ''.join(x for x in member)
        pass

    def get_fittest(self):
        return self.fittest, self.fitness_vals.max()


def main():
    population = Population(size=200, mutation_rate=0.01, max_len=len(target_sentence),
                            fitness_scale=100, fitness_exp=2, div_thresh=1e-10)
    generations = 1000000
    max_fit = 0
    log = 100
    for N in range(generations):
        population.fitness()
        best_guess, max_fit_N = population.get_fittest()
        if max_fit_N > max_fit:
            max_fit = max_fit_N
            print('---> Checkpoint: Generation ({}), best = {}, max. fitness = {:.2f}%, '
                  'avg. fitness = {:.2f}%'.format(N,
                                                  best_guess,
                                                  max_fit_N,
                                                  population.avg_fitness))
        if N % log == 0:
            max_fit = max_fit_N
            print('{} up: Generation ({}), best = {}, max. fitness = {:.2f}%, '
                  'avg. fitness = {:.2f}%'.format(log,
                                                  N,
                                                  best_guess,
                                                  max_fit_N,
                                                  population.avg_fitness))
        if best_guess == target_sentence:
            print('INFO: BEST MATCH FOUND AT GENERATION {}'.format(N))
            break
        population.select()
        population.reproduce(one_parent_only=False)
        population.mutate()
    pass


if __name__ == '__main__':
    main()













