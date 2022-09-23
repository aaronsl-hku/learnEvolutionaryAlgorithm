"""Implementation of TSP using `deap`.
"""

from functools import partial
import random
from typing import Callable, List
import numpy as np
from deap import base, creator, tools, algorithms

import sys
from pathlib import Path

DATA_PATH = Path('data/tsp')

def generate_eval(distance_matrix: List[List[float]]) -> Callable:
    def eval_tsp(individual: List[float]) -> List[float]:
        origin = i = -1
        total_dist = 0.
        for j in individual:
            total_dist += distance_matrix[i+1][j+1]
            i = j
        total_dist += distance_matrix[j+1][origin+1]
        return total_dist,
    return eval_tsp

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def tsp(distance_matrix: List[List[float]]):
    num_cities = len(distance_matrix)
    pop_size = num_cities * 20

    toolbox = base.Toolbox()
    gen_idx = partial(random.sample, range(0, num_cities-1), num_cities-1)
    toolbox.register("individual", tools.initIterate, creator.Individual, gen_idx)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    eval_tsp = generate_eval(distance_matrix=distance_matrix)
    toolbox.register("evaluate", eval_tsp)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=num_cities * 50,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof

if __name__ == "__main__":
    set_name = sys.argv[1]

    distance = []
    with (DATA_PATH / f"{set_name}_d.txt").open() as f:
        dist_str = f.readlines()

    for row_str in dist_str:
        row = list(map(float, row_str.split()))
        if len(row):
            distance.append(row)
            print(distance[-1])

    pop, log, hof = tsp(distance)
    print("Best solution: ", [0,] + [i+1 for i in hof[0]])