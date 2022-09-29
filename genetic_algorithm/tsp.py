"""Implementation of TSP using `deap`.
"""

from genetic_algorithm import crossovers

from functools import partial
import random
from typing import List
import numpy as np
from deap import base, creator, tools, algorithms

import tsplib95, networkx

import sys
from pathlib import Path
import requests

import multiprocessing

DATA_PATH = Path('data/tsp')


def selection_func(individuals, k, tournsize, fit_attr="fitness"):
    k_best = k // 10
    k_tourn = k - k_best
    return tools.selBest(individuals, k_best, fit_attr) \
        + tools.selTournament(individuals, k_tourn, tournsize, fit_attr)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class TSP:
    def __init__(self, set_name):
        random.seed(1234)

        self.set_name = set_name
        self.problem, self.distance = self._read_data(set_name)

    @property
    def num_cities(self):
        return len(self.distance)

    @staticmethod
    def eval_tsp_with_distance(individual: List[float], distance_matrix: List[List[float]]) ->  List[float]:
        origin = i = -1
        total_dist = 0.
        for j in individual:
            total_dist += distance_matrix[i+1][j+1]
            i = j
        total_dist += distance_matrix[j+1][origin+1]
        return total_dist,

    @property
    def best_solution(self):
        return ["0",] + [str(i+1) for i in self.hof[0]] + ["0",]

    @staticmethod
    def _read_data(set_name):
        file_path = DATA_PATH / f"{set_name}.tsp"
        if not file_path.exists():
            url = f"http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/{set_name}.tsp"
            response = requests.get(url)
            open(str(file_path), "wb").write(response.content)

        problem = tsplib95.load(str(file_path))
        # convert into a networkx.Graph
        graph = problem.get_graph()

        # convert into a numpy distance matrix
        distance_matrix = networkx.to_numpy_matrix(graph)
        return problem, distance_matrix.tolist()

    def setup_toolbox(self, toolbox=None):
        if toolbox is None:
            toolbox = base.Toolbox()
        self.toolbox = toolbox
        distance_matrix = self.distance
        pop_size = self.num_cities * 10
        self.ngen = self.num_cities * 20

        gen_idx = partial(random.sample, range(0, self.num_cities-1), self.num_cities-1)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, gen_idx)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        eval_tsp = partial(self.eval_tsp_with_distance, distance_matrix=distance_matrix)
        self.toolbox.register("evaluate", eval_tsp)
        self.toolbox.register("mate", crossovers.cxEdgeRecombination)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=max(0.1,1./self.num_cities))
        self.toolbox.register("select", selection_func, tournsize=3)

        self.pop = self.toolbox.population(n=pop_size)
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("median", np.median)
        self.stats.register("min", np.min)

    def run(self):
        self.pop, self.log = algorithms.eaSimple(self.pop, self.toolbox, cxpb=0.9, mutpb=0.1, ngen=self.ngen,
                                    stats=self.stats, halloffame=self.hof, verbose=True)
        best_fitness = self.toolbox.evaluate(self.hof[0])

        return self.hof, best_fitness



if __name__ == "__main__":
    set_name = sys.argv[1]
    with multiprocessing.Pool(5) as pool:
        tsp = TSP(set_name)
        tsp.setup_toolbox()
        tsp.toolbox.register("map", pool.map)
        hof, best_fitness = tsp.run()
    print("Best solution: ")
    print("Objective: ", best_fitness)
    print(' -> '.join(tsp.best_solution))