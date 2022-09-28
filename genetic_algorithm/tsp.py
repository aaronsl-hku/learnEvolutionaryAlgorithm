"""Implementation of TSP using `deap`.
"""

from copy import deepcopy
from functools import partial
from queue import PriorityQueue
import random
from typing import Callable, Dict, List, Tuple
import numpy as np
from deap import base, creator, tools, algorithms

import tsplib95, networkx

import sys
from pathlib import Path
import requests

DATA_PATH = Path('data/tsp')

def cxEdgeRecombination(ind1: List[int], ind2: List[int]) -> Tuple[List[int], List[int]]:
    def recombine(p_queue: PriorityQueue, neighbors: Dict[int,set], is_p_queue_neg: bool) -> List[int]:
        visited = set()
        offspring = []
        init_iter, idx = True, random.choice(list(neighbors.keys())) * (-1 if is_p_queue_neg else 1)
        while len(visited) < len(neighbors):
            if init_iter:
                init_iter = False
            else:
                _, idx = p_queue.get()
            curr_city = -idx if is_p_queue_neg else idx
            if curr_city in visited:
                continue


            while curr_city is not None:
                # print(curr_city, neighbors[curr_city])
                visited.add(curr_city)
                offspring.append(curr_city)
                if len(neighbors[curr_city]) == 0:
                    break

                curr_neighbors_by_cardinality = [[] for i in range(5)]
                for neighbor in neighbors[curr_city]:
                    neighbors[neighbor].remove(curr_city)
                    curr_neighbors_by_cardinality[len(neighbors[neighbor])].append(neighbor)

                for next_city_candidates in curr_neighbors_by_cardinality:
                    if next_city_candidates:
                        curr_city = random.choice(next_city_candidates)
                        break
                else:
                    curr_city = None
        return offspring


    neighbors = {node: set() for node in ind1}
    p_queue1 = PriorityQueue() # For offspring 1
    p_queue2 = PriorityQueue() # For offspring 2
    num_nodes = len(ind1)
    for i in range(num_nodes):
        neighbors[ind1[i]].add(ind1[i-1])
        neighbors[ind2[i]].add(ind2[i-1])
        if i == num_nodes - 1:
            neighbors[ind1[i]].add(ind1[0])
            neighbors[ind2[i]].add(ind2[0])
        else:
            neighbors[ind1[i]].add(ind1[i+1])
            neighbors[ind2[i]].add(ind2[i+1])


    for i in range(num_nodes):
        # neighbors[i].update([ind1[i-1], ind2[i-1]])
        # if i == num_nodes - 1:
        #     neighbors[i].update([ind1[0], ind2[0]])
        # else:
        #     neighbors[i].update([ind1[i+1], ind2[i+1]])
        p_queue1.put((len(neighbors[i]), i))
        p_queue2.put((len(neighbors[i]), -i))

    neighbors_copy = deepcopy(neighbors)
    offspring1 = recombine(p_queue1, neighbors, is_p_queue_neg = False)
    offspring2 = recombine(p_queue2, neighbors_copy, is_p_queue_neg = True)

    ind1[:], ind2[:] = offspring1[:], offspring2[:]
    return ind1, ind2




def read_data(set_name):
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
    return distance_matrix.tolist()

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
    random.seed(1234)
    num_cities = len(distance_matrix)
    pop_size = num_cities * 20

    toolbox = base.Toolbox()
    gen_idx = partial(random.sample, range(0, num_cities-1), num_cities-1)
    toolbox.register("individual", tools.initIterate, creator.Individual, gen_idx)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    eval_tsp = generate_eval(distance_matrix=distance_matrix)
    toolbox.register("evaluate", eval_tsp)
    toolbox.register("mate", cxEdgeRecombination)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=max(0.1,1./num_cities))
    toolbox.register("select", tools.selTournament, tournsize=2)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.3, ngen=num_cities * 50,
                                   stats=stats, halloffame=hof, verbose=True)
    best_fitness = toolbox.evaluate(hof[0])

    return pop, log, hof, best_fitness

if __name__ == "__main__":
    set_name = sys.argv[1]
    distance = read_data(set_name)

    pop, log, hof, best_fitness = tsp(distance)
    print("Best solution: ")
    print("Objective: ", best_fitness)
    print(' -> '.join(["0",] + [str(i+1) for i in hof[0]] + ["0",]))