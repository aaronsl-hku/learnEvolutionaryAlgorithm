"""Implementation of 0/1 Knapsack Problem, using `deap`.
"""
from pathlib import Path
import random
import sys
from deap import base, creator, tools, algorithms
import numpy as np

from typing import Callable, Sequence

DATA_PATH = Path('data/knapsack')

def generate_eval(capacity: int, weights: Sequence[int], values: Sequence[int]) -> Callable:
    def eval_knapsack(individual: Sequence[int]) -> Sequence[int]:
        total_weights = sum([w * i for w, i in zip(weights, individual)])
        if total_weights > capacity:
            return (min(values),)
        total_values = sum([v * i for v, i in zip(values, individual)])
        return (total_values,)
    return eval_knapsack

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def knapsack(capacity: int, weights: Sequence[int], values: Sequence[int]):
    num_items = len(weights)
    pop_size = num_items * 50

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_items)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    eval_knapsack = generate_eval(capacity=capacity, weights=weights, values=values)
    toolbox.register("evaluate", eval_knapsack)
    toolbox.register("mate", tools.cxUniform, indpb=0.05)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=num_items * 20,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof

if __name__ == "__main__":
    set_num = int(sys.argv[1])
    capacity = int((DATA_PATH / f"p{set_num:02}_c.txt").read_text())
    weights = list(map(int, (DATA_PATH / f"p{set_num:02}_w.txt").read_text().split()))
    values = list(map(int, (DATA_PATH / f"p{set_num:02}_p.txt").read_text().split()))
    print("Capacity: ", capacity)
    print("Weights: ", weights)
    print("Values: ", values)
    pop, log, hof = knapsack(capacity, weights, values)
    print("Best solution: ", hof[0])


