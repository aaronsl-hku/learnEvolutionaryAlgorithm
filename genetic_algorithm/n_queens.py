import argparse
import functools
import random
import array
import numpy as np
from typing import Tuple, List, Sequence

from deap import tools, creator, base, algorithms

COLORS = ['\033[41m', '\033[46m']


def n_queens(n: int) -> Tuple:
    ngen = n * 20
    pop_size = int(100 * np.log2(n))
    def fitness_func(positions: Sequence) -> List[int]:
        violations = 0

        # iterate over every pair of queens and find if they are on the same diagonal:
        for col1, row1 in enumerate(positions):
            for col2, row2 in enumerate(positions):
                if col2 <= col1:
                    continue

                if abs(col1 - col2) == abs(row1 - row2):
                    violations += 1

        return violations,

    creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
    creator.create("Individual", array.array, typecode='I', fitness = creator.FitnessMin)

    toolbox = base.Toolbox()
    gen_idx = functools.partial(random.sample, range(n), k = n)
    toolbox.register("individual", tools.initIterate, creator.Individual, gen_idx)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_func)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=2./n) # TODO: Study into this crossover method
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1./n)


    pop = toolbox.population(n = pop_size)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)


    pop, log = algorithms.eaSimple(pop, toolbox, # mu=pop_size, lambda_=pop_size,
                                        cxpb=0.9, mutpb=0.1, ngen=ngen,
                                        stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof

def print_solution(solution, colors):
    n = len(solution)
    for i, j in enumerate(solution):
        c = colors if i % 2 == 0 else colors[::-1]
        seq = ['Q' if j == col else ' ' for col in range(n)]
        s = ''
        for i, e in enumerate(seq):
            s += f'{c[i % len(c)]} {e} \033[00m'
        print(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="Number of Queens")

    args = parser.parse_args()
    pop, log, hof = n_queens(args.n)
    best_sol = hof[0]
    print_solution(best_sol, COLORS)
