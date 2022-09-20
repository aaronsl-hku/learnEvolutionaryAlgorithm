from typing import Callable, List, Tuple
import numpy.typing as NPT

import pygad
import numpy as np
import pandas as pd

"""Given a list of cities and the distances between each pair of cities,
what is the shortest possible route that visits each city and returns to the origin city?


"""



def _generate_initial_population(num_cities: int, population_size: int) -> NPT.NDArray:
    """Generate initial population for genetic algorithm.

    Parameters
    ----------
    num_cities : int
        Number of cities to travel, including origin (which is same as destination).

    Returns
    -------
    NPT.NDArray
        2D array of shape [population_size, num_cities-1].
        Each row is a solution with no duplicates.
    """
    population = np.tile(
        np.arange(1, num_cities, dtype=np.uint16),
        reps = (population_size, 1)
    )
    for i in range(population_size):
        np.random.shuffle(population[i,:])
    return population

def _generate_fitness_function(distance_arr: NPT.NDArray[np.float_]) -> Callable:
    def fitness_func(solution, solution_idx):
        distance = 0.
        src = 0
        for dst in solution:
            distance += distance_arr[src, dst]
            src = dst
        distance += distance_arr[src,0]
        fitness = 1. / distance
        return fitness
    return fitness_func

def tsp(distance_df: pd.DataFrame) -> Tuple[pygad.GA, List[str], float]:
    """Given distance dataframe, return optimized GA instance and the best solution.
    The starting city is set as the first city in the index.

    Parameters
    ----------
    distance_df : pd.DataFrame
        Values are distances (non-negative square matrix), and index and columns are city names.

    Returns
    -------
    Tuple[pygad.GA, List[str]]
        optimized GA instance and the best solution
    """

    sol_per_pop = 60
    num_parents_mating = 30

    map_city_index = {}
    map_index_city = {}
    for index, city in enumerate(distance_df.index):
        map_city_index[city] = index
        map_index_city[index] = city

    init_population = _generate_initial_population(len(distance_df.index), sol_per_pop)
    fitness_func = _generate_fitness_function(distance_df.values)

    ga = pygad.GA(
        random_seed = 0,
        num_generations = 3000,
        num_parents_mating = num_parents_mating,
        allow_duplicate_genes = False,
        parent_selection_type = "rank",
        keep_elitism = 15,
        initial_population = init_population,
        gene_type = np.uint16,
        fitness_func = fitness_func,
        mutation_type = "inversion",
        mutation_probability = 1,
        crossover_probability = 0,
        # save_solutions = True,
        save_best_solutions = True,
        on_generation = (
            lambda g: print(
                "Gen {:>4} - Best route distance: {:>6.2f} - Route: {}".format(
                    g.generations_completed,
                    1/g.best_solutions_fitness[g.generations_completed-1],
                    g.best_solutions[g.generations_completed-1]+1
                ))
        ),
        parallel_processing=["thread", 5],
    )
    ga.run()
    solution = ga.best_solutions[ga.best_solution_generation]
    solution_fitness = ga.best_solutions_fitness[ga.best_solution_generation]
    solution_route = [map_index_city[0]]
    for index in solution:
        solution_route.append(map_index_city[index])
    return ga, solution_route, solution_fitness


if __name__ == '__main__':
    df = pd.read_table("tsp_test_cases/gr17_d.txt", sep="\s+", header=None, index_col=None)
    df.index = [str(i) for i in range(1, df.shape[0]+1)]
    df.columns = df.index

    print(df)
    ga, sol, sol_fitness = tsp(df)
    print("Solution: ", sol)
    print("Distance: ", 1/sol_fitness)
    print("Solution generation: ", ga.best_solution_generation)
    # ga.plot_new_solution_rate()
    # ga.plot_result()
