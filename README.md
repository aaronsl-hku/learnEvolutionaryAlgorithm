# learnGeneticAlgorithms

My personal journal on learning Genetic Algorithms

## PyGAD

### From official documentation

To use the pygad module, here is a summary of the required steps:

1. Preparing the fitness_func parameter. (`fitness_func: solution_gene -> fitness_value`)
2. Preparing Other Parameters.
3. Import pygad.
4. Create an Instance of the pygad.GA Class.
5. Run the Genetic Algorithm.
6. Plotting Results.
7. Information about the Best Solution.
8. Saving & Loading the Results.

### TSP Implementation (`programs_with_pygad/tsp.py`)

> Running on `programs_with_pygad/tsp_test_cases/gr17_d.txt` data (from [https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html]). Optimal distance should be `2085`.
> Attempt implementing mutation-based (no crossover) GA for travelling salesman problem. Obviously there is still some issues in the configuration, such that the fitness is FAR FROM monotonic increasing (that is, the best solutions are not retained).
