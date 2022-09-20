from datetime import datetime
import genetic

def get_fitness(genes: str, target: str):
    fitness = 0
    for actual, expected in zip(genes, target):
        if actual == expected:
            fitness += 1
    return fitness

def display(genes: str, target: str, start_time: datetime):
    time_diff = datetime.now() - start_time
    fitness = get_fitness(genes, target)
    print(f"{genes}\t{fitness}\t{time_diff}")

def guess_password(target: str):
    gene_set = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."
    start_time = datetime.now()

    def wrapped_get_fitness(genes: str):
        return get_fitness(genes, target)

    def wrapped_display(genes: str):
        return display(genes, target, start_time)

    optimal_fitness = target_len = len(target)
    genetic.get_best(wrapped_get_fitness, target_len, optimal_fitness, gene_set, wrapped_display)

if __name__ == '__main__':
    target = "We are the champion!"
    guess_password(target)


