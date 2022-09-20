import random

def _generate_parent(length: int, gene_set: str) -> str:
    genes = random.choices(gene_set, k=length)
    return ''.join(genes)

def _mutate(parent, gene_set):
    index = random.randrange(0, len(parent))
    child_genes = list(parent)
    new_gene, alternate = random.sample(gene_set, 2)
    if new_gene == child_genes[index]:
        child_genes[index] = alternate
    else:
        child_genes[index] = new_gene
    return ''.join(child_genes)

def get_best(get_fitness, target_len, optimal_fitness, gene_set, display):

    raise NotImplementedError

    random.seed()
    #Init with random parent
    best_parent = _generate_parent(target_len, gene_set)
    best_fitness = get_fitness(best_parent)
    display(best_parent)
    if best_fitness >= optimal_fitness:
        return best_parent
    while True:
        child = _mutate(best_parent, gene_set)
        child_fitness = get_fitness(child)
        if best_fitness >= child_fitness:
            continue
        display(child)
        if child_fitness >= optimal_fitness:
            return child
        best_fitness = child_fitness
        best_parent = child
