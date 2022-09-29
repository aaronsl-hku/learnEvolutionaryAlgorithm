import random
from queue import PriorityQueue
from copy import deepcopy

from typing import Sequence, Dict, Tuple

def cxEdgeRecombination(ind1: Sequence[int], ind2: Sequence[int]) -> Tuple[Sequence[int], Sequence[int]]:

    def recombine(ind: Sequence[int], p_queue: PriorityQueue, neighbors: Dict[int,set], is_p_queue_neg: bool) -> None:
        unvisited = set(neighbors.keys())
        init_iter, curr_city = True, random.choice(list(neighbors.keys()))
        i = 0
        while len(unvisited):
            if init_iter:
                init_iter = False
            else:
                _, idx = p_queue.get()
                curr_city = -idx if is_p_queue_neg else idx


            while curr_city is not None and curr_city in unvisited:
                unvisited.remove(curr_city)
                ind[i] = curr_city
                i += 1

                min_active_edges = len(unvisited)
                next_city_candidates = []
                for neighbor in neighbors[curr_city]:
                    # Remove curr_city from its neighbors set
                    neighbors[neighbor].remove(curr_city)
                    # Pick the neighbors with least neighbor as next city candidates
                    active_edges = len(neighbors[neighbor])
                    if active_edges < min_active_edges:
                        min_active_edges = active_edges
                        next_city_candidates = [neighbor]
                    elif active_edges == min_active_edges:
                        next_city_candidates.append(neighbor)


                # Random choice from next city candidates
                if next_city_candidates:
                    curr_city = random.choice(next_city_candidates)
                else:
                    curr_city = None


    neighbors = {node: set() for node in ind1}
    p_queue1 = PriorityQueue() # For offspring 1
    p_queue2 = PriorityQueue() # For offspring 2
    num_nodes = len(ind1)

    for i in range(num_nodes):
        neighbors[ind1[i]].add(ind1[(i-1) % num_nodes])
        neighbors[ind2[i]].add(ind2[(i-1) % num_nodes])

        neighbors[ind1[i]].add(ind1[(i+1) % num_nodes])
        neighbors[ind2[i]].add(ind2[(i+1) % num_nodes])


    for i in range(num_nodes):
        p_queue1.put((len(neighbors[i]), i))
        p_queue2.put((len(neighbors[i]), -i))

    neighbors_copy = deepcopy(neighbors)
    recombine(ind1, p_queue1, neighbors, is_p_queue_neg = False)
    recombine(ind2, p_queue2, neighbors_copy, is_p_queue_neg = True)

    return ind1, ind2