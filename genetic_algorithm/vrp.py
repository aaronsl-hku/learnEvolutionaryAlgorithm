from copy import deepcopy
from genetic_algorithm import tsp

class VRP(tsp.TSP):
    def __init__(self, set_name, vehicles, depot_idx, toolbox = None):
        self.vehicles = vehicles
        self.depot_idx = depot_idx

        super().__init__(set_name)
        self._reformat_distance()
        self.solver_index_to_city_index = {}

    def _reformat_distance(self):
        self.actual_num_cities = self.num_cities

        depot_in_dist = [r[self.depot_idx] for r in self.distance]
        for _ in range(1, self.vehicles):
            depot_out_dist = [d for d in self.distance[self.depot_idx]]
            self.distance.append(depot_out_dist)

        raise NotImplementedError("In progress.")