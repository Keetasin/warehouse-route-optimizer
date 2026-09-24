import unittest

import numpy as np

from optimize_routes import (
    RouteProblem,
    cheapest_insertion,
    coverage_per_step,
    generate_pickups,
    lazy_cheapest_insertion,
    large_neighborhood_search,
    load_warehouse,
    particle_swarm,
)


class RouteTests(unittest.TestCase):
    def test_bfs_uses_walkways_instead_of_manhattan_distance(self):
        grid = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        problem = RouteProblem(grid, (0, 0), (0, 2), [])
        route = cheapest_insertion(problem)
        self.assertEqual(problem.cost(route), 6)
        self.assertEqual(len(problem.check(route)) - 1, 6)
        self.assertEqual(len(coverage_per_step(grid, (0, 0), (0, 2), [])) - 1, 6)

    def test_all_methods_pick_two_shelves_from_walkway(self):
        grid = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
        problem = RouteProblem(grid, (0, 1), (2, 1), [(1, 0), (1, 2)])
        insertion = cheapest_insertion(problem)
        swarm = particle_swarm(problem, seed=1, particles=12, iterations=30)
        light_problem = RouteProblem(grid, (0, 1), (2, 1), [(1, 0), (1, 2)], precompute=False)
        fast = coverage_per_step(grid, (0, 1), (2, 1), light_problem.pickups)
        lazy = lazy_cheapest_insertion(light_problem)
        self.assertEqual(problem.cost(insertion), 2)
        self.assertEqual(len(problem.pickup_order(problem.check(insertion))), 2)
        self.assertEqual(len(problem.pickup_order(problem.check(swarm))), 2)
        self.assertEqual(len(fast) - 1, 2)
        self.assertEqual(len(lazy) - 1, 2)
        light_problem.check_path(fast)
        light_problem.check_path(lazy)

    def test_large_neighborhood_search_shortens_ten_point_route(self):
        grid, entrance, exit_point = load_warehouse()
        pickups = generate_pickups(grid, 10, seed=20)
        problem = RouteProblem(grid, entrance, exit_point, pickups)
        initial = cheapest_insertion(problem)
        improved = large_neighborhood_search(problem, initial, seed=20, iterations=80)
        self.assertEqual(problem.cost(initial), 76)
        self.assertEqual(problem.cost(improved), 72)
        problem.check(improved)


if __name__ == "__main__":
    unittest.main()
