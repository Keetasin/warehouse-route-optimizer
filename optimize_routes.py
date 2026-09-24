"""Compare warehouse route heuristics and their total computation time.

Run ``python optimize_routes.py`` for 10, 20, 30, and 50 pickup points.
Use ``--counts 10 --details`` to print every walking cell and pickup order.
"""

import argparse
import ast
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np


DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def load_warehouse():
    """Read constants without executing Creat_warehouse.py's plotting code."""
    source = Path(__file__).with_name("Creat_warehouse.py").read_text(encoding="utf-8")
    assignments = {}
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                assignments[target.id] = statement.value

    grid_call = assignments["warehouse"]
    grid = np.array(ast.literal_eval(grid_call.args[0]), dtype=int)
    entrance = ast.literal_eval(assignments["entrance"])
    exit_point = ast.literal_eval(assignments["exit_point"])
    return grid, entrance, exit_point


def generate_pickups(grid, count, seed):
    shelves = list(zip(*np.where(grid == 1)))
    if count < 0 or count > len(shelves):
        raise ValueError(f"Pickup count must be between 0 and {len(shelves)}")
    indices = np.random.default_rng(seed).choice(len(shelves), size=count, replace=False)
    return [tuple(map(int, shelves[index])) for index in indices]


def neighbors(grid, cell):
    row, col = cell
    for dr, dc in DIRECTIONS:
        other = (row + dr, col + dc)
        if 0 <= other[0] < grid.shape[0] and 0 <= other[1] < grid.shape[1]:
            yield other


def bfs(grid, source):
    distance = {source: 0}
    parent = {source: None}
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        for other in neighbors(grid, cell):
            if grid[other] == 0 and other not in distance:
                distance[other] = distance[cell] + 1
                parent[other] = cell
                queue.append(other)
    return distance, parent


def trace(parent, target):
    segment = []
    while target is not None:
        segment.append(target)
        target = parent[target]
    segment.reverse()
    return segment


def coverage_per_step(grid, entrance, exit_point, pickups):
    """Visit cells with the most new pickups per walking step; use no distance matrix."""
    cover = {}
    for index, shelf in enumerate(pickups):
        for cell in neighbors(grid, shelf):
            if grid[cell] == 0:
                cover[cell] = cover.get(cell, 0) | (1 << index)

    full = (1 << len(pickups)) - 1
    collected = cover.get(entrance, 0)
    path = [entrance]
    while collected != full:
        distance, parent = bfs(grid, path[-1])
        choices = (
            (distance[cell] / new_count, distance[cell], -new_count, cell)
            for cell, bits in cover.items()
            if cell in distance
            if (new_count := (bits & ~collected).bit_count())
        )
        try:
            target = min(choices)[-1]
        except ValueError as error:
            raise ValueError("Some pickups cannot be reached") from error

        for cell in trace(parent, target)[1:]:
            path.append(cell)
            collected |= cover.get(cell, 0)
            if collected == full:
                break

    _, parent = bfs(grid, path[-1])
    if exit_point not in parent:
        raise ValueError("Exit cannot be reached")
    path.extend(trace(parent, exit_point)[1:])
    return path


class RouteProblem:
    def __init__(self, grid, entrance, exit_point, pickups, precompute=True):
        self.grid = grid
        self.pickups = pickups
        for point in (entrance, exit_point):
            if not (0 <= point[0] < grid.shape[0] and 0 <= point[1] < grid.shape[1]):
                raise ValueError(f"Point outside warehouse: {point}")
            if grid[point] != 0:
                raise ValueError(f"Entrance/exit is not a walkway: {point}")

        reachable, _ = bfs(grid, entrance)
        if exit_point not in reachable:
            raise ValueError("Exit cannot be reached from entrance")

        candidate_cells = []
        for shelf in pickups:
            if not (0 <= shelf[0] < grid.shape[0] and 0 <= shelf[1] < grid.shape[1]):
                raise ValueError(f"Pickup outside warehouse: {shelf}")
            if grid[shelf] != 1:
                raise ValueError(f"Pickup is not on a shelf: {shelf}")
            cells = [cell for cell in neighbors(grid, shelf) if cell in reachable]
            if not cells:
                raise ValueError(f"Pickup has no reachable walkway: {shelf}")
            candidate_cells.append(cells)

        self.nodes = list(dict.fromkeys([entrance, exit_point] + [
            cell for cells in candidate_cells for cell in cells
        ]))
        node_id = {cell: index for index, cell in enumerate(self.nodes)}
        self.start = node_id[entrance]
        self.end = node_id[exit_point]
        self.candidates = [[node_id[cell] for cell in cells] for cells in candidate_cells]
        self.covers = [set() for _ in self.nodes]
        for shelf_index, cells in enumerate(self.candidates):
            for cell in cells:
                self.covers[cell].add(shelf_index)
        self.coverage_mask = {
            cell: sum(1 << shelf for shelf in self.covers[index])
            for index, cell in enumerate(self.nodes)
        }
        self.full_mask = (1 << len(pickups)) - 1

        self.distances = []
        self.parents = []
        if precompute:
            for source in self.nodes:
                distance, parent = bfs(grid, source)
                self.distances.append([distance[target] for target in self.nodes])
                self.parents.append(parent)

    def cost(self, route):
        return sum(self.distances[a][b] for a, b in zip(route, route[1:]))

    def walking_path(self, route):
        path = []
        for source, target in zip(route, route[1:]):
            segment = trace(self.parents[source], self.nodes[target])
            path.extend(segment if not path else segment[1:])
        return path if path else [self.nodes[self.start]]

    def pickup_order(self, path):
        coverage = {cell: self.covers[index] for index, cell in enumerate(self.nodes)}
        seen = set()
        order = []
        for cell in path:
            for shelf in sorted(coverage.get(cell, ())):
                if shelf not in seen:
                    order.append(shelf)
                    seen.add(shelf)
        return order

    def picked_mask(self, path):
        mask = 0
        for cell in path:
            mask |= self.coverage_mask.get(cell, 0)
        return mask

    def check(self, route):
        path = self.walking_path(route)
        self.check_path(path)
        if len(path) - 1 != self.cost(route):
            raise AssertionError("Path length differs from route cost")
        return path

    def check_path(self, path):
        if path[0] != self.nodes[self.start] or path[-1] != self.nodes[self.end]:
            raise AssertionError("Route has wrong entrance or exit")
        if any(self.grid[cell] != 0 for cell in path):
            raise AssertionError("Route crosses a shelf")
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(path, path[1:])):
            raise AssertionError("Route contains an invalid step")
        if self.picked_mask(path) != self.full_mask:
            raise AssertionError("Route misses a pickup")


def cheapest_insertion(problem, distance=None):
    route = [problem.start, problem.end]
    remaining = set(range(len(problem.pickups)))
    remaining -= problem.covers[problem.start] | problem.covers[problem.end]
    if distance is None:
        distance = lambda a, b: problem.distances[a][b]

    while remaining:
        best = None
        for shelf in sorted(remaining):
            for cell in problem.candidates[shelf]:
                for position in range(1, len(route)):
                    before, after = route[position - 1:position + 1]
                    extra = distance(before, cell) + distance(after, cell) - distance(before, after)
                    choice = (extra, shelf, cell, position)
                    if best is None or choice < best:
                        best = choice
        _, _, cell, position = best
        route.insert(position, cell)
        remaining -= problem.covers[cell]

    # Symmetric walkway distances let 2-opt compare only the two changed edges.
    while True:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                a, b, c, d = route[i - 1], route[i], route[j], route[j + 1]
                if distance(a, c) + distance(b, d) < distance(a, b) + distance(c, d):
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True
                    break
            if improved:
                break
        if not improved:
            return route


def lazy_cheapest_insertion(problem):
    cache = {}

    def distance(a, b):
        if a not in cache:
            cache[a] = bfs(problem.grid, problem.nodes[a])
        return cache[a][0][problem.nodes[b]]

    route = cheapest_insertion(problem, distance)
    path = []
    for source, target in zip(route, route[1:]):
        distance(source, target)
        segment = trace(cache[source][1], problem.nodes[target])
        path.extend(segment if not path else segment[1:])
    return path if path else [problem.nodes[problem.start]]


def variable_neighborhood_descent(problem, initial_route):
    """Shorten walking distance by changing or removing service waypoints."""
    route = initial_route[:]
    best_cost = problem.cost(route)

    while True:
        interior = range(1, len(route) - 1)

        def alternatives():
            for i in interior:
                yield route[:i] + route[i + 1:]
            for i in interior:
                for node in range(2, len(problem.nodes)):
                    if node not in route:
                        yield route[:i] + [node] + route[i + 1:]
            for i in interior:
                for j in interior:
                    if i != j:
                        moved = route[:]
                        moved.insert(j, moved.pop(i))
                        yield moved
            for i in interior:
                for j in range(i + 1, len(route) - 1):
                    yield route[:i] + route[i:j + 1][::-1] + route[j + 1:]

        for candidate in alternatives():
            cost = problem.cost(candidate)
            if cost < best_cost and problem.picked_mask(problem.walking_path(candidate)) == problem.full_mask:
                route = candidate
                best_cost = cost
                break
        else:
            return route


def large_neighborhood_search(problem, initial_route, seed=20, iterations=80):
    """Remove several waypoints, repair missing pickups, then descend locally."""
    rng = np.random.default_rng(seed)
    best = variable_neighborhood_descent(problem, initial_route)
    best_cost = problem.cost(best)
    distance = problem.distances

    for _ in range(iterations):
        interior = list(range(1, len(best) - 1))
        if not interior:
            break
        remove_count = int(rng.integers(1, len(interior) // 2 + 2))
        removed = set(map(int, rng.choice(interior, size=remove_count, replace=False)))
        trial = [node for index, node in enumerate(best) if index not in removed]

        while True:
            mask = problem.picked_mask(problem.walking_path(trial))
            missing = [shelf for shelf in range(len(problem.pickups)) if not mask & (1 << shelf)]
            if not missing:
                break
            lowest = float("inf")
            choices = []
            for shelf in missing:
                for cell in problem.candidates[shelf]:
                    for position in range(1, len(trial)):
                        before, after = trial[position - 1:position + 1]
                        extra = distance[before][cell] + distance[cell][after] - distance[before][after]
                        if extra < lowest:
                            lowest = extra
                            choices = [(cell, position)]
                        elif extra == lowest:
                            choices.append((cell, position))
            cell, position = choices[int(rng.integers(len(choices)))]
            trial.insert(position, cell)

        trial = variable_neighborhood_descent(problem, trial)
        cost = problem.cost(trial)
        if cost < best_cost:
            best, best_cost = trial, cost

    return best


def particle_swarm(problem, seed=20, particles=40, iterations=150):
    count = len(problem.pickups)
    if count == 0:
        return [problem.start, problem.end]
    if particles < 1 or iterations < 0:
        raise ValueError("Particles must be positive and iterations cannot be negative")

    def decode(position):
        route = [problem.start]
        covered = problem.covers[problem.start] | problem.covers[problem.end]
        for shelf in np.argsort(position[:count], kind="stable"):
            shelf = int(shelf)
            if shelf in covered:
                continue
            choices = problem.candidates[shelf]
            slot = min(int(position[count + shelf] * len(choices)), len(choices) - 1)
            cell = choices[slot]
            route.append(cell)
            covered |= problem.covers[cell]
        route.append(problem.end)
        return route

    rng = np.random.default_rng(seed)
    positions = rng.random((particles, count * 2))
    velocities = np.zeros_like(positions)
    personal_best = positions.copy()
    personal_score = np.array([problem.cost(decode(p)) for p in positions])
    winner = int(np.argmin(personal_score))
    global_best = personal_best[winner].copy()
    global_score = int(personal_score[winner])

    for _ in range(iterations):
        velocities = np.clip(
            0.7 * velocities
            + 1.4 * rng.random(positions.shape) * (personal_best - positions)
            + 1.4 * rng.random(positions.shape) * (global_best - positions),
            -0.2,
            0.2,
        )
        positions = np.clip(positions + velocities, 0.0, 1.0)
        for index, position in enumerate(positions):
            score = problem.cost(decode(position))
            if score < personal_score[index]:
                personal_score[index] = score
                personal_best[index] = position.copy()
                if score < global_score:
                    global_score = score
                    global_best = position.copy()

    return decode(global_best)


def save_route_plot(grid, entrance, exit_point, cases, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    figure, axes = plt.subplots(len(cases), 2, figsize=(15, 4.5 * len(cases)), squeeze=False)
    colors = ListedColormap(["#eaf6ff", "#3c4148"])
    for row, (count, pickups, baseline, improved) in enumerate(cases):
        for column, (name, path, line_color) in enumerate((
            ("Insertion + 2-opt", baseline, "#4169e1"),
            ("Large Neighborhood Search", improved, "#e67e22"),
        )):
            ax = axes[row, column]
            ax.imshow(grid, cmap=colors, vmin=0, vmax=1, interpolation="nearest")
            ax.plot([cell[1] for cell in path], [cell[0] for cell in path],
                    color=line_color, linewidth=2.5, alpha=0.9, zorder=2)
            ax.scatter([cell[1] for cell in pickups], [cell[0] for cell in pickups],
                       marker="s", s=65, c="#f1c40f", edgecolors="#20242a",
                       linewidths=0.5, zorder=3)
            for cell, label, color in ((entrance, "IN", "#27ae60"),
                                       (exit_point, "OUT", "#e74c3c")):
                ax.scatter(cell[1], cell[0], s=180, c=color, edgecolors="white",
                           linewidths=1, zorder=4)
                ax.text(cell[1], cell[0], label, ha="center", va="center",
                        color="white", fontsize=7, weight="bold", zorder=5)
            ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.55)
            ax.tick_params(which="both", bottom=False, left=False,
                           labelbottom=False, labelleft=False)
            ax.set_title(f"{count} pickups · {name} · {len(path) - 1} steps", fontsize=13)

    figure.suptitle("Warehouse routes  |  yellow: pickups  ·  green: entrance  ·  red: exit",
                     fontsize=17)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", nargs="+", type=int, default=[10, 20, 30, 50])
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--particles", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--lns-iterations", type=int, default=300)
    parser.add_argument("--lns-restarts", type=int, default=2)
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--save-plot", type=Path, help="Save insertion and LNS routes as a PNG")
    args = parser.parse_args()
    if args.lns_iterations < 0 or args.lns_restarts < 1:
        parser.error("LNS iterations must be nonnegative and restarts must be positive")

    grid, entrance, exit_point = load_warehouse()
    # Keep the source script's 10-point sample, then extend it for fair scaling.
    largest = max(args.counts)
    shelf_count = int(np.count_nonzero(grid == 1))
    if largest > shelf_count:
        parser.error(f"Only {shelf_count} shelf cells are available")
    first_count = min(10, largest)
    sample = generate_pickups(grid, first_count, args.seed)
    if largest > first_count:
        chosen = set(sample)
        shuffled_shelves = generate_pickups(grid, shelf_count, args.seed + 1)
        sample.extend(point for point in shuffled_shelves if point not in chosen)

    plot_cases = []
    print("n  method                   steps      run_s   setup_s   total_s")
    for count in args.counts:
        pickups = sample[:count]
        started = perf_counter()
        fast_path = coverage_per_step(grid, entrance, exit_point, pickups)
        fast_seconds = perf_counter() - started

        started = perf_counter()
        light_problem = RouteProblem(grid, entrance, exit_point, pickups, precompute=False)
        lazy_path = lazy_cheapest_insertion(light_problem)
        lazy_seconds = perf_counter() - started
        light_problem.check_path(fast_path)
        light_problem.check_path(lazy_path)

        for name, path, elapsed in (
            ("Coverage per Step", fast_path, fast_seconds),
            ("Lazy Insertion+2opt", lazy_path, lazy_seconds),
        ):
            print(f"{count:<2} {name:<24} {len(path) - 1:>5} "
                  f"{elapsed:>10.4f} {0:>9.4f} {elapsed:>9.4f}")
            if args.details:
                order = light_problem.pickup_order(path)
                print("  pickup order:", [(index + 1, pickups[index]) for index in order])
                print("  path:", path)

        started = perf_counter()
        problem = RouteProblem(grid, entrance, exit_point, pickups)
        setup_seconds = perf_counter() - started

        started = perf_counter()
        insertion_route = cheapest_insertion(problem)
        insertion_seconds = perf_counter() - started

        started = perf_counter()
        lns_route = min(
            (
                large_neighborhood_search(
                    problem, insertion_route, seed=args.seed + restart,
                    iterations=args.lns_iterations,
                )
                for restart in range(args.lns_restarts)
            ),
            key=problem.cost,
        )
        lns_seconds = perf_counter() - started

        started = perf_counter()
        swarm_route = particle_swarm(
            problem, seed=args.seed, particles=args.particles, iterations=args.iterations
        )
        swarm_seconds = perf_counter() - started

        for name, route, search_seconds in (
            ("Cheapest Insertion+2opt", insertion_route, insertion_seconds),
            ("Large Neighborhood Search", lns_route, lns_seconds),
            ("Particle Swarm", swarm_route, swarm_seconds),
        ):
            path = problem.check(route)
            print(f"{count:<2} {name:<24} {len(path) - 1:>5} "
                  f"{search_seconds:>10.4f} {setup_seconds:>9.4f} {search_seconds + setup_seconds:>9.4f}")
            if args.details:
                order = problem.pickup_order(path)
                print("  pickup order:", [(index + 1, pickups[index]) for index in order])
                print("  path:", path)

        if args.save_plot:
            plot_cases.append((count, pickups, problem.check(insertion_route), problem.check(lns_route)))

    if args.save_plot:
        save_route_plot(grid, entrance, exit_point, plot_cases, args.save_plot)
        print("Plot saved:", args.save_plot)


if __name__ == "__main__":
    main()
