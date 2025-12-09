import itertools
import sys
from collections import deque
from typing import List, Tuple

import numpy as np

from aoc.helpers import build_location, locate, read_lines, timed
from aoc.printer import ANSIColors, get_meta_from_fn, print2

sys.setrecursionlimit(30_000)

_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")
test_input_3 = build_location(__file__, "test_3.txt")
test_input_4 = build_location(__file__, "test_4.txt")
test_input_5 = build_location(__file__, "test_5.txt")

type Coords = List[Tuple[int, int]]


class CompressedGrid:
    """ Create a grid that use arrays along each axis to scale the coordinates down to a manageable size """

    def __init__(self, lines: Coords):
        ys = sorted(set(y for y, _ in lines))
        xs = sorted(set(x for _, x in lines))

        y_mapping = {y: i * 2 for i, y in enumerate(ys)}
        x_mapping = {x: i * 2 for i, x in enumerate(xs)}

        self.inverse_y_mapping = {v: k for k, v in y_mapping.items()}
        self.inverse_x_mapping = {v: k for k, v in x_mapping.items()}

        max_y = y_mapping[ys[-1]] + 1
        max_x = x_mapping[xs[-1]] + 1

        self.grid = np.zeros((max_y + 2, max_x + 2), dtype=int)

        for y, x in lines:
            cy = y_mapping[y]
            cx = x_mapping[x]
            self.grid[cy, cx] = 1

    def get_original_coordinates(self, compressed_coord: Tuple[int, int]) -> Tuple[int, int]:
        cy, cx = compressed_coord
        original_y = self.inverse_y_mapping.get(cy)
        original_x = self.inverse_x_mapping.get(cx)

        if original_y is None or original_x is None:
            raise ValueError("Compressed coordinate does not map to an original coordinate.")

        return original_y, original_x


def find_path(lines: Coords) -> Coords:
    queue = deque(lines[1:])
    path = [lines[0]]
    while queue:
        current = queue.popleft()
        ly, lx = path[-1]  # last
        y, x = current

        if abs(y - ly) == 0:
            path.append(current)

        elif abs(x - lx) == 0:
            path.append(current)

        else:
            queue.append(current)

    return path


def find_edge(lines: Coords) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    distance = float('inf')
    ny, nx = None, None  # nearest y, nearest x

    for y, x in lines:
        # Simplest heuristic to find nearest to (0,0)
        if distance > (h := y + x):
            distance, ny, nx = h, y, x

    return (ny, nx), (ny + 1, nx + 1)  # just outside the edge in the opposite direction


def flood_fill(grid: np.ndarray, start: Tuple[int, int], fill_value: int) -> int:
    max_y, max_x = grid.shape
    target_value = grid[start]
    queue = deque([start])
    filled = 0

    while queue:
        y, x = queue.popleft()
        if grid[y, x] != target_value:
            continue

        grid[y, x] = fill_value
        filled += 1

        # Von Neumann neighborhood
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < max_y and 0 <= nx < max_x and grid[ny, nx] == target_value:
                queue.append((ny, nx))

    return filled


def walk_path(_path: Coords, grid: np.ndarray):
    """ Fills the grid along the given path """
    seen = set()
    path = itertools.cycle(_path)
    previous = None
    for current in path:
        if previous is None:
            previous = current
            seen.add(current)
            continue

        y, x = current
        prev_y, prev_x = previous

        if abs(y - prev_y) == 0:
            s, e = sorted([x, prev_x])
            for u in range(s, e + 1):
                grid[(y, u)] = 1

        if abs(x - prev_x) == 0:
            s, e = sorted([y, prev_y])
            for u in range(s, e + 1):
                grid[(u, x)] = 1

        if current in seen:
            break

        previous = current
        seen.add(current)


def find_largest_area(compressed_grid: CompressedGrid, where: Coords) -> Tuple:
    """ Finds the largest rectangular area that fits within the filled area """
    best = None
    largest = -float('inf')
    for l, r in itertools.combinations(where, 2):
        compressed_area = abs(l[0] - r[0]) + 1   # y
        compressed_area *= abs(l[1] - r[1]) + 1  # x
        y1, y2 = sorted((l[0], r[0]))
        x1, x2 = sorted((l[1], r[1]))
        sub_grid = compressed_grid.grid[y1:y2 + 1, x1: x2 + 1]

        # Make sure every compressed cell in area is covered
        if np.sum(sub_grid) == compressed_area:
            edge_left = compressed_grid.get_original_coordinates(l)
            edge_right = compressed_grid.get_original_coordinates(r)
            y_length = abs(edge_left[0] - edge_right[0]) + 1
            x_length = abs(edge_left[1] - edge_right[1]) + 1

            if largest < (area := y_length * x_length):
                largest = area
                best = edge_left, edge_right

    return largest, *best


def solve_(__input=None):
    """
    :challenge: 24
    :expect: 1396494456
    """
    lines: Coords = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            x, y = line.split(",")
            lines.append((int(y), int(x)))

    compressed = CompressedGrid(lines)
    corners = [(a, b) for a, b in np.argwhere(compressed.grid == 1)]
    path = find_path(corners)
    walk_path(path, compressed.grid)
    edge, candidate = find_edge(corners)
    flood_fill(compressed.grid, candidate, 1)
    area, edge_0, edge_1 = find_largest_area(compressed, corners)
    return area


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
