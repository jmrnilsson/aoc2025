import itertools
import sys
from copy import deepcopy
from typing import Generator, List, Tuple

import numpy as np

from aoc.helpers import build_location, locate, read_lines
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


class HikingAutomaton:
    trails: List[List[Tuple[int, int]]]
    done: List[List[Tuple[int, int]]]

    def __init__(self, starting_pos: List[List[Tuple[int, int]]], grid):
        self.trails = list(starting_pos)
        self.done = []
        self.grid = grid

    def in_bound(self, y: int, x: int) -> bool:
        return -1 < y < self.grid.shape[0] and -1 < x < self.grid.shape[1]

    def _step(self, y, x) -> Generator[Tuple[int, int, int], None, None]:
        steps = (
            (y - 1, x + 0),
            (y + 0, x + 1),
            (y + 1, x + 0),
            (y + 0, x + -1)
        )
        for step in steps:
            if self.in_bound(*step) and (value := self.grid[step]) == self.grid[y, x] + 1:
                yield *step, value

    def walk(self):
        trails = list(self.trails)
        self.trails.clear()
        for n, trail_ in enumerate(trails):
            last_step = trail_[-1]
            for y, x, value in self._step(*last_step):
                trail = deepcopy(trail_)
                trail.append((y, x))
                if value == 9:
                    self.done.append(trail)
                else:
                    self.trails.append(trail)

    def score(self):
        head_and_tails = {
            (trail[0], trail[-1])
            for trail in self.done
        }
        return len(head_and_tails)

    def rat(self):
        return len(self.done)

    def is_accepting(self):
        return len(self.trails) == 0


def moore_neighborhood(y_x: Tuple[int, int], shape: np.shape):
    neighbors_ = list()
    moore_neigh = [(1, 0), (0, 1), (-1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dy, dx in moore_neigh:
        neighbor = y_x[0] + dy, y_x[1] + dx

        if -1 < neighbor[0] < shape[0] and -1 < neighbor[1] < shape[1]:
            neighbors_.append(neighbor)

    return neighbors_


def solve_(__input=None):
    """
    :challenge: 43
    :expect: 8366
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(map(str, list(line))))

    shape = (len(lines[0]), len(lines))
    storage = np.zeros(shape, dtype=int)

    for y, x in itertools.product(range(0, shape[0]), range(0, shape[1])):
        if lines[y][x] == "@":
            storage[(y, x)] = 1

    where: List[Tuple[int, int]] = sorted((y, x) for y, x in np.argwhere(storage == 1))
    removed_rolls = 0
    last_mnemonic = tuple()

    while 1:
        if len(where) < 1:
            for y, x in np.argwhere(storage == 1):
                where.append((y, x))

            where.sort()

            if last_mnemonic == (current_mnemonic := tuple(where)) or len(where) < 1:
                break

            last_mnemonic = current_mnemonic

        fork_lift: Tuple[int, int] = where.pop()
        roll_count = sum(1 for neigh in moore_neighborhood(fork_lift, shape) if storage[neigh] > 0)

        if roll_count < 4:
            removed_rolls += 1
            storage[fork_lift] = 0


    return removed_rolls


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
