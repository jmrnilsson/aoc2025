import sys
from typing import Tuple

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
    :challenge: 13
    :expect: 1409
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(map(str, list(line))))

    shape = (len(lines[0]), len(lines))
    storage = np.zeros(shape, dtype=int)

    for y in range(0, len(lines)):
        for x in range(0, len(lines)):
            if lines[y][x] == "@":
                storage[(y, x)] = 1

    where = np.argwhere(storage == 1)

    fork_lift_access = 0
    for y, x in where:
        neighborhood = moore_neighborhood((y, x), shape)
        current_sum = 0
        for neigh in neighborhood:
            value = storage[neigh]

            if value > 0:
                current_sum += 1

        if current_sum < 4:
            fork_lift_access += 1

    return fork_lift_access


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
