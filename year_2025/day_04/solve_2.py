import itertools
import sys
from bisect import insort
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


def moore_neighborhood(y_x: Tuple[int, int], shape: np.shape) -> Generator[Tuple[int, int], None, None]:
    moore_neigh = [(1, 0), (0, 1), (-1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dy, dx in moore_neigh:
        neighbor = y_x[0] + dy, y_x[1] + dx

        if -1 < neighbor[0] < shape[0] and -1 < neighbor[1] < shape[1]:
            yield neighbor


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
                insort(where, (y, x))

            if last_mnemonic == (current_mnemonic := tuple(where)):
                break

            if len(where) < 1:
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
