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


def moore_neighborhood(m, coord):
    r, c = coord
    r0 = max(r-1, 0)
    r1 = min(r+2, m.shape[0])  # +2 because slice end is exclusive
    c0 = max(c-1, 0)
    c1 = min(c+2, m.shape[1])
    return m[r0:r1, c0:c1]

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
    mask = np.vstack([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    while 1:
        if len(where) < 1:
            mask = ~storage
            mask.args

        coords = np.argwhere(m == 1)
        for co in coords:
            sub = moore_subgrid(m, co)
            print("coord:", co)
            print(sub)
            print("---")
            # x = np.convolve([[1, 1, 1], [1, 0, 1], [1, 1, 1]], storage)
            for y, x in np.argwhere(storage == 1):
                insort(where, (y, x))

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
