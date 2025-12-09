import itertools
import sys
from copy import deepcopy
from typing import Generator, List, Tuple

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


def solve_(__input=None):
    """
    :challenge: 50
    :expect: 4763040296
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(tuple(map(int, line.split(","))))

    # prefer this over heapq nlargest which is *VERY* compact
    largest = -float('inf')
    for u, w in itertools.combinations(lines, 2):
        yr = abs(u[0] - w[0]) + 1
        xr = abs(u[1] - w[1]) + 1
        if largest < (area := yr * xr):
            largest = area

    return largest



if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
