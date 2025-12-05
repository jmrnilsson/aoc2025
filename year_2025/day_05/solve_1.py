import sys
from typing import List, Tuple

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
    :challenge: 3
    :expect: 701
    """
    fresh_ingredient_range: List[Tuple[int, ...]] = []
    fruits = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            if "-" in line:
                fresh_ingredient_range.append(tuple(map(int, line.split("-"))))
            else:
                fruits.append(int(line))

    return sum(1 for f in fruits if any(1 for begin, end in fresh_ingredient_range if begin < f <= end))

if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
