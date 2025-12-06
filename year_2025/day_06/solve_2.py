import operator
import re
import sys
from functools import reduce
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
    :challenge: 3263827
    :expect: 8342588849093
    """

    lines = []
    with open(locate(__input), "r") as fp:
        lines = list(read_lines(fp))

    match_ranges: List[Tuple[slice, str]] = []
    matches = [m for m in re.finditer(r"\S+", lines[-1])]
    matches_rev = reversed(matches)
    last_start = len(lines[-1]) + 2
    operator_lookup = {'*': operator.mul, '+': operator.add, '-': operator.sub}

    for match in matches_rev:
        match_ranges.append((slice(match.start(), last_start - 1), match.group(0)))
        last_start = match.end()

    total = 0
    for ms, ope in match_ranges:
        numbers: List[int] = []

        for j in range(ms.start, ms.stop - 1):
            number = reduce(lambda acc, i: acc + lines[i][j].replace(" ", ""), range(0, len(lines) - 1), "")
            numbers.append(int(number))

        total += reduce(operator_lookup[ope], map(int, numbers))

    return total


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
