import operator
import sys
from functools import reduce
from typing import List

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

    lines: List[List[str]] = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append([char for char in line])

    lines[-1].append(" ")
    height, width = len(lines), len(lines[0])

    numbers: List[int] = []
    oper: str = ""
    total: int = 0
    operator_lookup = {'*': operator.mul, '+': operator.add, '-': operator.sub}
    for x in range(0, width):
        if (optional_oper := lines[height - 1][x]) != " ":
            if any(numbers):
                total += reduce(operator_lookup[oper], numbers)
                numbers.clear()

            oper = optional_oper

        vertical_generator = range(height - 2, -1, -1)
        if number := reduce(lambda acc, y: lines[y][x].replace(" ", "") + acc, vertical_generator, ""):
            numbers.append(int(number))

    return total + reduce(operator_lookup[oper], numbers)


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
