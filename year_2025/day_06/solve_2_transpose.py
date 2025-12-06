import operator
import re
import sys
from functools import reduce
from typing import Callable, List

from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2
from aoc.tools import transpose

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
        for line in read_lines(fp):
            lines.append([ch for ch in line])

    text_transposed = transpose(matrix=lines)
    operator_lookup = {'*': operator.mul, '+': operator.add, '-': operator.sub}

    numbers: List[int] = []
    op_text = text_transposed[0][-1]
    total = 0
    for index, l in enumerate(text_transposed):
        if re.match(r"\S", next_op_text := l[-1]) and index > 0:
            total += reduce(operator_lookup[op_text], numbers)
            numbers.clear()
            op_text = next_op_text

        number = "".join(l[:-1]).replace(" ", "")
        if number != "":
            numbers.append(int(number))

    total += reduce(operator_lookup[op_text], numbers)
    numbers.clear()

    return total


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
