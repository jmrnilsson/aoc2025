import operator
import sys
from functools import reduce

from defaultlist import defaultlist

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

    operators = defaultlist(lambda: " ")
    with open(locate(__input), "r") as fp:
        _rl = read_lines(fp)
        lines = _rl[:-1]
        for o in _rl[-1]:
            operators.append(o)

    total, numbers, x = 0, [], len(lines[0])
    while x > -1:
        x -= 1
        if (number := "".join(lines[y][x] for y in range(0, len(lines))).strip()) != "":
            numbers.append(int(number))

        if (op := operators[x]) != " ":
            match op:
                case '+': total += reduce(operator.add, numbers)
                case '*': total += reduce(operator.mul, numbers)

            numbers.clear()

    return total


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
