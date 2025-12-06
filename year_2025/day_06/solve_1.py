import operator
import re
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
    :challenge: 4277556
    :expect: 3261038365331
    """

    lines = defaultlist(list)
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            for i, m in enumerate(re.findall(r"\S+", line)):
                lines[i].append(m)

    operator_lookup = {'*': operator.mul, '+': operator.add, '-': operator.sub}
    total = 0
    for m in lines:
        match_reversed = reversed(m)
        ope, *rest = match_reversed
        total += reduce(operator_lookup[ope], map(int, rest))

    return total

if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)




































