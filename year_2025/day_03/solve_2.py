import sys
from typing import List

from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(10_000)

_default_puzzle_input = "year_2024/day_01/puzzle.txt"
_default_test_input = "year_2024/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")
test_input_3 = build_location(__file__, "test_3.txt")
test_input_4 = build_location(__file__, "test_4.txt")
test_input_5 = build_location(__file__, "test_5.txt")


def max_jolt(accumulate: str, number: List[int], k, start, end_at):
    best, best_at = -1, -1
    i = start
    next_end_at = min(len(number), end_at + 1)
    while i < end_at:

        current = number[i]
        if current > best:
            best = current
            best_at = i

        i += 1

    accumulate += str(best)

    if len(accumulate) == k:
        return accumulate
    else:
        return max_jolt(accumulate, number, k, best_at + 1, next_end_at)


def solve_(__input=None):
    """
    :challenge: 3121910778619
    :expect: 173065202451341
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(map(int, list(line))))

    jolts = []
    k = 12
    for word in lines:
        end_at = len(word) - k + 1
        jolts.append(max_jolt("", word, k, 0, end_at))

    return sum(int(j) for j in jolts)


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
