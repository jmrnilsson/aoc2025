import sys

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
    :challenge: 357
    :expect: 17432
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(map(int, list(line))))

    jolts = []
    for l in lines:
        max_ab = 0
        for i, a in enumerate(l):
            max_b = 0
            for j in range(i, len(l)):
                # for j, b in enumerate(l):
                if i == j:
                    continue

                b = l[j]

                if b > max_b:
                    max_b = b

            ab = int(str(a) + str(max_b))
            if str(ab).count("0"):
                continue

            if ab > max_ab:
                max_ab = ab

        jolts.append(max_ab)

    return sum(int(j) for j in jolts)


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
