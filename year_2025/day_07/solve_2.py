import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Set

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

type YAxis = Dict[int, Set[int]]

class QuantumTachyonBeam:
    x_range: Dict[int, int]
    y: int
    ys: YAxis
    max_y: int

    def __init__(self, starting_pos: int, ys: YAxis, max_y: int):
        self.y = 0
        self.x_range = Counter({starting_pos: 1})
        self.ys = ys
        self.max_y = max_y

    def travel(self):
        self.y += 1
        queue = list(self.x_range.items())
        self.x_range.clear()

        while queue:
            x, count_ = queue.pop(0)
            if x in self.ys[self.y]:
                self.x_range.update({x - 1: count_})
                self.x_range.update({x + 1: count_})
            else:
                self.x_range.update({x: count_})

    def sum_timelines(self):
        return sum(self.x_range.values())

    def is_accepting(self):
        return self.y + 1 >= self.max_y


def solve_(__input=None):
    """
    :challenge: 40
    :expect: 15650261281478
    """
    max_y = -1
    starting_x: int = -1
    lines: YAxis = defaultdict(set)
    with open(locate(__input), "r") as fp:
        for i, line in enumerate(read_lines(fp)):
            max_y = i
            for m in re.finditer(r"[\^S]", line):
                if m.group(0) == "S":
                    starting_x = m.start()
                else:
                    lines[i].add(m.start())

    beam = QuantumTachyonBeam(starting_x, lines, max_y)
    while not beam.is_accepting():
        beam.travel()

    return beam.sum_timelines()


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
