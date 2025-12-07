import sys
from collections import Counter
from typing import Dict

import numpy as np

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


class QuantumTachyonBeam:
    x_range: Dict[int, int]
    y: int

    def __init__(self, starting_pos: int, grid):
        self.y = 0
        self.x_range = Counter({starting_pos: 1})
        self.grid = grid

    def travel(self):
        self.y += 1
        queue = list(self.x_range.items())
        self.x_range.clear()

        while queue:
            x, count_ = queue.pop(0)
            if self.grid[(self.y, x)] == "^":
                if (left := x - 1) > -1:
                    self.x_range.update({left: count_})
                if (right := x + 1) < self.grid.shape[1]:
                    self.x_range.update({right: count_})
            else:
                self.x_range.update({x: count_})

    def sum_timelines(self):
        return sum(self.x_range.values())

    def is_accepting(self):
        return self.y + 1 == self.grid.shape[0]


def solve_(__input=None):
    """
    :challenge: 40
    :expect: 15650261281478
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(map(str, list(line))))

    grid = np.matrix(lines)
    starting_position, = [int(x) for _, x in np.argwhere(grid == "S")]

    beam = QuantumTachyonBeam(starting_position, grid)
    while not beam.is_accepting():
        beam.travel()

    return beam.sum_timelines()


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
