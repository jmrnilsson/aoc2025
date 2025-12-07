import sys
from typing import Set, Tuple

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


class TachyonBeam:
    x_range: Set[int]
    splits = 0
    y: int

    def __init__(self, starting_pos: Tuple[int, int], grid):
        self.y = 0
        self.x_range = {starting_pos[1]}
        self.grid = grid

    def travel(self):
        current_x_range = list(self.x_range)
        self.x_range.clear()
        self.y += 1

        while current_x_range:
            x: int = current_x_range.pop(0)
            if self.grid[(self.y, x)] == "^":
                self.splits += 1
                if (left := x - 1) > -1:
                    self.x_range.add(left)
                if (right := x + 1) < self.grid.shape[1]:
                    self.x_range.add(right)
            else:
                self.x_range.add(x)

    def count_splits(self):
        return self.splits

    def is_accepting(self):
        return self.y + 1 == self.grid.shape[0]


def solve_(__input=None):
    """
    :challenge: 21
    :expect: 1594
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(map(str, list(line))))

    grid = np.matrix(lines)
    starting_positions_, = [[(int(y), int(x))] for y, x in np.argwhere(grid == "S")]
    starting_position, *_ = starting_positions_

    beam = TachyonBeam(starting_position, grid)
    while not beam.is_accepting():
        beam.travel()

    return beam.count_splits()


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
