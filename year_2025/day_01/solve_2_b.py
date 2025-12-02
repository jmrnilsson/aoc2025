import sys
from typing import List, Tuple

import numpy as np

from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2

# from numba import cuda, np as npc, vectorize

# requires https://developer.nvidia.com/cuda-downloads CUDA toolkit. There are some sketchy versions in pip, but it's
# almost impossible to find the right versions.
# - pip install nvidia-pyindex
# - pip install nvidia-cuda-runtime-cuXX
# python -m numba -s
# print(numba.__version__)
# print(cuda.gpus)
# print(cuda.detect())

# if cuda.is_available():
#     print("CUDA is available!")
#     print("Device:", cuda.get_current_device().name)
# else:
#     print("CUDA is not available.")

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
    :challenge: 3
    :expect: 969
    """
    lines: List[Tuple[str, int]] = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            _dir, *_dist = list(line)
            lines.append((_dir, int("".join(_dist))))

    mod, pos, n = 100, 50, 0
    for direction, distance in lines:
        if direction == "L":
            pos -= distance
            n += pos // mod
            pos %= mod
        else:
            pos += distance
            n += pos // mod
            pos = pos % mod

        if pos == 0:
            n += 1

    return n

if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
