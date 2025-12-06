import itertools
import operator
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Callable, Tuple, Literal, Set, Generator, Any

import more_itertools
# import numba
import numpy as np
from defaultlist import defaultlist
from more_itertools import windowed, chunked
from more_itertools.recipes import sliding_window

from aoc.helpers import locate, build_location, read_lines
from aoc.poll_printer import PollPrinter
from aoc.printer import get_meta_from_fn, print_, ANSIColors, print2
from aoc.tests.test_fixtures import get_challenges_from_meta
from aoc.tools import transpose
from year_2021.day_05 import direction

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




































