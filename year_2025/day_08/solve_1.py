import heapq
import itertools
import math
import operator
import sys
from functools import reduce
from typing import Dict, List, Set, Tuple

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

type JunctionBox = Tuple[int, int, int]


def solve_(__input=None):
    """
    :challenge: 40
    :expect: 129564
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            jb = line.split(",")
            lines.append(tuple(map(int, jb)))

    pairs = []
    for left, right in itertools.combinations(lines, 2):
        pairs.append((math.dist(left, right), left, right))

    heapq.heapify(pairs)

    # DSU is probably preferable but roughly the same performance
    k = 10 if "test" in __input else 1000
    heapq.heapify(pairs)
    circuits: List[Set[JunctionBox]] = []
    lookup: Dict[JunctionBox, Set[JunctionBox]] = {}

    for line in lines:
        circuit = {line}
        lookup[line] = circuit
        circuits.append(circuit)

    while k > 0:
        k -= 1
        _, a, b = heapq.heappop(pairs)

        circuit_a = lookup[a]
        circuit_b = lookup[b]

        if circuit_a is circuit_b:
            continue

        for swap in list(circuit_b):
            circuit_a.add(swap)
            lookup[swap] = circuit_a

        circuit_b.clear()

    circuits_lengths = [len(f) for f in circuits]
    return reduce(operator.mul, heapq.nlargest(3, circuits_lengths))


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
