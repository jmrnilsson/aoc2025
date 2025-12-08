import heapq
import itertools
import math
import sys
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
    :challenge: 25272
    :expect: 42047840
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            jb = line.split(",")
            lines.append(tuple(map(int, jb)))

    pairs = []
    for left, right in itertools.combinations(lines, 2):
        pairs.append((math.dist(left, right), left, right))

    # DSU is probably preferable but roughly the same performance
    heapq.heapify(pairs)
    circuits: List[Set[JunctionBox]] = []
    lookup: Dict[JunctionBox, Set[JunctionBox]] = {}
    last: None | Tuple[JunctionBox, JunctionBox] = None

    for line in lines:
        circuit = {line}
        lookup[line] = circuit
        circuits.append(circuit)

    while sum(1 for f in circuits if len(f) > 0) > 1:
        _, a, b = heapq.heappop(pairs)

        circuit_a = lookup[a]
        circuit_b = lookup[b]

        if circuit_a is circuit_b:
            continue

        last = a, b

        for swap in list(circuit_b):
            circuit_a.add(swap)
            lookup[swap] = circuit_a

        circuit_b.clear()

    return last[0][0] * last[1][0]


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
