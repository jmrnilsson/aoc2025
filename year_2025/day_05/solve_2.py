import sys
from typing import List

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


class MutableSlice:
    """
    A mutable replacement for Python's built-in slice objects, which are immutable.
    This class allows the `begin` and `end` bounds to be updated in place, avoiding
    the need to create new slices or rebuild data structures that reference them.
    """
    begin: int
    end: int

    def __init__(self, begin: int, end: int):
        self.begin = begin
        self.end = end

    def __repr__(self):
        return f"Slice({self.begin}, {self.end})"

    def __hash__(self):
        hashable = (self.begin, self.end)
        return hash(hashable)

    def __eq__(self, value):
        return value.begin == self.begin and value.end == self.end


def compress(current: List[MutableSlice]) -> List[MutableSlice]:
    others: List[MutableSlice] = list()
    while current:
        current_slice = current.pop(0)
        cb, ce = current_slice.begin, current_slice.end
        mutated_existing = False

        for other in others:
            ob, oe = other.begin, other.end

            # outside - adjust both begin and end
            if cb <= ob and oe <= ce:
                other.begin = cb
                other.end = ce
                mutated_existing = True
                break

            # inside - skip
            if ob <= cb and ce <= oe:
                mutated_existing = True
                break

            # overlap - adjust end
            elif ob <= cb <= oe:
                mutated_existing = True
                other.end = max(ce, oe)
                break

            # overlap - adjust begin
            elif cb <= ob <= ce:
                mutated_existing = True
                other.begin = min(cb, ob)
                break

        if not mutated_existing:
            others.append(MutableSlice(cb, ce))

    return others


def solve_(__input=None):
    """
    :challenge: 14
    :expect: 352340558684863
    """
    fresh_ids = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            if "-" in line:
                a, b = line.split("-")
                fresh_ids.append(MutableSlice(int(a), int(b)))

    mnemonic = set()
    while 1:
        fresh_ids = compress(fresh_ids)

        if mnemonic == (current_mnemonic := set(fresh_ids)):
            break

        mnemonic = current_mnemonic

    return sum(i.end - i.begin + 1 for i in fresh_ids)


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
