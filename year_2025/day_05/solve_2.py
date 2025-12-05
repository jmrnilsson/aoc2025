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
    Python slices are immutable. Use this instead.
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

    def __str__(self):
        return f"[{self.begin}, {self.end}]"


def compress(current: List[MutableSlice]) -> List[MutableSlice]:
    others: List[MutableSlice] = list()
    while current:
        current_slice = current.pop(0)
        begin, end = current_slice.begin, current_slice.end
        mutated_existing = False

        for other in others:
            other_begin, other_end = other.begin, other.end

            begin_in_range = other_begin <= begin <= other_end
            end_in_range = other_begin <= end <= other_end
            other_begin_in_range = begin <= other_begin <= end
            other_end_in_range = begin <= other_end <= end

            # outside - adjust both begin and end
            if other_begin_in_range and other_end_in_range:
                other.begin = begin
                other.end = end
                mutated_existing = True
                break

            # inside - skip
            if begin_in_range and end_in_range:
                mutated_existing = True
                break

            # overlap - adjust end
            elif begin_in_range:
                mutated_existing = True
                other.end = max(end, other_end)
                break

            # overlap - adjust begin
            elif other_begin_in_range:
                mutated_existing = True
                other.begin = min(begin, other_begin)
                break

        if not mutated_existing:
            others.append(MutableSlice(begin, end))

    return others


def solve_(__input=None):
    """
    :challenge: 14
    :expect: 352340558684863
    """
    inbound = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            if "-" in line:
                a, b = line.split("-")
                inbound.append(MutableSlice(int(a), int(b)))

    mnemonic = set()
    while 1:
        outbound = compress(inbound)

        if mnemonic == (current_mnemonic := set(outbound)):
            inbound = outbound
            break

        mnemonic = current_mnemonic
        inbound = outbound

    return sum(i.end - i.begin + 1 for i in inbound)


if __name__ == "__main__":
    challenge = get_meta_from_fn(solve_, "challenge")
    expect = get_meta_from_fn(solve_, "expect")
    print2(solve_, test_input, challenge)
    print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
