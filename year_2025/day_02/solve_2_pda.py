# import sys
# from enum import Enum
# from typing import List, Tuple
#
# from aoc.helpers import build_location, locate, read_lines
# from aoc.printer import ANSIColors, get_meta_from_fn, print2
# from year_2021.day_18_part_2 import split
#
# sys.setrecursionlimit(30_000)
#
# _default_puzzle_input = "year_2024/day_01/puzzle.txt"
# _default_test_input = "year_2024/day_01/test.txt"
#
# puzzle_input = build_location(__file__, "puzzle.txt")
# test_input = build_location(__file__, "test.txt")
# test_input_2 = build_location(__file__, "test_2.txt")
# test_input_3 = build_location(__file__, "test_3.txt")
# test_input_4 = build_location(__file__, "test_4.txt")
# test_input_5 = build_location(__file__, "test_5.txt")
#
#
# class Result(Enum):
#     Empty = 1,
#     Continue = 2,
#     Done = 3
#
# class PDA:
#     n: int
#     left: int
#     left_text: str
#     right: int
#     right_text: str
#     open: List[str]
#     generate_next = False
#
#     def __init__(self, left: int, right: int):
#         self.left = left
#         self.left_text = str(left)
#         self.right = right
#         self.end_at = 0
#         self.open = []
#         self.n = 0
#         self.generate_next = False
#
#     def transition(self) -> Result:
#         if len(self.open) < 0:
#             return Result.Empty
#
#         self.n += 1
#
#         if not self.generate_next and self.left_text[0: self.n] < self.right_text[0: self.n]:
#             self.generate_next = True
#
#         if self.generate_next:
#             if
#
#
#         self.open.append(wo)
#
#
#
#         if this.reached_end:
#             for still_open in open:
#                 valid.append(still_open)
#
#             self.end_at += 1
#
#         self.a
#
#
#
#     def accepting(self):
#         if
#         for split_len in range(1, len_m // 2 + 1):
#
#
# def solve_(__input=None):
#     """
#     :challenge: 4174379265
#     :expect: 50857215650
#     """
#     lines: List[Tuple[int, ...]] = []
#     with open(locate(__input), "r") as fp:
#         for line in read_lines(fp):
#             for pair in filter(lambda p: len(p) > 0, line.split(",")):
#                 lines.append(tuple(map(int, list(pair.split("-")))))
#
#     numbers = []
#     for x, y in lines:
#         for n in range(x, y + 1):
#             m = str(n)
#             len_m = len(m)
#
#             for split_len in range(1, len_m // 2 + 1):
#                 if len_m % split_len == 1:
#                     continue
#
#                 word = m[0: split_len]
#                 counter = m.count(word)
#
#                 if counter * split_len == len_m:
#                     numbers.append(n)
#                     break
#
#     return sum(numbers)
#
#
# if __name__ == "__main__":
#     challenge = get_meta_from_fn(solve_, "challenge")
#     expect = get_meta_from_fn(solve_, "expect")
#     print2(solve_, test_input, challenge)
#     print2(solve_, puzzle_input, expect, ANSIColors.OK_GREEN)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
