# 🎄 Advent of Code 2024 😅

## Install
```bash
python.exe -m venv .venv
python.exe -m pip install --upgrade pip
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage
- Run `python3 year_2023/day_07/solve.py`
- Generate README: `python aoc/template.py`

## Focus for 2024

- Explore Automata theory. Any of the following PDS, DFA or NFA.
- Stay close to the language used in the puzzles.
- Attempt a PyTorch GPU brute at some point.
- Try a pathfinding concept aside A*, Dijkstra, BFS and DFS. Perhaps Bi-directional Search, JPS, D-Lite**, Theta*,
  Bellman-Ford or Floyd-Warshall.


| Type | Algorithm                                         |
|-------|---------------------------------------------------|
| Grid-based games or simulations:| A*, JPS, Theta*                                   |
| Dynamic environments: | D* or [D*-Lite](https://en.wikipedia.org/wiki/D*) |
| Unweighted graphs: | BFS                                               |
| Weighted graphs: | Dijkstra or A*                                    |
| Negative weights: |  [Bellman-Ford](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)                                      |
| Memory constraints:| [IDA*](https://en.wikipedia.org/wiki/Iterative_deepening_A*)                                          |
| All-pairs shortest paths:| [Floyd-Warshall](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)                                    |


## year_2025\day_06\solve_2.py

```py
import operator
import re
import sys
from functools import reduce
from typing import List, Tuple
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(30_000)

def solve_(__input=None):
    """
    A version that solves this with regex matches. Check for a rewrite named solve_2_transpose.py for a
    version solving this with transpose instead.
    :challenge: 3263827
    :expect: 8342588849093
    """

    lines = []
    with open(locate(__input), "r") as fp:
        lines = list(read_lines(fp))

    match_ranges: List[Tuple[slice, str]] = []
    matches = [m for m in re.finditer(r"\S+", lines[-1])]
    matches_rev = reversed(matches)
    last_start = len(lines[-1]) + 2
    operator_lookup = {'*': operator.mul, '+': operator.add, '-': operator.sub}

    for match in matches_rev:
        match_ranges.append((slice(match.start(), last_start - 1), match.group(0)))
        last_start = match.end()

    total = 0
    for ms, ope in match_ranges:
        numbers: List[int] = []

        for j in range(ms.start, ms.stop - 1):
            number = reduce(lambda acc, i: acc + lines[i][j].replace(" ", ""), range(0, len(lines) - 1), "")
            numbers.append(int(number))

        total += reduce(operator_lookup[ope], map(int, numbers))

    return total

```
## year_2025\day_06\solve_1.py

```py
import operator
import re
import sys
from functools import reduce
from defaultlist import defaultlist
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(30_000)

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

```
## year_2025\day_05\solve_2.py

```py
import sys
from typing import List
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(30_000)

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

```
## year_2025\day_05\solve_1.py

```py
import sys
from typing import List, Tuple
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(30_000)

def solve_(__input=None):
    """
    :challenge: 3
    :expect: 701
    """
    fresh_ingredient_range: List[Tuple[int, ...]] = []
    fruits = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            if "-" in line:
                fresh_ingredient_range.append(tuple(map(int, line.split("-"))))
            else:
                fruits.append(int(line))

    return sum(1 for f in fruits if any(1 for begin, end in fresh_ingredient_range if begin < f <= end))

```
## year_2025\day_04\solve_2.py

```py
import itertools
import sys
from bisect import insort
from typing import Generator, List, Tuple
import numpy as np
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(30_000)

def moore_neighborhood(y_x: Tuple[int, int], shape: np.shape) -> Generator[Tuple[int, int], None, None]:
    moore_neigh = [(1, 0), (0, 1), (-1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dy, dx in moore_neigh:
        neighbor = y_x[0] + dy, y_x[1] + dx

        if -1 < neighbor[0] < shape[0] and -1 < neighbor[1] < shape[1]:
            yield neighbor

def solve_(__input=None):
    """
    :challenge: 43
    :expect: 8366
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(map(str, list(line))))

    shape = (len(lines[0]), len(lines))
    storage = np.zeros(shape, dtype=int)

    for y, x in itertools.product(range(0, shape[0]), range(0, shape[1])):
        if lines[y][x] == "@":
            storage[(y, x)] = 1

    where: List[Tuple[int, int]] = sorted((y, x) for y, x in np.argwhere(storage == 1))
    removed_rolls = 0
    last_mnemonic = tuple()

    while 1:
        if len(where) < 1:
            for y, x in np.argwhere(storage == 1):
                insort(where, (y, x))

            if last_mnemonic == (current_mnemonic := tuple(where)):
                break

            if len(where) < 1:
                break

            last_mnemonic = current_mnemonic

        fork_lift: Tuple[int, int] = where.pop()
        roll_count = sum(1 for neigh in moore_neighborhood(fork_lift, shape) if storage[neigh] > 0)

        if roll_count < 4:
            removed_rolls += 1
            storage[fork_lift] = 0

    return removed_rolls

```
## year_2025\day_04\solve_1.py

```py
import sys
from typing import Tuple
import numpy as np
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(30_000)

def moore_neighborhood(y_x: Tuple[int, int], shape: np.shape):
        neighbors_ = list()
        moore_neigh = [(1, 0), (0, 1), (-1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dy, dx in moore_neigh:
            neighbor = y_x[0] + dy, y_x[1] + dx

            if -1 < neighbor[0] < shape[0] and -1 < neighbor[1] < shape[1]:
                neighbors_.append(neighbor)

        return neighbors_

def solve_(__input=None):
    """
    :challenge: 13
    :expect: 1409
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(map(str, list(line))))

    shape = (len(lines[0]), len(lines))
    storage = np.zeros(shape, dtype=int)

    for y in range(0, len(lines)):
        for x in range(0, len(lines)):
            if lines[y][x] == "@":
                storage[(y, x)] = 1

    where = np.argwhere(storage == 1)

    fork_lift_access = 0
    for y, x in where:
        neighborhood = moore_neighborhood((y, x), shape)
        current_sum = 0
        for neigh in neighborhood:
            value = storage[neigh]

            if value > 0:
                current_sum += 1

        if current_sum < 4:
            fork_lift_access += 1

    return fork_lift_access

```
## year_2025\day_03\solve_2.py

```py
import sys
from typing import List
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(10_000)

def max_jolt(accumulate: str, number: List[int], k, start, end_at):
    best, best_at = -1, -1
    i = start
    next_end_at = min(len(number), end_at + 1)
    while i < end_at:

        current = number[i]
        if current > best:
            best = current
            best_at = i

        i += 1

    accumulate += str(best)

    if len(accumulate) == k:
        return accumulate
    else:
        return max_jolt(accumulate, number, k, best_at + 1, next_end_at)

def solve_(__input=None):
    """
    :challenge: 3121910778619
    :expect: 173065202451341
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(map(int, list(line))))

    jolts = []
    k = 12
    for word in lines:
        end_at = len(word) - k + 1
        jolts.append(max_jolt("", word, k, 0, end_at))

    return sum(int(j) for j in jolts)

```
## year_2025\day_03\solve_1.py

```py
import sys
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(30_000)

def solve_(__input=None):
    """
    :challenge: 357
    :expect: 17432
    """
    lines = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            lines.append(list(map(int, list(line))))

    jolts = []
    for l in lines:
        max_ab = 0
        for i, a in enumerate(l):
            max_b = 0
            for j in range(i, len(l)):
                # for j, b in enumerate(l):
                if i == j:
                    continue

                b = l[j]

                if b > max_b:
                    max_b = b

            ab = int(str(a) + str(max_b))
            if str(ab).count("0"):
                continue

            if ab > max_ab:
                max_ab = ab

        jolts.append(max_ab)

    return sum(int(j) for j in jolts)

```
## year_2025\day_02\solve_2.py

```py
import sys
from typing import List, Tuple
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(30_000)

def solve_(__input=None):
    """
    :challenge: 4174379265
    :expect: 50857215650
    """
    lines: List[Tuple[int, ...]] = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            for pair in filter(lambda p: len(p) > 0, line.split(",")):
                lines.append(tuple(map(int, list(pair.split("-")))))

    numbers = []
    for x, y in lines:
        for n in range(x, y + 1):
            m = str(n)
            len_m = len(m)

            for split_len in range(1, len_m // 2 + 1):
                if len_m % split_len == 1:
                    continue

                word = m[0: split_len]
                counter = m.count(word)

                if counter * split_len == len_m:
                    numbers.append(n)
                    break

    return sum(numbers)

```
## year_2025\day_02\solve_1.py

```py
import sys
from typing import List, Tuple
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(30_000)

def solve_(__input=None):
    """
    :challenge: 1227775554
    :expect: 40055209690
    """
    lines: List[Tuple[int, ...]] = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            for pair in filter(lambda p: len(p) > 0, line.split(",")):
                lines.append(tuple(map(int, list(pair.split("-")))))

    total = 0
    for x, y in lines:
        for n in range(x, y + 1):
            m = str(n)
            if (len_m := len(m)) == 0:
                continue

            split_at = len_m // 2

            c, d = m[0: split_at], m[split_at:]
            if c == d:
                total += n

    return total

```
## year_2025\day_01\solve_2.py

```py
import sys
from typing import List, Tuple
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(30_000)

def solve_(__input=None):
    """
    :challenge: 6
    :expect: 5887
    """
    lines: List[Tuple[str, int]] = []
    with open(locate(__input), "r") as fp:
        for line in read_lines(fp):
            _dir, *_dist = list(line)
            lines.append((_dir, int("".join(_dist))))

    mod, pos, n = 100, 50, 0
    for direction, distance in lines:
        for i in range(0, distance):
            if direction == "L":
                pos -= 1
                pos %= mod
            else:
                pos += 1
                pos %= mod

            if pos == 0:
                n += 1

    return n

```
## year_2025\day_01\solve_1.py

```py
import sys
from typing import List, Tuple
from aoc.helpers import build_location, locate, read_lines
from aoc.printer import ANSIColors, get_meta_from_fn, print2


sys.setrecursionlimit(30_000)

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
            pos %= mod
        else:
            pos += distance
            pos = pos % mod

        if pos == 0:
            n += 1

    return n

```
