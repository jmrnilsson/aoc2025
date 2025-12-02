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

- Explore Automatons preferably NFA but also a DFA.
  - Focus transition of states and accepting states.
  - Try to find something for push-down Automata.
  - Stay close to the language used in the puzzles.
- Attempt a CUDA brute at some point.
- Try a pathfinding concept aside A*, Dijkstra, BFS and DFS. Perhaps Bidirectional Search, JPS, D-Lite**, Theta*,
  Bellman-Ford or Floyd-Warshall.


| Type | Algorithm                                         |
|-------|---------------------------------------------------|
|Grid-based games or simulations:| A*, JPS, Theta*                                   |
|Dynamic environments:| D* or [D*-Lite](https://en.wikipedia.org/wiki/D*) |
|Unweighted graphs:| BFS                                               |
|Weighted graphs:| Dijkstra or A*                                    |
|Negative weights:| [Bellman-Ford](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)                                      |
|Memory constraints:| [IDA*](https://en.wikipedia.org/wiki/Iterative_deepening_A*)                                          |
|All-pairs shortest paths:| [Floyd-Warshall](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)                                    |


## year_2025\day_02\solve_2.py

```py
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


sys.setrecursionlimit(30_000)

class HikingAutomaton:
    trails: List[List[Tuple[int, int]]]
    done: List[List[Tuple[int, int]]]

    def __init__(self, starting_pos: List[List[Tuple[int, int]]], grid):
        self.trails = list(starting_pos)
        self.done = []
        self.grid = grid

    def in_bound(self, y: int, x: int) -> bool:
        return -1 < y < self.grid.shape[0] and -1 < x < self.grid.shape[1]

    def _step(self, y, x) -> Generator[Tuple[int, int, int], None, None]:
        steps = (
            (y - 1, x + 0),
            (y + 0, x + 1),
            (y + 1, x + 0),
            (y + 0, x + -1)
        )
        for step in steps:
            if self.in_bound(*step) and (value := self.grid[step]) == self.grid[y, x] + 1:
                yield *step, value

    def walk(self):
        trails = list(self.trails)
        self.trails.clear()
        for n, trail_ in enumerate(trails):
            last_step = trail_[-1]
            for y, x, value in self._step(*last_step):
                trail = deepcopy(trail_)
                trail.append((y, x))
                if value == 9:
                    self.done.append(trail)
                else:
                    self.trails.append(trail)

    def score(self):
        head_and_tails = {
            (trail[0], trail[-1])
            for trail in self.done
        }
        return len(head_and_tails)

    def rat(self):
        return len(self.done)

    def is_accepting(self):
        return len(self.trails) == 0

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


sys.setrecursionlimit(30_000)

class HikingAutomaton:
    trails: List[List[Tuple[int, int]]]
    done: List[List[Tuple[int, int]]]

    def __init__(self, starting_pos: List[List[Tuple[int, int]]], grid):
        self.trails = list(starting_pos)
        self.done = []
        self.grid = grid

    def in_bound(self, y: int, x: int) -> bool:
        return -1 < y < self.grid.shape[0] and -1 < x < self.grid.shape[1]

    def _step(self, y, x) -> Generator[Tuple[int, int, int], None, None]:
        steps = (
            (y - 1, x + 0),
            (y + 0, x + 1),
            (y + 1, x + 0),
            (y + 0, x + -1)
        )
        for step in steps:
            if self.in_bound(*step) and (value := self.grid[step]) == self.grid[y, x] + 1:
                yield *step, value

    def walk(self):
        trails = list(self.trails)
        self.trails.clear()
        for n, trail_ in enumerate(trails):
            last_step = trail_[-1]
            for y, x, value in self._step(*last_step):
                trail = deepcopy(trail_)
                trail.append((y, x))
                if value == 9:
                    self.done.append(trail)
                else:
                    self.trails.append(trail)

    def score(self):
        head_and_tails = {
            (trail[0], trail[-1])
            for trail in self.done
        }
        return len(head_and_tails)

    def rat(self):
        return len(self.done)

    def is_accepting(self):
        return len(self.trails) == 0

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
import numpy as np
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
