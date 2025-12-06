from typing import List

import pytest

from aoc.tests.test_fixtures import make_fixture, AdventFixture
from . import solve_1 as solution_1
from .sink import solve_2 as solution_2, solve_2_bottom_up as solve_2_bottom_up_fix_up, \
    solve_2_transpose as solve_2_transpose


class TestAdvent202405:

    @pytest.fixture
    def fixt(self) -> List[AdventFixture]:
        return [None, make_fixture(solution_1.solve_), make_fixture(solution_2.solve_)]

    def test_part_1_challenge(self, fixt: List[AdventFixture]):
        assert solution_1.solve_(solution_1.test_input) == int(fixt[1].challenge)

    def test_part_1_expected(self, fixt: List[AdventFixture]):
        assert solution_1.solve_(solution_1.puzzle_input) == int(fixt[1].expect)

    def test_part_2_challenge(self, fixt: List[AdventFixture]):
        assert solution_2.solve_(solution_2.test_input) == int(fixt[2].challenge)

    def test_part_2_expected(self, fixt: List[AdventFixture]):
        assert solution_2.solve_(solution_2.puzzle_input) == int(fixt[2].expect)

    def test_part_2_challenge_bottom_up(self, fixt: List[AdventFixture]):
        assert solve_2_bottom_up_fix_up.solve_(solution_2.test_input) == int(fixt[2].challenge)

    def test_part_2_expected_bottom_up(self, fixt: List[AdventFixture]):
        assert solve_2_bottom_up_fix_up.solve_(solution_2.puzzle_input) == int(fixt[2].expect)

    def test_part_2_challenge_transpose(self, fixt: List[AdventFixture]):
        assert solve_2_transpose.solve_(solution_2.test_input) == int(fixt[2].challenge)

    def test_part_2_expected_transpose(self, fixt: List[AdventFixture]):
        assert solve_2_transpose.solve_(solution_2.puzzle_input) == int(fixt[2].expect)
