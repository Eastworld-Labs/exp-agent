"""The counter scene, in a unit test.

Every case here is the same room: the robot faces a counter, and the object is
on the floor behind it. That scene is the reason `search.py` exists, so it is
the scene the tests are written in rather than an abstract grid.
"""
from __future__ import annotations

import math
import unittest

from system2_agent.local_planner import COST_FREE, COST_LETHAL, CostGrid
from system2_agent.search import (
    VisibilityMap,
    image_fraction_for,
    standpoints,
    sweep_yaws,
)

HFOV = math.radians(87.0)
RANGE_M = 6.0

# The room: 8 x 8 m at 0.05 m, origin at the corner, robot at the centre (4, 4)
# facing +x. A counter 2.4 m wide and 0.6 m deep sits 2 m in front of it.
RES = 0.05
CELLS = 160
ROBOT = (4.0, 4.0, 0.0)
COUNTER_X = (6.0, 6.6)
COUNTER_Y = (2.8, 5.2)


def room(counter: bool = True) -> CostGrid:
    cost = [COST_FREE] * (CELLS * CELLS)

    def fill(x0: float, x1: float, y0: float, y1: float, value: int) -> None:
        for row in range(int(y0 / RES), int(y1 / RES)):
            for col in range(int(x0 / RES), int(x1 / RES)):
                cost[row * CELLS + col] = value

    # The room's own walls, so rays terminate somewhere even without a counter.
    fill(0.0, 8.0, 0.0, 0.1, COST_LETHAL)
    fill(0.0, 8.0, 7.9, 8.0, COST_LETHAL)
    fill(0.0, 0.1, 0.0, 8.0, COST_LETHAL)
    fill(7.9, 8.0, 0.0, 8.0, COST_LETHAL)
    if counter:
        fill(COUNTER_X[0], COUNTER_X[1], COUNTER_Y[0], COUNTER_Y[1], COST_LETHAL)
    return CostGrid(
        frame="odom",
        width=CELLS,
        height=CELLS,
        resolution=RES,
        origin_x=0.0,
        origin_y=0.0,
        cost=cost,
    )


def looked_from_start(grid: CostGrid, radius_m: float = 4.0) -> VisibilityMap:
    """The robot has arrived and looked straight ahead. Nothing more."""
    visibility = VisibilityMap.around(ROBOT[0], ROBOT[1], radius_m)
    visibility.observe(grid, *ROBOT, hfov_rad=HFOV, max_range_m=RANGE_M)
    return visibility


class TestVisibility(unittest.TestCase):
    def test_the_counter_stops_the_view(self) -> None:
        """The floor just in front of the counter is seen; just behind is not.

        This is the property the whole module rests on. If it ever fails, the
        search will happily propose standing in a place nothing has sensed.
        """
        visibility = looked_from_start(room())
        self.assertTrue(
            visibility.is_seen(5.5, 4.0),
            "the floor between the robot and the counter must be observed",
        )
        self.assertFalse(
            visibility.is_seen(7.0, 4.0),
            "the floor BEHIND the counter must stay unobserved",
        )

    def test_without_the_counter_the_same_floor_is_seen(self) -> None:
        """Proves the previous test measures occlusion, not range or the cone."""
        visibility = looked_from_start(room(counter=False))
        self.assertTrue(visibility.is_seen(7.0, 4.0))

    def test_unobserved_is_not_the_same_as_occupied(self) -> None:
        """The hidden strip reads FREE on the costmap. That is the trap."""
        grid = room()
        self.assertEqual(grid.cost_at(7.0, 4.0), COST_FREE)
        self.assertFalse(looked_from_start(grid).is_seen(7.0, 4.0))

    def test_observing_twice_is_news_only_once(self) -> None:
        grid = room()
        visibility = VisibilityMap.around(ROBOT[0], ROBOT[1], 4.0)
        first = visibility.observe(grid, *ROBOT, hfov_rad=HFOV, max_range_m=RANGE_M)
        second = visibility.observe(grid, *ROBOT, hfov_rad=HFOV, max_range_m=RANGE_M)
        self.assertGreater(first, 0)
        self.assertEqual(second, 0)

    def test_nothing_behind_the_robot_is_seen(self) -> None:
        visibility = looked_from_start(room())
        self.assertFalse(visibility.is_seen(2.0, 4.0))

    def test_a_ray_stops_at_the_costmap_edge(self) -> None:
        """Off the window, nothing says whether the view is blocked.

        A map wider than the costmap must NOT come back claiming the robot has
        seen the parts of itself the costmap never described.
        """
        grid = CostGrid(
            frame="odom",
            width=40,
            height=40,
            resolution=RES,
            origin_x=3.0,
            origin_y=3.0,
            cost=[COST_FREE] * 1600,
        )
        visibility = VisibilityMap.around(ROBOT[0], ROBOT[1], 4.0)
        visibility.observe(grid, *ROBOT, hfov_rad=HFOV, max_range_m=RANGE_M)
        self.assertTrue(visibility.is_seen(4.5, 4.0))
        self.assertFalse(visibility.is_seen(6.0, 4.0), "past the window edge")


class TestFrontier(unittest.TestCase):
    def test_the_frontier_forms_at_the_counter_ends(self) -> None:
        """Not in the middle of the counter: that floor is solid, not hidden."""
        visibility = looked_from_start(room())
        edge = [visibility.cell_centre(cell) for cell in visibility.frontier()]
        self.assertTrue(edge, "a partly-observed room must have a frontier")
        near_left = [p for p in edge if p[1] < COUNTER_Y[0] and p[0] > 5.0]
        near_right = [p for p in edge if p[1] > COUNTER_Y[1] and p[0] > 5.0]
        self.assertTrue(near_left, "no frontier off the counter's near end")
        self.assertTrue(near_right, "no frontier off the counter's far end")

    def test_a_fully_observed_map_has_no_frontier(self) -> None:
        visibility = VisibilityMap.around(0.0, 0.0, 1.0)
        visibility.seen = bytearray([1] * len(visibility.seen))
        self.assertEqual(visibility.frontier(), [])


class TestStandpoints(unittest.TestCase):
    def candidates(self, radius_m: float = 4.0, **kwargs):
        grid = room()
        visibility = looked_from_start(grid, radius_m)
        free = grid.to_gridmap().inflated(0.35)
        return visibility, standpoints(
            visibility,
            grid,
            free,
            ROBOT,
            anchor=(ROBOT[0], ROBOT[1]),
            radius_m=radius_m,
            hfov_rad=HFOV,
            max_range_m=RANGE_M,
            **kwargs,
        )

    def test_it_proposes_going_round_the_counter(self) -> None:
        _, found = self.candidates()
        self.assertTrue(found, "the counter scene must yield somewhere to look")
        self.assertTrue(
            all(c.gain_cells > 0 for c in found),
            "a candidate that reveals nothing is not a candidate",
        )

    def test_no_candidate_stands_in_unobserved_space(self) -> None:
        """The single most important refusal in the module."""
        visibility, found = self.candidates()
        for candidate in found:
            self.assertTrue(
                visibility.is_seen(candidate.x, candidate.y),
                f"candidate at ({candidate.x:.2f}, {candidate.y:.2f}) has never been sensed",
            )

    def test_no_candidate_is_inside_the_counter(self) -> None:
        _, found = self.candidates()
        for candidate in found:
            inside = (
                COUNTER_X[0] <= candidate.x <= COUNTER_X[1]
                and COUNTER_Y[0] <= candidate.y <= COUNTER_Y[1]
            )
            self.assertFalse(inside, "a candidate inside the counter")

    def test_candidates_stay_inside_the_radius(self) -> None:
        _, found = self.candidates(radius_m=1.5)
        for candidate in found:
            self.assertLessEqual(
                math.hypot(candidate.x - ROBOT[0], candidate.y - ROBOT[1]),
                1.5 + 1e-6,
            )

    def test_best_first(self) -> None:
        _, found = self.candidates()
        gains = [c.gain_cells for c in found]
        self.assertEqual(gains, sorted(gains, reverse=True))

    def test_visited_standpoints_are_not_offered_again(self) -> None:
        _, first = self.candidates()
        self.assertTrue(first)
        been = [(first[0].x, first[0].y)]
        _, again = self.candidates(visited=been)
        for candidate in again:
            self.assertGreaterEqual(
                math.hypot(candidate.x - been[0][0], candidate.y - been[0][1]),
                0.8,
            )

    def test_an_exhausted_area_yields_nothing(self) -> None:
        grid = room()
        visibility = VisibilityMap.around(ROBOT[0], ROBOT[1], 4.0)
        visibility.seen = bytearray([1] * len(visibility.seen))
        free = grid.to_gridmap().inflated(0.35)
        self.assertEqual(
            standpoints(
                visibility, grid, free, ROBOT,
                anchor=(ROBOT[0], ROBOT[1]), radius_m=4.0,
                hfov_rad=HFOV, max_range_m=RANGE_M,
            ),
            [],
        )

    def test_candidates_are_separated(self) -> None:
        _, found = self.candidates(min_separation_m=1.0)
        for i, a in enumerate(found):
            for b in found[i + 1:]:
                self.assertGreaterEqual(math.hypot(a.x - b.x, a.y - b.y), 1.0)


class TestFraming(unittest.TestCase):
    def test_a_candidate_behind_the_robot_is_not_in_the_picture(self) -> None:
        self.assertIsNone(image_fraction_for(math.pi, 0.0, HFOV))

    def test_left_of_centre_is_left_of_the_picture(self) -> None:
        left = image_fraction_for(math.radians(30), 0.0, HFOV)
        self.assertIsNotNone(left)
        self.assertLess(left, 0.5)

    def test_dead_ahead_is_the_middle(self) -> None:
        self.assertAlmostEqual(image_fraction_for(0.0, 0.0, HFOV), 0.5)


class TestSweep(unittest.TestCase):
    def test_a_sweep_does_not_re_photograph_the_current_heading_first(self) -> None:
        yaws = sweep_yaws(0.0)
        self.assertNotAlmostEqual(yaws[0], 0.0)
        self.assertAlmostEqual(yaws[-1], 0.0, places=6)

    def test_a_sweep_covers_the_circle(self) -> None:
        self.assertEqual(len(sweep_yaws(0.0, quarters=4)), 4)
        self.assertEqual(len(set(round(y, 6) for y in sweep_yaws(1.0, quarters=6))), 6)

    def test_a_sweep_actually_reveals_what_was_behind(self) -> None:
        grid = room()
        visibility = looked_from_start(grid)
        self.assertFalse(visibility.is_seen(2.0, 4.0))
        for yaw in sweep_yaws(ROBOT[2]):
            visibility.observe(grid, ROBOT[0], ROBOT[1], yaw, hfov_rad=HFOV, max_range_m=RANGE_M)
        self.assertTrue(visibility.is_seen(2.0, 4.0))
        self.assertFalse(
            visibility.is_seen(7.0, 4.0),
            "turning on the spot cannot see behind the counter -- that needs a walk",
        )


if __name__ == "__main__":
    unittest.main()
