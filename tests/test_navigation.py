import math
import unittest

from system2_agent.modules import Pose3D
from system2_agent.navigation_core import (
    AStarPlanner,
    GridMap,
    PathFollower,
    PlannedNavigationBackend,
    VelocityCommand,
)
from system2_agent.sonic_bridge import pack_sonic_planner


class FakeBase:
    name = "fake"

    def __init__(self):
        self.value = Pose3D(0.0, 0.0)

    def pose(self):
        return self.value

    def command_velocity(self, command: VelocityCommand, dt: float):
        c, s = math.cos(self.value.yaw), math.sin(self.value.yaw)
        self.value = Pose3D(
            self.value.x + (c * command.vx - s * command.vy) * dt,
            self.value.y + (s * command.vx + c * command.vy) * dt,
            yaw=self.value.yaw + command.yaw_rate * dt,
        )

    def stop(self):
        pass


class NavigationTests(unittest.TestCase):
    def test_astar_routes_through_gap(self):
        occupied = frozenset((5, y) for y in range(10) if y != 5)
        grid = GridMap(10, 10, 1.0, 0.0, 0.0, occupied)
        path = AStarPlanner(grid, footprint_radius_m=0.0).plan(
            Pose3D(1.5, 1.5), Pose3D(8.5, 8.5)
        )
        self.assertTrue(any(abs(p.x - 5.5) < 0.1 and abs(p.y - 5.5) < 0.1 for p in path))

    def test_closed_loop_reaches_goal(self):
        grid = GridMap(20, 20, 0.25, -1.0, -1.0, frozenset())
        base = FakeBase()
        nav = PlannedNavigationBackend(
            AStarPlanner(grid, footprint_radius_m=0.0),
            PathFollower(),
            base,
            control_hz=20,
            timeout_s=2,
        )
        result = nav.navigate(Pose3D(1.0, 0.5, yaw=0.3))
        self.assertEqual(result["state"], "succeeded")
        self.assertLess(math.hypot(base.value.x - 1.0, base.value.y - 0.5), 0.15)

    def test_sonic_planner_wire_protocol(self):
        planner = pack_sonic_planner(
            mode=1,
            movement=(1.0, 0.0, 0.0),
            facing=(1.0, 0.0, 0.0),
            speed=0.3,
        )
        self.assertTrue(planner.startswith(b"planner"))
        self.assertGreater(len(planner), 1280)


if __name__ == "__main__":
    unittest.main()
