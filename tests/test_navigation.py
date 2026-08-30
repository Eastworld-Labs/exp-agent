import math
import json
import struct
import unittest

from system2_agent.modules import Pose3D
from system2_agent.navigation_core import (
    AStarPlanner,
    GridMap,
    LocalNavigationObservation,
    LocalObstacle,
    PathFollower,
    PlannedNavigationBackend,
    RegulatedTrajectoryFollower,
    SmoothTrajectoryPlanner,
    VelocityCommand,
)
from system2_agent.sonic_bridge import pack_sonic_planner, slew_heading
from system2_agent.sim.g1_mujoco import G1MuJoCoBase


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
    def test_local_sensor_obstacle_brakes_then_allows_navigation(self):
        class RecordingBase(FakeBase):
            def __init__(self):
                super().__init__()
                self.commands = []

            def command_velocity(self, command, dt):
                self.commands.append(command)
                super().command_velocity(command, dt)

        class ClearingObserver:
            def __init__(self):
                self.calls = 0

            def observe(self, pose):
                self.calls += 1
                obstacles = (
                    (LocalObstacle(0.2, 0.0, 0.1, "person"),)
                    if self.calls <= 4
                    else ()
                )
                return LocalNavigationObservation("test_depth", obstacles)

        grid = GridMap(20, 20, 0.1, -0.5, -0.5, frozenset())
        planner = SmoothTrajectoryPlanner(grid, footprint_radius_m=0.0)
        base = RecordingBase()
        observer = ClearingObserver()
        nav = PlannedNavigationBackend(
            planner,
            RegulatedTrajectoryFollower(planner.grid, max_speed=0.3),
            base,
            local_observer=observer,
            timeout_s=2,
        )

        result = nav.navigate(Pose3D(1.0, 0.0))

        self.assertEqual(result["state"], "succeeded")
        self.assertGreaterEqual(observer.calls, 5)
        self.assertTrue(any(command.vx == 0.0 for command in base.commands[:4]))
        self.assertEqual(result["local_observation"]["source"], "test_depth")

    def test_unhealthy_local_sensor_fails_closed(self):
        class BrokenObserver:
            def observe(self, pose):
                return LocalNavigationObservation(
                    "depth", healthy=False, detail="stale depth stream"
                )

        grid = GridMap(20, 20, 0.1, -0.5, -0.5, frozenset())
        planner = SmoothTrajectoryPlanner(grid, footprint_radius_m=0.0)
        nav = PlannedNavigationBackend(
            planner,
            RegulatedTrajectoryFollower(planner.grid),
            FakeBase(),
            local_observer=BrokenObserver(),
            timeout_s=2,
        )

        result = nav.navigate(Pose3D(1.0, 0.0))

        self.assertEqual(result["state"], "failed")
        self.assertIn("stale depth stream", result["error"])

    def test_astar_routes_through_gap(self):
        occupied = frozenset((5, y) for y in range(10) if y != 5)
        grid = GridMap(10, 10, 1.0, 0.0, 0.0, occupied)
        path = AStarPlanner(grid, footprint_radius_m=0.0).plan(
            Pose3D(1.5, 1.5), Pose3D(8.5, 8.5)
        )
        # The simplified segment crosses the wall at its sole free cell (5, 5).
        self.assertEqual([(p.x, p.y) for p in path], [(1.5, 1.5), (8.5, 8.5)])

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

    def test_executes_preplanned_multi_stop_path(self):
        grid = GridMap(20, 20, 0.25, -1.0, -1.0, frozenset())
        base = FakeBase()
        nav = PlannedNavigationBackend(
            AStarPlanner(grid, footprint_radius_m=0.0),
            PathFollower(align_final_yaw=False),
            base,
            control_hz=20,
            timeout_s=2,
        )
        route = [Pose3D(0.0, 0.0), Pose3D(0.5, 0.5), Pose3D(1.0, 0.0)]
        result = nav.navigate_path(route)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["waypoint_count"], 3)
        self.assertLess(math.hypot(base.value.x - 1.0, base.value.y), 0.15)

    def test_nonholonomic_follower_turns_before_walking(self):
        follower = PathFollower(holonomic=False, turn_in_place_threshold=0.2)
        turning, reached = follower.command(
            Pose3D(0.0, 0.0, yaw=0.0), Pose3D(0.0, 1.0), final=True
        )
        self.assertFalse(reached)
        self.assertEqual((turning.vx, turning.vy), (0.0, 0.0))
        self.assertGreater(turning.yaw_rate, 0.0)

    def test_sonic_style_follower_walks_through_heading_error(self):
        follower = PathFollower(
            holonomic=False, turn_in_place=False, align_final_yaw=False, max_speed=0.30
        )
        command, reached = follower.command(
            Pose3D(0.0, 0.0, yaw=0.0), Pose3D(0.0, 1.0, yaw=math.pi), final=True
        )
        self.assertFalse(reached)
        self.assertEqual(command.vx, 0.30)
        self.assertAlmostEqual(command.facing_yaw, math.pi / 2)

    def test_sonic_facing_reference_is_rate_limited(self):
        heading = slew_heading(0.0, math.pi, max_rate=0.3, dt=0.05)
        self.assertAlmostEqual(heading, -0.015)
        self.assertLess(abs(heading), 0.02)

    def test_nonholonomic_follower_rotates_before_large_heading_change(self):
        follower = PathFollower(
            holonomic=False,
            turn_in_place=True,
            turn_in_place_threshold=0.3,
            max_yaw_rate=0.3,
        )
        command, reached = follower.command(
            Pose3D(0.0, 0.0, yaw=0.0), Pose3D(-1.0, 0.0), final=True
        )
        self.assertFalse(reached)
        self.assertEqual((command.vx, command.vy), (0.0, 0.0))
        self.assertAlmostEqual(abs(command.yaw_rate), 0.3)

    def test_astar_simplifies_open_path(self):
        grid = GridMap(20, 20, 0.25, -1.0, -1.0, frozenset())
        path = AStarPlanner(grid, footprint_radius_m=0.0).plan(
            Pose3D(0.0, 0.0), Pose3D(1.0, 0.5)
        )
        self.assertEqual(len(path), 2)

    def test_smooth_trajectory_stays_collision_free(self):
        occupied = frozenset((5, y) for y in range(10) if y != 5)
        planner = SmoothTrajectoryPlanner(
            GridMap(10, 10, 1.0, 0.0, 0.0, occupied),
            footprint_radius_m=0.0,
            sample_spacing_m=0.4,
        )
        path = planner.plan(Pose3D(1.5, 1.5), Pose3D(8.5, 8.5))
        self.assertGreater(len(path), 2)
        self.assertTrue(
            all(
                planner.grid.segment_is_free((a.x, a.y), (b.x, b.y))
                for a, b in zip(path, path[1:])
            )
        )
        self.assertEqual(
            planner.last_plan_metrics["trajectory_optimizer"],
            "collision_checked_elastic_band",
        )

    def test_regulated_trajectory_follower_reaches_goal(self):
        grid = GridMap(20, 20, 0.25, -1.0, -1.0, frozenset())
        planner = SmoothTrajectoryPlanner(grid, footprint_radius_m=0.0)
        base = FakeBase()
        nav = PlannedNavigationBackend(
            planner,
            RegulatedTrajectoryFollower(planner.grid, max_speed=0.3),
            base,
            control_hz=20,
            timeout_s=2,
        )
        result = nav.navigate(Pose3D(1.0, 0.5))
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("planning", result)
        self.assertLess(math.hypot(base.value.x - 1.0, base.value.y - 0.5), 0.15)

    def test_lookahead_does_not_fake_measured_path_progress(self):
        grid = GridMap(20, 20, 0.25, -1.0, -1.0, frozenset())
        follower = RegulatedTrajectoryFollower(grid, lookahead_m=0.35)
        path = [Pose3D(index * 0.1, 0.0) for index in range(10)]
        waypoint = 1
        for _ in range(20):
            _, _, waypoint = follower.command_path(Pose3D(0.0, 0.0), path, waypoint)
        self.assertEqual(waypoint, 1)

    def test_sonic_planner_wire_protocol(self):
        planner = pack_sonic_planner(
            mode=1,
            movement=(1.0, 0.0, 0.0),
            facing=(1.0, 0.0, 0.0),
            speed=0.3,
        )
        self.assertTrue(planner.startswith(b"planner"))
        self.assertGreater(len(planner), 1280)

    def test_sonic_planner_carries_bounded_upper_body_targets(self):
        planner = pack_sonic_planner(
            mode=0,
            movement=(0.0, 0.0, 0.0),
            facing=(1.0, 0.0, 0.0),
            speed=0.0,
            upper_body_position=tuple(float(index) for index in range(17)),
            upper_body_velocity=(0.0,) * 17,
        )
        header = json.loads(planner[7:1287].rstrip(b"\0"))
        self.assertEqual(header["fields"][-2]["name"], "upper_body_position")
        values = struct.unpack("<17f", planner[-136:-68])
        self.assertEqual(values, tuple(float(index) for index in range(17)))

    def test_sonic_planner_rejects_wrong_upper_body_shape(self):
        with self.assertRaises(ValueError):
            pack_sonic_planner(
                mode=0,
                movement=(0.0, 0.0, 0.0),
                facing=(1.0, 0.0, 0.0),
                speed=0.0,
                upper_body_position=(0.0,) * 16,
            )

    def test_fast_g1_adapter_uses_kinematic_integration(self):
        class FakeMuJoCo:
            forwarded = 0

            def mj_forward(self, model, data):
                self.forwarded += 1

        class FakeRobot:
            config = type("Config", (), {"timestep": 0.01})()
            _model = object()
            _data = object()
            _mujoco = FakeMuJoCo()
            integrated = 0

            def send_action(self, action):
                self.action = action

            def _integrate_base_velocity(self):
                self.integrated += 1

            def step(self, _):
                raise AssertionError("fast kinematic adapter must not advance dynamics")

        base = object.__new__(G1MuJoCoBase)
        base.robot = FakeRobot()
        base.command_velocity(VelocityCommand(0.2, 0.1, 0.3), 0.05)

        self.assertEqual(base.robot.integrated, 5)
        self.assertEqual(base.robot._mujoco.forwarded, 1)
        self.assertEqual(base.robot.action["base.vx"], 0.2)

    def test_fast_g1_adapter_persists_scene_initial_pose(self):
        class FakeMuJoCo:
            def mj_forward(self, model, data):
                self.forwarded = (model, data)

        class Array(list):
            def __setitem__(self, key, value):
                if isinstance(key, slice):
                    try:
                        replacement = list(value)
                    except TypeError:
                        start, stop, step = key.indices(len(self))
                        replacement = [value] * len(range(start, stop, step))
                    super().__setitem__(key, replacement)
                else:
                    super().__setitem__(key, value)

        class FakeRobot:
            _root_qposadr = 0
            _model = type("Model", (), {"qpos0": Array([0.0] * 10)})()
            _data = type("Data", (), {"qpos": Array([0.0] * 10), "qvel": Array([1.0] * 8)})()
            _mujoco = FakeMuJoCo()

        base = object.__new__(G1MuJoCoBase)
        base.robot = FakeRobot()
        base.set_initial_pose(Pose3D(2.0, -3.0, yaw=math.pi / 2))

        self.assertEqual(base.robot._model.qpos0[:2], [2.0, -3.0])
        self.assertEqual(base.robot._data.qpos[:2], [2.0, -3.0])
        self.assertEqual(base.robot._data.qvel[:6], [0.0] * 6)


if __name__ == "__main__":
    unittest.main()
