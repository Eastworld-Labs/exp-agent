import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

from system2_agent.modules import Pose3D
from system2_agent.navigation_core import VelocityCommand
from system2_agent.scene_bundle import SceneBundle
from system2_agent.sim.environment import (
    SimulationEnvironment,
    create_simulation_environment,
)
from system2_agent.sim.isaac import (
    DepthFrame,
    IsaacCameraBackend,
    IsaacDepthObserver,
    IsaacHeadCamera,
    IsaacSimBase,
    rotation_from_quaternion,
)
from system2_agent.sim.isaac_scene import select_chase_camera_eye
from system2_agent.navigation_core import GridMap


# A small synthetic head camera: 20x12 pixels, 10 px focal length (about 90
# degrees horizontal by 62 degrees vertical), 0.92 m above the G1 root, pitched
# 40 degrees towards the floor like a real downward-looking head sensor.
HEAD_CAMERA_FX = 10.0
HEAD_CAMERA_FY = 10.0
HEAD_CAMERA_WIDTH = 20
HEAD_CAMERA_HEIGHT = 12
HEAD_CAMERA_PITCH_DOWN = math.radians(40.0)


def optical_rotation(yaw: float, pitch_down: float) -> tuple[tuple[float, float, float], ...]:
    """Rows of the optical frame (+X right, +Y down, +Z forward) as world columns."""
    c, s = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch_down), math.sin(pitch_down)
    right = (s, -c, 0.0)
    down = (-c * sp, -s * sp, -cp)
    forward = (c * cp, s * cp, -sp)
    return tuple(
        (right[axis], down[axis], forward[axis]) for axis in range(3)
    )


def head_camera_pose(robot: Pose3D) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]:
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    position = (robot.x + 0.04 * c, robot.y + 0.04 * s, robot.z + 0.92)
    return optical_rotation(robot.yaw, HEAD_CAMERA_PITCH_DOWN), position


def _box_range(origin, direction, low, high):
    near, far = 0.0, math.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if not low[axis] <= origin[axis] <= high[axis]:
                return None
            continue
        first = (low[axis] - origin[axis]) / direction[axis]
        second = (high[axis] - origin[axis]) / direction[axis]
        if first > second:
            first, second = second, first
        near, far = max(near, first), min(far, second)
        if near > far:
            return None
    return near if near > 0 else None


def render_depth(rotation, position, *, boxes=(), floor_z=0.0):
    """Analytic z-depth of a floor plane and axis-aligned boxes; inf where nothing is hit."""
    import numpy as np

    rows, cols = HEAD_CAMERA_HEIGHT, HEAD_CAMERA_WIDTH
    cx, cy = cols / 2, rows / 2
    rotation = np.asarray(rotation, dtype=float)
    depth = np.full((rows, cols), np.inf)
    for v in range(rows):
        for u in range(cols):
            # Unit optical z, so the ray parameter is the distance to the image plane.
            direction = rotation @ np.array(
                [(u + 0.5 - cx) / HEAD_CAMERA_FX, (v + 0.5 - cy) / HEAD_CAMERA_FY, 1.0]
            )
            best = math.inf
            if direction[2] < 0:
                to_floor = (floor_z - position[2]) / direction[2]
                if to_floor > 0:
                    best = min(best, to_floor)
            for low, high in boxes:
                hit = _box_range(position, direction, low, high)
                if hit is not None:
                    best = min(best, hit)
            depth[v, u] = best
    return depth


def head_depth_frame(robot: Pose3D, *, boxes=(), floor_z=0.0) -> DepthFrame:
    rotation, position = head_camera_pose(robot)
    return DepthFrame(
        depth=render_depth(rotation, position, boxes=boxes, floor_z=floor_z),
        fx=HEAD_CAMERA_FX,
        fy=HEAD_CAMERA_FY,
        cx=HEAD_CAMERA_WIDTH / 2,
        cy=HEAD_CAMERA_HEIGHT / 2,
        rotation=rotation,
        position=position,
    )


class FakeIsaacRuntime:
    def __init__(self) -> None:
        self.current = Pose3D(1.0, 2.0, 0.8, math.pi / 2)
        self.steps: list[float] = []
        self.closed = False
        self.frame: DepthFrame | None = None
        self.physics_enabled = None
        self.rgb_frames = ()

    def pose(self):
        return self.current

    def set_pose(self, pose):
        self.current = pose

    def step(self, seconds):
        self.steps.append(seconds)

    def set_physics_enabled(self, enabled):
        self.physics_enabled = enabled

    def rgb(self):
        return self.rgb_frames

    def depth_frame(self):
        if self.frame is None:
            raise RuntimeError("fake runtime has no depth frame")
        return self.frame

    def close(self):
        self.closed = True


class FakeActuator:
    def __init__(self) -> None:
        self.commands = []
        self.stopped = False

    def command_velocity(self, command, dt):
        self.commands.append((command, dt))

    def stop(self):
        self.stopped = True


class IsaacBackendTests(unittest.TestCase):
    def test_chase_camera_stays_close_with_clear_sight_line(self):
        grid = GridMap(40, 40, 0.25, -5.0, -5.0, frozenset())
        pose = Pose3D(0.0, 0.0, 0.8, 0.0)

        eye = select_chase_camera_eye(grid, pose)

        self.assertLessEqual(math.hypot(eye[0] - pose.x, eye[1] - pose.y), 2.61)
        self.assertTrue(grid.segment_is_free((pose.x, pose.y), (eye[0], eye[1])))

    def test_chase_camera_avoids_wall_behind_robot(self):
        # A wall blocks the straight rear boom, but a rear-quarter view exists.
        occupied = frozenset((14, y) for y in range(16, 25))
        grid = GridMap(40, 40, 0.25, -5.0, -5.0, occupied)
        pose = Pose3D(0.0, 0.0, 0.8, 0.0)

        eye = select_chase_camera_eye(grid, pose)

        self.assertTrue(grid.segment_is_free((pose.x, pose.y), (eye[0], eye[1])))
        self.assertFalse(math.isclose(float(eye[1]), 0.0, abs_tol=0.05))

    def test_kinematic_adapter_integrates_body_velocity(self):
        runtime = FakeIsaacRuntime()
        base = IsaacSimBase(runtime)

        base.command_velocity(VelocityCommand(0.4, 0.0, 0.2), 0.5)

        self.assertAlmostEqual(base.pose().x, 1.0)
        self.assertAlmostEqual(base.pose().y, 2.2)
        self.assertAlmostEqual(base.pose().yaw, math.pi / 2 + 0.1)
        self.assertEqual(runtime.steps, [0.5])
        self.assertFalse(runtime.physics_enabled)
        self.assertEqual(base.name, "isaac-sim-kinematic-velocity")

    def test_step_hooks_run_after_every_step_on_both_paths(self):
        runtime = FakeIsaacRuntime()
        ticks = []
        base = IsaacSimBase(runtime)
        base.step_hooks.append(lambda: ticks.append(len(runtime.steps)))
        base.command_velocity(VelocityCommand(0.1, 0.0, 0.0), 0.05)

        actuated = IsaacSimBase(FakeIsaacRuntime(), FakeActuator())
        actuated.step_hooks.append(lambda: ticks.append("actuated"))
        actuated.command_velocity(VelocityCommand(0.1, 0.0, 0.0), 0.05)

        self.assertEqual(ticks, [1, "actuated"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Isaac depth needs NumPy")
    def test_head_camera_captures_colour_and_sensor_ranged_depth(self):
        import numpy as np

        runtime = FakeIsaacRuntime()
        runtime.rgb_frames = (("g1_head_rgb", np.full((12, 20, 3), 7, dtype=np.uint8)),)
        rotation, position = head_camera_pose(runtime.pose())
        depth = np.full((12, 20), 2.5, dtype=np.float32)
        depth[0, :] = np.inf  # sky
        depth[1, :] = 0.1  # inside the sensor's minimum range
        runtime.frame = DepthFrame(depth, 10.0, 10.0, 10.0, 6.0, rotation, position)

        frame = IsaacHeadCamera(runtime).capture()

        self.assertEqual(frame.rgb.shape, (12, 20, 3))
        self.assertEqual(frame.depth.shape, (12, 20))
        self.assertTrue(np.isnan(frame.depth[0]).all())
        self.assertTrue(np.isnan(frame.depth[1]).all())
        self.assertTrue((frame.depth[2:] == np.float32(2.5)).all())
        self.assertGreater(frame.stamp_s, 0)

    def test_initial_pose_preserves_authored_robot_height(self):
        runtime = FakeIsaacRuntime()
        base = IsaacSimBase(runtime)

        base.set_initial_pose(Pose3D(-2.0, 3.0, yaw=-0.4))

        self.assertEqual(base.pose(), Pose3D(-2.0, 3.0, 0.8, -0.4))

    def test_optional_actuator_owns_velocity_execution(self):
        runtime = FakeIsaacRuntime()
        actuator = FakeActuator()
        base = IsaacSimBase(runtime, actuator)
        command = VelocityCommand(0.2, 0.1, -0.1)

        base.command_velocity(command, 0.05)
        base.close()

        self.assertEqual(actuator.commands, [(command, 0.05)])
        self.assertTrue(actuator.stopped)
        self.assertTrue(runtime.closed)
        self.assertTrue(runtime.physics_enabled)
        self.assertEqual(base.name, "isaac-sim-actuated-velocity")

    def test_environment_cleanup_is_idempotent(self):
        runtime = FakeIsaacRuntime()
        base = IsaacSimBase(runtime)
        environment = SimulationEnvironment("isaac", base, _owned=(base,))

        environment.close()
        environment.close()

        self.assertTrue(runtime.closed)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Isaac depth needs NumPy")
    def test_depth_observer_ignores_floor_seen_by_pitched_camera(self):
        runtime = FakeIsaacRuntime()
        runtime.frame = head_depth_frame(runtime.pose())
        # Control: with floor rejection effectively disabled (the floor is
        # placed a metre above the "ground"), the very same frame reports an
        # obstacle well inside trigger range. That is the bug this guards
        # against: a pitched head camera reporting the floor straight ahead.
        unfiltered = IsaacDepthObserver(
            runtime, ground_z_m=-1.0, min_obstacle_height_m=0.0
        ).observe(runtime.pose())
        self.assertEqual(len(unfiltered.obstacles), 1)
        floor = unfiltered.obstacles[0]
        self.assertLess(math.hypot(floor.x - 1.0, floor.y - 2.0), 1.25)

        observation = IsaacDepthObserver(runtime).observe(runtime.pose())

        self.assertTrue(observation.healthy, observation.detail)
        self.assertEqual(observation.obstacles, ())
        self.assertRegex(observation.detail, r"floor_px=[1-9]")
        self.assertIn("corridor_px=0", observation.detail)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Isaac depth needs NumPy")
    def test_depth_observer_reports_robot_relative_obstacle(self):
        runtime = FakeIsaacRuntime()
        # A 0.6 m wide, 1 m tall box whose front face is 0.55 m ahead of the
        # robot (which faces +y), standing on the same floor the camera sees.
        runtime.frame = head_depth_frame(
            runtime.pose(), boxes=(((0.7, 2.55, 0.0), (1.3, 2.85, 1.0)),)
        )
        observer = IsaacDepthObserver(runtime)

        observation = observer.observe(runtime.pose())

        self.assertTrue(observation.healthy, observation.detail)
        self.assertEqual(len(observation.obstacles), 1)
        obstacle = observation.obstacles[0]
        self.assertAlmostEqual(obstacle.x, 1.0, delta=0.05)
        self.assertAlmostEqual(obstacle.y, 2.55, delta=0.03)
        self.assertEqual(obstacle.label, "head_depth_forward")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Isaac depth needs NumPy")
    def test_depth_observer_ignores_objects_outside_body_corridor(self):
        runtime = FakeIsaacRuntime()
        # Same box, shifted half a metre to the robot's left: visible, but
        # clear of the forward corridor.
        runtime.frame = head_depth_frame(
            runtime.pose(), boxes=(((0.25, 2.55, 0.0), (0.5, 2.85, 1.0)),)
        )

        observation = IsaacDepthObserver(runtime).observe(runtime.pose())

        self.assertTrue(observation.healthy, observation.detail)
        self.assertEqual(observation.obstacles, ())
        self.assertRegex(observation.detail, r"body_px=[1-9]")
        self.assertIn("corridor_px=0", observation.detail)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Isaac depth needs NumPy")
    def test_depth_observer_honours_configured_floor_height(self):
        runtime = FakeIsaacRuntime()
        # A scene whose floor is authored 0.5 m above the world origin, with the
        # robot standing on it. The default ground_z_m would read that floor as
        # a half-metre-high wall.
        runtime.current = Pose3D(1.0, 2.0, 1.3, math.pi / 2)
        runtime.frame = head_depth_frame(runtime.pose(), floor_z=0.5)

        default = IsaacDepthObserver(runtime).observe(runtime.pose())
        configured = IsaacDepthObserver(runtime, ground_z_m=0.5).observe(runtime.pose())

        self.assertEqual(len(default.obstacles), 1)
        self.assertEqual(configured.obstacles, ())

    def test_rotation_from_quaternion_matches_yaw(self):
        half = math.sqrt(0.5)
        rotation = rotation_from_quaternion((half, 0.0, 0.0, half))

        # A quarter turn about +z carries +x onto +y.
        self.assertAlmostEqual(rotation[0][0], 0.0)
        self.assertAlmostEqual(rotation[1][0], 1.0)
        self.assertAlmostEqual(rotation[2][2], 1.0)
        with self.assertRaises(ValueError):
            rotation_from_quaternion((0.0, 0.0, 0.0, 0.0))

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"),
        "Isaac camera encoding needs NumPy and Pillow",
    )
    def test_camera_backend_encodes_labeled_rgb(self):
        import numpy as np

        runtime = FakeIsaacRuntime()
        runtime.rgb_frames = (
            ("g1_head_rgb", np.zeros((4, 6, 3), dtype=np.uint8)),
        )

        frames = IsaacCameraBackend(runtime).capture()

        self.assertEqual(frames[0].label, "g1_head_rgb")
        self.assertTrue(frames[0].url.startswith("data:image/jpeg;base64,"))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Isaac depth needs NumPy")
    def test_depth_observer_fails_closed_on_invalid_frame(self):
        runtime = FakeIsaacRuntime()
        rotation, position = head_camera_pose(runtime.pose())
        runtime.frame = DepthFrame([], 10.0, 10.0, 10.0, 6.0, rotation, position)

        observation = IsaacDepthObserver(runtime).observe(runtime.pose())

        self.assertFalse(observation.healthy)
        self.assertIn("non-empty", observation.detail)

    def test_depth_observer_fails_closed_without_camera_model(self):
        runtime = FakeIsaacRuntime()

        observation = IsaacDepthObserver(runtime).observe(runtime.pose())

        self.assertFalse(observation.healthy)
        self.assertIn("no depth frame", observation.detail)

    def test_factory_rejects_isaac_without_backend_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            navigation = root / "grid.json"
            semantics = root / "semantics.json"
            navigation.write_text("{}", encoding="utf-8")
            semantics.write_text("{}", encoding="utf-8")
            bundle = SceneBundle(None, navigation, semantics)

            with self.assertRaisesRegex(ValueError, "no isaac_sim configuration"):
                create_simulation_environment("isaac", bundle, workspace=root)


if __name__ == "__main__":
    unittest.main()
