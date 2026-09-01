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
from system2_agent.sim.isaac import IsaacCameraBackend, IsaacDepthObserver, IsaacSimBase
from system2_agent.sim.isaac_scene import select_chase_camera_eye
from system2_agent.navigation_core import GridMap


class FakeIsaacRuntime:
    def __init__(self) -> None:
        self.current = Pose3D(1.0, 2.0, 0.8, math.pi / 2)
        self.steps: list[float] = []
        self.closed = False
        self.depth_image = [[5.0] * 20 for _ in range(12)]
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

    def depth(self):
        return self.depth_image

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
    def test_depth_observer_reports_robot_relative_obstacle(self):
        runtime = FakeIsaacRuntime()
        for row in runtime.depth_image:
            row[8:12] = [0.55] * 4
        observer = IsaacDepthObserver(runtime)

        observation = observer.observe(runtime.pose())

        self.assertTrue(observation.healthy)
        self.assertEqual(len(observation.obstacles), 1)
        obstacle = observation.obstacles[0]
        self.assertAlmostEqual(obstacle.x, 1.0, places=5)
        self.assertAlmostEqual(obstacle.y, 2.55, places=5)

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
        runtime.depth_image = []

        observation = IsaacDepthObserver(runtime).observe(runtime.pose())

        self.assertFalse(observation.healthy)
        self.assertIn("non-empty", observation.detail)

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
