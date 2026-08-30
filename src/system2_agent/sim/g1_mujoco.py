from __future__ import annotations

import base64
import io
import math
import sys
from pathlib import Path
from typing import Any

from ..modules.camera import CameraFrame
from ..modules.semantic_map import Pose3D
from ..navigation_core import VelocityCommand


class G1MuJoCoBase:
    """Adapter over the sibling robot_class MuJoCo G1 backend.

    This backend is deliberately named ``mujoco-kinematic-velocity``: robot_class
    integrates the free base for fast navigation-stack tests. It is not SONIC.
    Use ``SonicZmqBase`` for the actual SONIC target-velocity planner boundary.
    """

    name = "mujoco-kinematic-velocity"

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        robot_class_path: str | Path | None = None,
        steps_per_action: int = 0,
    ) -> None:
        if robot_class_path is not None:
            path = str(Path(robot_class_path).resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
        from robot import UnitreeG1Sim, UnitreeG1SimConfig

        self.robot = UnitreeG1Sim(
            UnitreeG1SimConfig(
                model_path=model_path,
                action_mode="base_velocity",
                apply_mode="pd",
                steps_per_action=steps_per_action,
            )
        )
        self.robot.connect()

    def close(self) -> None:
        self.stop()
        self.robot.disconnect()

    def pose(self) -> Pose3D:
        adr = self.robot._root_qposadr
        data = self.robot._data
        if adr is None or data is None:
            raise RuntimeError("G1 model does not have a free root joint")
        w, x, y, z = [float(v) for v in data.qpos[adr + 3 : adr + 7]]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return Pose3D(float(data.qpos[adr]), float(data.qpos[adr + 1]), float(data.qpos[adr + 2]), yaw)

    def set_initial_pose(self, pose: Pose3D) -> None:
        """Set live and reset-state base pose before a simulation mission."""
        adr = self.robot._root_qposadr
        model, data = self.robot._model, self.robot._data
        if adr is None or model is None or data is None:
            raise RuntimeError("G1 model does not have a connected free root joint")
        quaternion = (
            math.cos(pose.yaw / 2.0),
            0.0,
            0.0,
            math.sin(pose.yaw / 2.0),
        )
        model.qpos0[adr : adr + 2] = (pose.x, pose.y)
        model.qpos0[adr + 3 : adr + 7] = quaternion
        data.qpos[adr : adr + 2] = (pose.x, pose.y)
        data.qpos[adr + 3 : adr + 7] = quaternion
        data.qvel[:6] = 0.0
        self.robot._mujoco.mj_forward(model, data)

    def command_velocity(self, command: VelocityCommand, dt: float) -> None:
        self.robot.send_action(
            {"base.vx": command.vx, "base.vy": command.vy, "base.vyaw": command.yaw_rate}
        )
        steps = max(1, round(dt / float(self.robot.config.timestep)))
        # This is intentionally the fast kinematic integration path. Advancing
        # unconstrained MuJoCo dynamics here would make the floating-base model
        # fall while no locomotion policy is controlling its legs.
        for _ in range(steps):
            self.robot._integrate_base_velocity()
        self.robot._mujoco.mj_forward(self.robot._model, self.robot._data)

    def stop(self) -> None:
        if self.robot.is_connected:
            self.robot.send_action({"base.vx": 0.0, "base.vy": 0.0, "base.vyaw": 0.0})


class MuJoCoCamera:
    """Fresh RGB from a named MuJoCo camera, encoded for a VLM request."""

    def __init__(self, base: G1MuJoCoBase, *, camera: str | int | None = None, width: int = 640, height: int = 480) -> None:
        self.base = base
        self.camera = camera
        self.width = width
        self.height = height

    def capture(self) -> list[CameraFrame]:
        image = self.base.robot.render(width=self.width, height=self.height, camera=self.camera)
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("MuJoCoCamera needs Pillow: pip install -e '.[sim]'") from exc
        output = io.BytesIO()
        Image.fromarray(image).save(output, format="JPEG", quality=85)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return [CameraFrame("g1_head_rgb", f"data:image/jpeg;base64,{encoded}")]
