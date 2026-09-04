from __future__ import annotations

import base64
import io
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ..modules.camera import CameraFrame
from ..modules.semantic_map import Pose3D
from ..navigation_core import VelocityCommand
from .head_camera import D455, HeadCameraFrame, HeadCameraSpec, mask_depth


class G1MuJoCoBase:
    """Adapter over the sibling robot_class MuJoCo G1 backend.

    This backend is deliberately named ``mujoco-kinematic-velocity``: robot_class
    integrates the free base for fast navigation-stack tests. It is not SONIC.
    Use ``SonicZmqBase`` for the actual SONIC target-velocity planner boundary.
    """

    name = "mujoco-kinematic-velocity"

    # Class-level defaults so the adapter methods stay usable on an instance built
    # without __init__ -- the navigation tests drive command_velocity() and
    # set_initial_pose() against a fake robot via object.__new__.
    _viewer = None
    #: Serialises mjData between the stepping loop and anything that renders
    #: from another thread (the head camera stream). Re-entrant: close() stops.
    lock: threading.RLock = threading.RLock()

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        robot_class_path: str | Path | None = None,
        steps_per_action: int = 0,
        viewer: bool = False,
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
        self.lock = threading.RLock()
        self._viewer = self._open_viewer() if viewer else None

    def _open_viewer(self):
        """Attach MuJoCo's passive viewer, or None if no display can host it.

        Passive rather than managed: this process already owns the stepping
        loop. A missing display is not a mission failure -- the navigation run
        is identical without a window -- so a failure here degrades to
        headless instead of aborting. macOS is the exception worth naming:
        MuJoCo's viewer needs the `mjpython` launcher there, so a plain
        `python` run reports the reason and continues without a window.
        """
        try:
            from mujoco import viewer as mujoco_viewer

            return mujoco_viewer.launch_passive(self.robot._model, self.robot._data)
        except Exception as exc:
            print(f"MuJoCo viewer unavailable, continuing headless: {exc}", file=sys.stderr)
            return None

    def _sync_viewer(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        self.stop()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
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
        with self.lock:
            model.qpos0[adr : adr + 2] = (pose.x, pose.y)
            model.qpos0[adr + 3 : adr + 7] = quaternion
            data.qpos[adr : adr + 2] = (pose.x, pose.y)
            data.qpos[adr + 3 : adr + 7] = quaternion
            data.qvel[:6] = 0.0
            self.robot._mujoco.mj_forward(model, data)
        self._sync_viewer()

    def command_velocity(self, command: VelocityCommand, dt: float) -> None:
        steps = max(1, round(dt / float(self.robot.config.timestep)))
        with self.lock:
            self.robot.send_action(
                {"base.vx": command.vx, "base.vy": command.vy, "base.vyaw": command.yaw_rate}
            )
            # This is intentionally the fast kinematic integration path. Advancing
            # unconstrained MuJoCo dynamics here would make the floating-base model
            # fall while no locomotion policy is controlling its legs.
            for _ in range(steps):
                self.robot._integrate_base_velocity()
            self.robot._mujoco.mj_forward(self.robot._model, self.robot._data)
        self._sync_viewer()

    def stop(self) -> None:
        if self.robot.is_connected:
            with self.lock:
                self.robot.send_action({"base.vx": 0.0, "base.vy": 0.0, "base.vyaw": 0.0})

    def head_camera(self, spec: HeadCameraSpec = D455) -> "MuJoCoHeadCamera":
        """The head depth camera SceneLoader attached to this model."""
        return MuJoCoHeadCamera(self.robot._model, self.robot._data, spec=spec, lock=self.lock)


class MuJoCoHeadCamera:
    """Colour and metric depth from the head camera of a MuJoCo model.

    Rendering reads mjData while a stepping loop mutates it, so each capture
    holds ``lock`` when one is given. GL contexts are thread-affine: the
    renderer is created lazily on the first capturing thread and only used
    there, so a second thread wanting frames takes its own instance.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        spec: HeadCameraSpec = D455,
        lock: Any | None = None,
        clock: Any = time.time,
    ) -> None:
        self.model = model
        self.data = data
        self.spec = spec
        self.lock = lock if lock is not None else threading.RLock()
        self._clock = clock
        self._renderer: Any = None
        self._thread: int | None = None
        # The offscreen buffer must hold the camera's full resolution; the
        # scene loader sizes it, but a model loaded some other way may not have.
        offscreen = self.model.vis.global_
        offscreen.offwidth = max(int(offscreen.offwidth), spec.width)
        offscreen.offheight = max(int(offscreen.offheight), spec.height)

    def _renderer_for_this_thread(self) -> Any:
        import mujoco

        current = threading.get_ident()
        if self._renderer is not None and self._thread != current:
            raise RuntimeError(
                "MuJoCoHeadCamera renders on one thread; create another instance "
                "for a second thread"
            )
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.spec.height, width=self.spec.width)
            self._thread = current
        return self._renderer

    def capture(self) -> HeadCameraFrame:
        import numpy as np

        renderer = self._renderer_for_this_thread()
        with self.lock:
            renderer.update_scene(self.data, camera=self.spec.name)
            renderer.disable_depth_rendering()
            rgb = np.array(renderer.render(), dtype=np.uint8, copy=True)
            renderer.enable_depth_rendering()
            try:
                depth = np.array(renderer.render(), dtype=np.float32, copy=True)
            finally:
                renderer.disable_depth_rendering()
        # MuJoCo reports a depth for every pixel out to its far plane; the
        # sensor does not. Clip to the D455's working range so sky and far
        # walls read as "no return", as they would on the robot.
        return HeadCameraFrame(rgb, mask_depth(depth, self.spec), float(self._clock()))

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            finally:
                self._renderer = None


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
