from __future__ import annotations

import base64
import io
import math
from collections.abc import Sequence
from typing import Any, Protocol

from ..modules.camera import CameraFrame
from ..modules.semantic_map import Pose3D
from ..navigation_core import (
    LocalNavigationObservation,
    LocalObstacle,
    VelocityCommand,
)
from ..scene_bundle import IsaacCameraSpec, IsaacSimScene


class IsaacRuntime(Protocol):
    """Small testable boundary around the Kit/PhysX runtime."""

    def pose(self) -> Pose3D: ...

    def set_pose(self, pose: Pose3D) -> None: ...

    def step(self, seconds: float) -> None: ...

    def set_physics_enabled(self, enabled: bool) -> None: ...

    def rgb(self) -> Sequence[tuple[str, Any]]: ...

    def depth(self) -> Any: ...

    def close(self) -> None: ...


class IsaacVelocityActuator(Protocol):
    """Optional physical/learned locomotion controller for an Isaac G1."""

    def command_velocity(self, command: VelocityCommand, dt: float) -> None: ...

    def stop(self) -> None: ...


class IsaacSimBase:
    """Mobile-base adapter for an Isaac stage.

    Without an actuator this uses root-kinematic integration for System-2,
    semantic-map and camera evaluation. Supply a SONIC/perceptive actuator for
    policy-faithful locomotion; the mission agent and navigation API do not
    change.
    """

    def __init__(
        self,
        runtime: IsaacRuntime,
        actuator: IsaacVelocityActuator | None = None,
    ) -> None:
        self.runtime = runtime
        self.actuator = actuator
        self.runtime.set_physics_enabled(actuator is not None)

    @property
    def name(self) -> str:
        return (
            "isaac-sim-kinematic-velocity"
            if self.actuator is None
            else "isaac-sim-actuated-velocity"
        )

    def pose(self) -> Pose3D:
        return self.runtime.pose()

    def set_initial_pose(self, pose: Pose3D) -> None:
        current = self.runtime.pose()
        # Scene manifests define planar spawn only. Preserve the authored G1
        # root height instead of placing its pelvis at z=0.
        self.runtime.set_pose(
            Pose3D(pose.x, pose.y, current.z, pose.yaw, pose.frame)
        )

    def command_velocity(self, command: VelocityCommand, dt: float) -> None:
        if self.actuator is not None:
            self.actuator.command_velocity(command, dt)
            self.runtime.step(dt)
            return
        pose = self.pose()
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        world_vx = c * command.vx - s * command.vy
        world_vy = s * command.vx + c * command.vy
        self.runtime.set_pose(
            Pose3D(
                pose.x + world_vx * dt,
                pose.y + world_vy * dt,
                pose.z,
                pose.yaw + command.yaw_rate * dt,
                pose.frame,
            )
        )
        self.runtime.step(dt)

    def stop(self) -> None:
        if self.actuator is not None:
            self.actuator.stop()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.runtime.close()


class IsaacCameraBackend:
    """Encode fresh Isaac RTX camera frames for System-2/VLM observations."""

    def __init__(self, runtime: IsaacRuntime, *, jpeg_quality: int = 85) -> None:
        self.runtime = runtime
        self.jpeg_quality = jpeg_quality

    def capture(self) -> list[CameraFrame]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Isaac camera encoding needs Pillow") from exc
        frames: list[CameraFrame] = []
        for label, array in self.runtime.rgb():
            output = io.BytesIO()
            Image.fromarray(array).convert("RGB").save(
                output, format="JPEG", quality=self.jpeg_quality
            )
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            frames.append(
                CameraFrame(str(label), f"data:image/jpeg;base64,{encoded}")
            )
        return frames


class IsaacDepthObserver:
    """Conservative robot-relative obstacle observation from the head depth image.

    This is a safety/local-navigation input, independent of the slower VLM RGB
    loop. A learned perceptive controller or an nvblox/ESDF adapter can replace
    it behind the same LocalNavigationObserver contract.
    """

    def __init__(
        self,
        runtime: IsaacRuntime,
        *,
        maximum_obstacle_distance_m: float = 1.25,
        obstacle_radius_m: float = 0.22,
        corridor_fraction: float = 0.35,
    ) -> None:
        if maximum_obstacle_distance_m <= 0 or obstacle_radius_m < 0:
            raise ValueError("depth observer distances must be non-negative")
        if not 0 < corridor_fraction <= 1:
            raise ValueError("corridor_fraction must be in (0, 1]")
        self.runtime = runtime
        self.maximum_obstacle_distance_m = maximum_obstacle_distance_m
        self.obstacle_radius_m = obstacle_radius_m
        self.corridor_fraction = corridor_fraction

    def observe(self, pose: Pose3D) -> LocalNavigationObservation:
        try:
            import numpy as np

            depth = np.asarray(self.runtime.depth(), dtype=np.float32)
            if depth.ndim != 2 or not depth.size:
                raise ValueError("depth image must be a non-empty HxW array")
            half = max(1, round(depth.shape[1] * self.corridor_fraction / 2))
            center = depth.shape[1] // 2
            # Ignore sky/ceiling and the bottom floor-heavy region. This is a
            # forward body corridor, not a generic closest-pixel detector.
            row_start = round(depth.shape[0] * 0.15)
            row_stop = max(row_start + 1, round(depth.shape[0] * 0.75))
            corridor = depth[
                row_start:row_stop, max(0, center - half) : center + half
            ]
            valid = corridor[np.isfinite(corridor) & (corridor > 0)]
            if not valid.size:
                raise ValueError("depth image contains no finite positive range")
            # A low percentile rejects isolated invalid/edge pixels while still
            # reacting to a meaningful object in the forward corridor.
            distance = float(np.percentile(valid, 5.0))
        except Exception as exc:
            return LocalNavigationObservation(
                source="isaac_head_depth",
                frame=pose.frame,
                healthy=False,
                detail=str(exc),
            )
        obstacles: tuple[LocalObstacle, ...] = ()
        if distance <= self.maximum_obstacle_distance_m:
            obstacles = (
                LocalObstacle(
                    pose.x + distance * math.cos(pose.yaw),
                    pose.y + distance * math.sin(pose.yaw),
                    self.obstacle_radius_m,
                    "head_depth_forward",
                ),
            )
        return LocalNavigationObservation(
            source="isaac_head_depth",
            obstacles=obstacles,
            frame=pose.frame,
            detail=f"forward_range_m={distance:.3f}",
        )


class OmniverseIsaacRuntime:
    """Lazy Isaac Sim 6 runtime for USD/PhysX/RTX scene evaluation.

    All Omniverse imports intentionally happen after SimulationApp starts. Run
    the CLI with Isaac Sim's Python environment, not a regular virtualenv.
    """

    def __init__(
        self,
        scene: IsaacSimScene,
        *,
        headless: bool = True,
        physics_hz: float = 60.0,
    ) -> None:
        if physics_hz <= 0:
            raise ValueError("physics_hz must be positive")
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise ImportError(
                "Isaac backend requires Isaac Sim; run with its python.sh or Python environment"
            ) from exc
        self._app = SimulationApp(
            {
                "headless": headless,
                "renderer": scene.renderer,
                "width": max((camera.width for camera in scene.cameras), default=640),
                "height": max((camera.height for camera in scene.cameras), default=480),
            }
        )
        try:
            import omni.timeline
            from isaacsim.core.prims import SingleXFormPrim
            from isaacsim.core.utils.prims import is_prim_path_valid
            from isaacsim.core.utils.stage import (
                add_reference_to_stage,
                is_stage_loading,
                open_stage,
            )
            from isaacsim.sensors.camera import Camera

            if not open_stage(str(scene.stage_usd)):
                raise RuntimeError(f"Isaac Sim could not open stage: {scene.stage_usd}")
            while is_stage_loading():
                self._app.update()
            if not is_prim_path_valid(scene.robot_prim) and scene.robot_usd is not None:
                add_reference_to_stage(str(scene.robot_usd), scene.robot_prim)
                while is_stage_loading():
                    self._app.update()
            if not is_prim_path_valid(scene.robot_prim):
                raise ValueError(
                    "Isaac robot prim does not exist in the stage and no usable "
                    f"robot_usd was supplied: {scene.robot_prim}"
                )
            self._root = SingleXFormPrim(
                scene.robot_prim,
                name="exp_agent_robot_root",
                reset_xform_properties=False,
            )
            self._timeline = omni.timeline.get_timeline_interface()
            self._timeline.play()
            self._physics_hz = physics_hz
            self._cameras: list[tuple[IsaacCameraSpec, Any]] = []
            for spec in scene.cameras:
                if not is_prim_path_valid(spec.prim_path):
                    raise ValueError(
                        f"Isaac camera prim does not exist in the stage: {spec.prim_path}"
                    )
                camera = Camera(
                    prim_path=spec.prim_path,
                    resolution=(spec.width, spec.height),
                )
                camera.initialize()
                camera.add_distance_to_image_plane_to_frame()
                self._cameras.append((spec, camera))
            self._app.update()
            self._app.update()
            self._closed = False
        except Exception:
            self._app.close()
            raise

    def pose(self) -> Pose3D:
        position, quaternion = self._root.get_world_pose()
        w, x, y, z = (float(value) for value in quaternion)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return Pose3D(
            float(position[0]), float(position[1]), float(position[2]), yaw
        )

    def set_pose(self, pose: Pose3D) -> None:
        import numpy as np

        self._root.set_world_pose(
            position=np.asarray((pose.x, pose.y, pose.z), dtype=np.float32),
            orientation=np.asarray(
                (math.cos(pose.yaw / 2), 0.0, 0.0, math.sin(pose.yaw / 2)),
                dtype=np.float32,
            ),
        )

    def step(self, seconds: float) -> None:
        steps = max(1, round(max(0.0, seconds) * self._physics_hz))
        for _ in range(steps):
            self._app.update()

    def set_physics_enabled(self, enabled: bool) -> None:
        if enabled:
            self._timeline.play()
        else:
            self._timeline.pause()

    def rgb(self) -> Sequence[tuple[str, Any]]:
        self._app.update()
        frames = []
        for spec, camera in self._cameras:
            image = camera.get_rgba()
            if image is None or not getattr(image, "size", 0):
                raise RuntimeError(f"Isaac camera has no RGB frame: {spec.prim_path}")
            frames.append((spec.label, image[:, :, :3].copy()))
        return frames

    def depth(self) -> Any:
        if not self._cameras:
            raise RuntimeError("Isaac scene has no camera configured for depth")
        self._app.update()
        depth = self._cameras[0][1].get_depth()
        if depth is None:
            raise RuntimeError("Isaac head camera has no depth frame")
        return depth.copy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        for _, camera in self._cameras:
            destroy = getattr(camera, "destroy", None)
            if destroy is not None:
                try:
                    destroy()
                except Exception as exc:
                    errors.append(exc)
        try:
            self._timeline.stop()
        except Exception as exc:
            errors.append(exc)
        finally:
            self._app.close()
        if errors:
            raise RuntimeError(
                "Isaac runtime cleanup failed: "
                + "; ".join(str(error) for error in errors)
            )
