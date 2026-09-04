from __future__ import annotations

import base64
import io
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..modules.camera import CameraFrame
from ..modules.semantic_map import Pose3D
from ..navigation_core import (
    LocalNavigationObservation,
    LocalObstacle,
    VelocityCommand,
)
from ..scene_bundle import IsaacCameraSpec, IsaacSimScene
from .head_camera import D455, HeadCameraFrame, HeadCameraSpec, mask_depth


@dataclass(frozen=True)
class DepthFrame:
    """One depth image plus the camera model needed to place its pixels in the world.

    ``depth`` is an HxW array of distance to the image plane (range along the
    optical axis) in metres. ``fx``/``fy``/``cx``/``cy`` are pixel intrinsics
    for that resolution. ``rotation`` (3x3, row-major) and ``position`` are the
    pose of the camera's *optical* frame -- +X right, +Y down, +Z forward -- in
    the same world frame the runtime reports the robot pose in.
    """

    depth: Any
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: tuple[tuple[float, float, float], ...]
    position: tuple[float, float, float]


def rotation_from_quaternion(
    quaternion: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Row-major rotation matrix for a (w, x, y, z) quaternion."""
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not norm or not math.isfinite(norm):
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


class IsaacRuntime(Protocol):
    """Small testable boundary around the Kit/PhysX runtime."""

    def pose(self) -> Pose3D: ...

    def set_pose(self, pose: Pose3D) -> None: ...

    def step(self, seconds: float) -> None: ...

    def set_physics_enabled(self, enabled: bool) -> None: ...

    def rgb(self) -> Sequence[tuple[str, Any]]: ...

    def depth_frame(self) -> DepthFrame: ...

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
        #: Called after every step, on the stepping thread. Kit renders only
        #: from the thread that drives it, so the head camera stream ticks here.
        self.step_hooks: list[Callable[[], None]] = []

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
            self._after_step()
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
        self._after_step()

    def _after_step(self) -> None:
        for hook in list(self.step_hooks):
            hook()

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


class IsaacHeadCamera:
    """Colour and metric depth from the runtime's head camera (its first camera).

    Must be used from the thread driving the Kit app, which is why the stream
    that consumes it ticks from ``IsaacSimBase.step_hooks``.
    """

    def __init__(self, runtime: IsaacRuntime, *, spec: HeadCameraSpec = D455, clock: Any = time.time) -> None:
        self.runtime = runtime
        self.spec = spec
        self._clock = clock

    def capture(self) -> HeadCameraFrame:
        import numpy as np

        frames = list(self.runtime.rgb())
        if not frames:
            raise RuntimeError("Isaac runtime has no camera to capture from")
        rgb = np.asarray(frames[0][1], dtype=np.uint8)
        depth = np.asarray(self.runtime.depth_frame().depth, dtype=np.float32)
        return HeadCameraFrame(rgb, mask_depth(depth, self.spec), float(self._clock()))

    def close(self) -> None:
        return None


class IsaacDepthObserver:
    """Robot-relative obstacle observation from the head depth image.

    Every depth pixel is back-projected with the camera intrinsics and placed in
    the world with the camera pose, so the floor is rejected by its height
    rather than by where it happens to fall in the image. A head camera pitched
    towards the ground therefore does not report the floor as an obstacle. What
    remains is the nearest return inside the robot's forward body corridor that
    lies between ``min_obstacle_height_m`` and ``max_obstacle_height_m`` above
    the floor at ``ground_z_m``. Anything lower than the minimum height counts
    as floor, so kerbs and steps are a job for terrain perception, not this.

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
        corridor_half_width_m: float = 0.35,
        ground_z_m: float = 0.0,
        min_obstacle_height_m: float = 0.15,
        max_obstacle_height_m: float = 2.0,
    ) -> None:
        if maximum_obstacle_distance_m <= 0 or obstacle_radius_m < 0:
            raise ValueError("depth observer distances must be non-negative")
        if corridor_half_width_m <= 0:
            raise ValueError("corridor_half_width_m must be positive")
        if not math.isfinite(ground_z_m):
            raise ValueError("ground_z_m must be finite")
        if not 0 <= min_obstacle_height_m < max_obstacle_height_m:
            raise ValueError("obstacle heights must satisfy 0 <= min < max")
        self.runtime = runtime
        self.maximum_obstacle_distance_m = maximum_obstacle_distance_m
        self.obstacle_radius_m = obstacle_radius_m
        self.corridor_half_width_m = corridor_half_width_m
        self.ground_z_m = ground_z_m
        self.min_obstacle_height_m = min_obstacle_height_m
        self.max_obstacle_height_m = max_obstacle_height_m

    def observe(self, pose: Pose3D) -> LocalNavigationObservation:
        try:
            import numpy as np

            frame = self.runtime.depth_frame()
            depth = np.asarray(frame.depth, dtype=np.float32)
            if depth.ndim != 2 or not depth.size:
                raise ValueError("depth image must be a non-empty HxW array")
            intrinsics = (frame.fx, frame.fy, frame.cx, frame.cy)
            if (
                not all(math.isfinite(value) for value in intrinsics)
                or frame.fx <= 0
                or frame.fy <= 0
            ):
                raise ValueError(
                    "depth camera intrinsics must be finite with positive focal lengths"
                )
            rotation = np.asarray(frame.rotation, dtype=np.float64)
            position = np.asarray(frame.position, dtype=np.float64)
            if rotation.shape != (3, 3) or position.shape != (3,):
                raise ValueError(
                    "depth camera pose needs a 3x3 rotation and a 3-vector position"
                )
            if not (np.isfinite(rotation).all() and np.isfinite(position).all()):
                raise ValueError("depth camera pose must be finite")

            valid = np.isfinite(depth) & (depth > 0)
            if not valid.any():
                raise ValueError("depth image contains no finite positive range")
            rows, cols = np.nonzero(valid)
            z = depth[rows, cols].astype(np.float64)
            # Pixel centres sit half a pixel past the index; the principal point
            # is expressed in the same continuous pixel coordinates.
            x = (cols + 0.5 - frame.cx) / frame.fx * z
            y = (rows + 0.5 - frame.cy) / frame.fy * z
            world = np.stack((x, y, z), axis=-1) @ rotation.T + position

            # Reject the floor by height, not by image row: a pitched camera
            # puts the floor anywhere in the frame, and at any range.
            height = world[:, 2] - self.ground_z_m
            above_floor = height >= self.min_obstacle_height_m
            body = above_floor & (height <= self.max_obstacle_height_m)

            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            dx = world[:, 0] - pose.x
            dy = world[:, 1] - pose.y
            forward = dx * c + dy * s
            left = -dx * s + dy * c
            corridor = (
                body & (forward > 0) & (np.abs(left) <= self.corridor_half_width_m)
            )
            counts = (
                f"floor_px={int(np.count_nonzero(~above_floor))} "
                f"body_px={int(np.count_nonzero(body))} "
                f"corridor_px={int(np.count_nonzero(corridor))}"
            )
            if corridor.any():
                # A low percentile rejects isolated invalid/edge pixels while
                # still reacting to a meaningful object in the forward corridor.
                distance = float(np.percentile(forward[corridor], 5.0))
                near = corridor & (forward <= distance + self.obstacle_radius_m)
                lateral = float(np.median(left[near]))
            else:
                distance = math.inf
                lateral = 0.0
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
                    pose.x + distance * c - lateral * s,
                    pose.y + distance * s + lateral * c,
                    self.obstacle_radius_m,
                    "head_depth_forward",
                ),
            )
        return LocalNavigationObservation(
            source="isaac_head_depth",
            obstacles=obstacles,
            frame=pose.frame,
            detail=f"forward_range_m={distance:.3f} {counts}",
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
                if not is_prim_path_valid(spec.prim_path) and spec.mount_prim is not None:
                    self._define_mounted_camera(spec, is_prim_path_valid)
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

    @staticmethod
    def _define_mounted_camera(spec: IsaacCameraSpec, is_prim_path_valid: Any) -> None:
        """Author a head camera prim under a robot link, D455-shaped by default.

        USD cameras share MuJoCo's convention (-Z forward, +Y up), so the same
        quaternion places both. The focal length is chosen for the spec's
        horizontal FOV at the default 20.955 mm aperture; the vertical aperture
        follows the resolution's aspect so pixels stay square.
        """
        import omni.usd
        from pxr import Gf, UsdGeom

        assert spec.mount_prim is not None
        if not is_prim_path_valid(spec.mount_prim):
            raise ValueError(
                f"Isaac camera mount prim does not exist in the stage: {spec.mount_prim}"
            )
        head = HeadCameraSpec(
            name=spec.label,
            width=spec.width,
            height=spec.height,
            horizontal_fov_deg=spec.horizontal_fov_deg,
            mount_xyz=spec.mount_xyz,
            pitch_down_deg=spec.pitch_down_deg,
        )
        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera.Define(stage, spec.prim_path)
        aperture = 20.955
        camera.CreateFocalLengthAttr(float(head.usd_focal_length_mm(aperture)))
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(float(head.usd_vertical_aperture_mm(aperture)))
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
        # Mount offsets are metres; the stage may not be.
        per_metre = 1.0 / float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*(value * per_metre for value in head.mount_xyz)))
        w, x, y, z = head.mujoco_quat()
        xform.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))

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

    def depth_frame(self) -> DepthFrame:
        if not self._cameras:
            raise RuntimeError("Isaac scene has no camera configured for depth")
        import numpy as np

        self._app.update()
        _, camera = self._cameras[0]
        depth = camera.get_depth()
        if depth is None:
            raise RuntimeError("Isaac head camera has no depth frame")
        intrinsics = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)
        # Isaac's "ros" camera axes are the optical convention the observer
        # back-projects in: +X right, +Y down, +Z forward.
        position, quaternion = camera.get_world_pose(camera_axes="ros")
        return DepthFrame(
            depth=depth.copy(),
            fx=float(intrinsics[0, 0]),
            fy=float(intrinsics[1, 1]),
            cx=float(intrinsics[0, 2]),
            cy=float(intrinsics[1, 2]),
            rotation=rotation_from_quaternion(quaternion),
            position=tuple(float(value) for value in position),
        )

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
