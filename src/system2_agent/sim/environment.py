from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..navigation_core import LocalNavigationObserver, MobileBase
from ..scene_bundle import SceneBundle
from .head_camera import D455, FramePublisher, HeadCameraSpec, HeadCameraStream


SimulationBackend = Literal["mujoco", "isaac"]


@dataclass
class SimulationEnvironment:
    """Backend-neutral resources consumed by the System-2 simulation CLI."""

    backend: SimulationBackend
    base: MobileBase
    camera: Any | None = None
    local_observer: LocalNavigationObserver | None = None
    #: The head-camera publisher when streaming to a broker was requested.
    stream: HeadCameraStream | None = None
    _owned: tuple[Any, ...] = ()
    _closed: bool = field(default=False, init=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        for resource in reversed(self._owned):
            try:
                resource.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                "simulation environment cleanup failed: "
                + "; ".join(str(error) for error in errors)
            )

    def __enter__(self) -> "SimulationEnvironment":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def create_simulation_environment(
    backend: SimulationBackend,
    scene: SceneBundle,
    *,
    workspace: str | Path,
    with_vision: bool = False,
    camera: str | int | None = None,
    splat: str | Path | None = None,
    headless: bool = True,
    isaac_local_depth: bool = True,
    isaac_actuator: Any | None = None,
    head_camera: HeadCameraSpec | None = D455,
    stream: FramePublisher | None = None,
    stream_hz: float = 6.0,
) -> SimulationEnvironment:
    """Create either simulator without leaking its API into System-2 modules.

    ``head_camera`` puts the G1's head depth camera on the robot: MuJoCo gets
    it attached to the robot MJCF, Isaac expects the scene bundle's first
    camera to be it (see ``IsaacCameraSpec.mount_prim``). With vision on and
    no named ``camera``, the model then sees the robot's own two previews --
    colour and false-colour range -- rather than a free camera. ``stream``
    additionally publishes those previews and a health line to a broker as
    the sim robot, so the dashboard shows them.
    """

    workspace = Path(workspace).expanduser().resolve()
    if stream is not None and head_camera is None:
        raise ValueError("streaming the head camera needs a head_camera spec")
    if backend == "mujoco":
        from ..scene_loader import SceneLoader
        from .g1_mujoco import G1MuJoCoBase, MuJoCoCamera
        from .mugs_camera import MuGSCamera

        robot_scene = (
            workspace
            / "GR00T-WholeBodyControl"
            / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
        )
        from .head_camera import HeadCameraBackend

        loaded = SceneLoader(robot_scene).load(scene, head_camera=head_camera)
        base = None
        head_stream = None
        try:
            base = G1MuJoCoBase(
                model_path=loaded.model_path,
                robot_class_path=workspace / "robot_class",
                viewer=not headless,
            )
            selected_splat = Path(splat).expanduser().resolve() if splat else scene.gaussian_splat
            camera_backend = None
            if with_vision or selected_splat is not None:
                if selected_splat is not None:
                    # The head camera is a named camera, so MuGS can render
                    # from it without a scene bundle supplying one.
                    if camera is None and head_camera is not None:
                        camera = head_camera.name
                    if camera is None:
                        raise ValueError("MuGS requires a named MuJoCo camera")
                    camera_backend = MuGSCamera(
                        base,
                        selected_splat,
                        camera=camera,
                        world_T_gs=scene.gaussian_alignment,
                    )
                elif camera is None and head_camera is not None:
                    camera_backend = HeadCameraBackend(base.head_camera(head_camera))
                else:
                    camera_backend = MuJoCoCamera(base, camera=camera)
            if stream is not None:
                # Its own camera instance: the stream renders on its thread,
                # under the base lock, while the model's frames come from here.
                head_stream = HeadCameraStream(
                    base.head_camera(head_camera), stream, hz=stream_hz, source="sim:mujoco"
                )
                head_stream.start()
        except Exception:
            if head_stream is not None:
                head_stream.close()
            if base is not None:
                try:
                    base.close()
                finally:
                    loaded.close()
            else:
                loaded.close()
            raise
        owned: tuple[Any, ...] = (loaded, base)
        if camera_backend is not None and hasattr(camera_backend, "close"):
            owned += (camera_backend,)
        if head_stream is not None:
            owned += (head_stream,)
        return SimulationEnvironment(
            "mujoco", base, camera_backend, stream=head_stream, _owned=owned
        )
    if backend == "isaac":
        if splat is not None:
            raise ValueError(
                "Isaac renders splats authored in the USD/USDZ stage; --splat is MuJoCo-only"
            )
        if scene.isaac_sim is None:
            raise ValueError("scene bundle has no isaac_sim configuration")
        from .head_camera import HeadCameraBackend
        from .isaac import (
            IsaacCameraBackend,
            IsaacDepthObserver,
            IsaacHeadCamera,
            IsaacSimBase,
            OmniverseIsaacRuntime,
        )

        runtime = OmniverseIsaacRuntime(scene.isaac_sim, headless=headless)
        if (with_vision or stream is not None) and not scene.isaac_sim.cameras:
            runtime.close()
            raise ValueError("Isaac vision requested but the scene defines no cameras")
        base = IsaacSimBase(runtime, isaac_actuator)
        camera_backend: Any = None
        if with_vision:
            camera_backend = (
                HeadCameraBackend(IsaacHeadCamera(runtime, spec=head_camera))
                if head_camera is not None
                else IsaacCameraBackend(runtime)
            )
        local_observer = (
            IsaacDepthObserver(runtime)
            if isaac_local_depth and scene.isaac_sim.cameras
            else None
        )
        head_stream = None
        owned: tuple[Any, ...] = (base,)
        if stream is not None:
            # Kit renders only from its own thread, so the stream ticks after
            # every step instead of running a thread of its own.
            head_stream = HeadCameraStream(
                IsaacHeadCamera(runtime, spec=head_camera), stream, hz=stream_hz, source="sim:isaac"
            )
            base.step_hooks.append(head_stream.tick)
            owned = (base, head_stream)
        return SimulationEnvironment(
            "isaac",
            base,
            camera_backend,
            local_observer,
            stream=head_stream,
            _owned=owned,
        )
    raise ValueError(f"unsupported simulation backend: {backend}")
