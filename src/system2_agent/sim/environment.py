from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..navigation_core import LocalNavigationObserver, MobileBase
from ..scene_bundle import SceneBundle


SimulationBackend = Literal["mujoco", "isaac"]


@dataclass
class SimulationEnvironment:
    """Backend-neutral resources consumed by the System-2 simulation CLI."""

    backend: SimulationBackend
    base: MobileBase
    camera: Any | None = None
    local_observer: LocalNavigationObserver | None = None
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
) -> SimulationEnvironment:
    """Create either simulator without leaking its API into System-2 modules."""

    workspace = Path(workspace).expanduser().resolve()
    if backend == "mujoco":
        from ..scene_loader import SceneLoader
        from .g1_mujoco import G1MuJoCoBase, MuJoCoCamera
        from .mugs_camera import MuGSCamera

        robot_scene = (
            workspace
            / "GR00T-WholeBodyControl"
            / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
        )
        loaded = SceneLoader(robot_scene).load(scene)
        base = None
        try:
            base = G1MuJoCoBase(
                model_path=loaded.model_path,
                robot_class_path=workspace / "robot_class",
            )
            selected_splat = Path(splat).expanduser().resolve() if splat else scene.gaussian_splat
            camera_backend = None
            if with_vision or selected_splat is not None:
                if selected_splat is not None:
                    if camera is None:
                        raise ValueError("MuGS requires a named MuJoCo camera")
                    camera_backend = MuGSCamera(
                        base,
                        selected_splat,
                        camera=camera,
                        world_T_gs=scene.gaussian_alignment,
                    )
                else:
                    camera_backend = MuJoCoCamera(base, camera=camera)
        except Exception:
            if base is not None:
                try:
                    base.close()
                finally:
                    loaded.close()
            else:
                loaded.close()
            raise
        return SimulationEnvironment(
            "mujoco", base, camera_backend, _owned=(loaded, base)
        )
    if backend == "isaac":
        if splat is not None:
            raise ValueError(
                "Isaac renders splats authored in the USD/USDZ stage; --splat is MuJoCo-only"
            )
        if scene.isaac_sim is None:
            raise ValueError("scene bundle has no isaac_sim configuration")
        from .isaac import (
            IsaacCameraBackend,
            IsaacDepthObserver,
            IsaacSimBase,
            OmniverseIsaacRuntime,
        )

        runtime = OmniverseIsaacRuntime(scene.isaac_sim, headless=headless)
        if with_vision and not scene.isaac_sim.cameras:
            runtime.close()
            raise ValueError("Isaac vision requested but the scene defines no cameras")
        base = IsaacSimBase(runtime, isaac_actuator)
        camera_backend = IsaacCameraBackend(runtime) if with_vision else None
        local_observer = (
            IsaacDepthObserver(runtime)
            if isaac_local_depth and scene.isaac_sim.cameras
            else None
        )
        return SimulationEnvironment(
            "isaac",
            base,
            camera_backend,
            local_observer,
            _owned=(base,),
        )
    raise ValueError(f"unsupported simulation backend: {backend}")
