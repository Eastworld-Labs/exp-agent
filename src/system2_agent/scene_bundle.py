from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IsaacCameraSpec:
    label: str
    prim_path: str
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class IsaacSimScene:
    """Isaac Sim-specific references inside an otherwise shared scene bundle."""

    stage_usd: Path
    robot_prim: str
    cameras: tuple[IsaacCameraSpec, ...] = ()
    renderer: str = "RaytracedLighting"
    robot_usd: Path | None = None


@dataclass(frozen=True)
class SceneBundle:
    """The three representations needed by a robot simulation scene."""

    mujoco_xml: Path | None
    navigation_grid: Path
    semantic_map: Path
    gaussian_splat: Path | None = None
    gaussian_alignment: tuple[tuple[float, ...], ...] | None = None
    collision_mesh: Path | None = None
    collision_mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    collision_mesh_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    collision_mesh_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    navigation_footprint_radius_m: float = 0.32
    initial_pose: tuple[float, float, float] | None = None
    isaac_sim: IsaacSimScene | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "SceneBundle":
        manifest = Path(path).expanduser().resolve()
        raw: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
        base = manifest.parent

        def resolve(value: str | None) -> Path | None:
            if value is None:
                return None
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

        alignment = raw.get("gaussian_alignment")
        collision = raw.get("collision_mesh")
        collision_path = collision.get("path") if isinstance(collision, dict) else collision
        isaac = raw.get("isaac_sim")
        bundle = cls(
            # external_mjcf is the preferred name. mujoco_xml remains accepted
            # for manifests created before scenes became independently loaded.
            mujoco_xml=resolve(raw.get("external_mjcf", raw.get("mujoco_xml"))),
            navigation_grid=resolve(raw["navigation_grid"]),  # type: ignore[arg-type]
            semantic_map=resolve(raw["semantic_map"]),  # type: ignore[arg-type]
            gaussian_splat=resolve(raw.get("gaussian_splat")),
            gaussian_alignment=None if alignment is None else tuple(tuple(float(v) for v in row) for row in alignment),
            collision_mesh=resolve(collision_path),
            collision_mesh_scale=cls._tuple(collision, "scale", 3, (1.0, 1.0, 1.0)),
            collision_mesh_position=cls._tuple(collision, "position", 3, (0.0, 0.0, 0.0)),
            collision_mesh_quaternion=cls._tuple(
                collision, "quaternion", 4, (1.0, 0.0, 0.0, 0.0)
            ),
            navigation_footprint_radius_m=float(raw.get("navigation_footprint_radius_m", 0.32)),
            initial_pose=cls._initial_pose(raw.get("initial_pose")),
            isaac_sim=cls._isaac_scene(isaac, base),
        )
        bundle.validate()
        return bundle

    def validate(self) -> None:
        for label, path in (
            ("navigation_grid", self.navigation_grid),
            ("semantic_map", self.semantic_map),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if self.mujoco_xml is not None and not self.mujoco_xml.exists():
            raise FileNotFoundError(f"mujoco_xml does not exist: {self.mujoco_xml}")
        if self.gaussian_splat is not None and not self.gaussian_splat.exists():
            raise FileNotFoundError(f"gaussian_splat does not exist: {self.gaussian_splat}")
        if self.collision_mesh is not None and not self.collision_mesh.exists():
            raise FileNotFoundError(f"collision_mesh does not exist: {self.collision_mesh}")
        if self.isaac_sim is not None and not self.isaac_sim.stage_usd.exists():
            raise FileNotFoundError(
                f"isaac_sim.stage_usd does not exist: {self.isaac_sim.stage_usd}"
            )
        if (
            self.isaac_sim is not None
            and self.isaac_sim.robot_usd is not None
            and not self.isaac_sim.robot_usd.exists()
        ):
            raise FileNotFoundError(
                f"isaac_sim.robot_usd does not exist: {self.isaac_sim.robot_usd}"
            )
        if self.gaussian_alignment is not None and (
            len(self.gaussian_alignment) != 4 or any(len(row) != 4 for row in self.gaussian_alignment)
        ):
            raise ValueError("gaussian_alignment must be a 4x4 world_T_gs matrix")
        if self.navigation_footprint_radius_m <= 0:
            raise ValueError("navigation_footprint_radius_m must be positive")

    @staticmethod
    def _isaac_scene(value: object, base: Path) -> IsaacSimScene | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("isaac_sim must be an object")
        try:
            stage = Path(str(value["stage_usd"])).expanduser()
            stage = stage.resolve() if stage.is_absolute() else (base / stage).resolve()
            robot_prim = str(value["robot_prim"])
        except KeyError as exc:
            raise ValueError("isaac_sim requires stage_usd and robot_prim") from exc
        if not robot_prim.startswith("/"):
            raise ValueError("isaac_sim.robot_prim must be an absolute USD prim path")
        cameras: list[IsaacCameraSpec] = []
        for item in value.get("cameras", []):
            if not isinstance(item, dict):
                raise ValueError("each isaac_sim camera must be an object")
            try:
                camera = IsaacCameraSpec(
                    label=str(item["label"]),
                    prim_path=str(item["prim_path"]),
                    width=int(item.get("width", 640)),
                    height=int(item.get("height", 480)),
                )
            except KeyError as exc:
                raise ValueError("Isaac camera requires label and prim_path") from exc
            if not camera.prim_path.startswith("/"):
                raise ValueError("Isaac camera prim paths must be absolute")
            if not camera.label:
                raise ValueError("Isaac camera labels must not be empty")
            if camera.width <= 0 or camera.height <= 0:
                raise ValueError("Isaac camera dimensions must be positive")
            cameras.append(camera)
        if len({camera.label for camera in cameras}) != len(cameras):
            raise ValueError("Isaac camera labels must be unique")
        renderer = str(value.get("renderer", "RaytracedLighting"))
        if not renderer:
            raise ValueError("isaac_sim.renderer must not be empty")
        robot_usd_value = value.get("robot_usd")
        robot_usd = None
        if robot_usd_value is not None:
            robot_usd = Path(str(robot_usd_value)).expanduser()
            robot_usd = (
                robot_usd.resolve()
                if robot_usd.is_absolute()
                else (base / robot_usd).resolve()
            )
        return IsaacSimScene(
            stage_usd=stage,
            robot_prim=robot_prim,
            cameras=tuple(cameras),
            renderer=renderer,
            robot_usd=robot_usd,
        )

    @staticmethod
    def _initial_pose(value: object) -> tuple[float, float, float] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("initial_pose must be an object with x, y, and optional yaw")
        try:
            pose = (float(value["x"]), float(value["y"]), float(value.get("yaw", 0.0)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("initial_pose must contain numeric x, y, and yaw values") from exc
        if not all(math.isfinite(component) for component in pose):
            raise ValueError("initial_pose values must be finite")
        return pose

    @staticmethod
    def _tuple(
        block: object, key: str, length: int, default: tuple[float, ...]
    ) -> tuple[float, ...]:
        if not isinstance(block, dict) or key not in block:
            return default
        values = tuple(float(value) for value in block[key])
        if len(values) != length:
            raise ValueError(f"collision_mesh.{key} must contain {length} values")
        return values
