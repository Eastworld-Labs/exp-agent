from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        if self.gaussian_alignment is not None and (
            len(self.gaussian_alignment) != 4 or any(len(row) != 4 for row in self.gaussian_alignment)
        ):
            raise ValueError("gaussian_alignment must be a 4x4 world_T_gs matrix")
        if self.navigation_footprint_radius_m <= 0:
            raise ValueError("navigation_footprint_radius_m must be positive")

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
