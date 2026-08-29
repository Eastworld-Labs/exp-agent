from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SceneBundle:
    """The three representations needed by a robot simulation scene."""

    mujoco_xml: Path
    navigation_grid: Path
    semantic_map: Path
    gaussian_splat: Path | None = None
    gaussian_alignment: tuple[tuple[float, ...], ...] | None = None

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
        bundle = cls(
            mujoco_xml=resolve(raw["mujoco_xml"]),  # type: ignore[arg-type]
            navigation_grid=resolve(raw["navigation_grid"]),  # type: ignore[arg-type]
            semantic_map=resolve(raw["semantic_map"]),  # type: ignore[arg-type]
            gaussian_splat=resolve(raw.get("gaussian_splat")),
            gaussian_alignment=None if alignment is None else tuple(tuple(float(v) for v in row) for row in alignment),
        )
        bundle.validate()
        return bundle

    def validate(self) -> None:
        for label, path in (
            ("mujoco_xml", self.mujoco_xml),
            ("navigation_grid", self.navigation_grid),
            ("semantic_map", self.semantic_map),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if self.gaussian_splat is not None and not self.gaussian_splat.exists():
            raise FileNotFoundError(f"gaussian_splat does not exist: {self.gaussian_splat}")
        if self.gaussian_alignment is not None and (
            len(self.gaussian_alignment) != 4 or any(len(row) != 4 for row in self.gaussian_alignment)
        ):
            raise ValueError("gaussian_alignment must be a 4x4 world_T_gs matrix")
