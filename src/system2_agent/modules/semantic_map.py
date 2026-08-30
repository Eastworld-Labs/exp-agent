from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..tools import Tool, object_schema
from ..types import Json


@dataclass(frozen=True)
class Pose3D:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    frame: str = "map"

    def as_json(self) -> Json:
        return asdict(self)


class SemanticMapModule:
    """Language names on top of poses supplied by SLAM/mapping infrastructure."""

    name = "semantic_map"

    def __init__(
        self,
        locations: Mapping[str, Pose3D],
        metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._locations = {key.casefold(): value for key, value in locations.items()}
        self._display_names = {key.casefold(): key for key in locations}
        self._metadata = {
            key.casefold(): dict(value) for key, value in (metadata or {}).items()
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "SemanticMapModule":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # Importers may attach category/interactivity metadata. Navigation
        # consumes only the pose and should remain forward-compatible with
        # those richer semantic records.
        locations = {
            name: Pose3D(
                x=float(pose["x"]),
                y=float(pose["y"]),
                z=float(pose.get("z", 0.0)),
                yaw=float(pose.get("yaw", 0.0)),
                frame=str(pose.get("frame", "map")),
            )
            for name, pose in raw["locations"].items()
        }
        pose_fields = {"x", "y", "z", "yaw", "frame"}
        metadata = {
            name: {key: value for key, value in pose.items() if key not in pose_fields}
            for name, pose in raw["locations"].items()
        }
        return cls(locations, metadata)

    def resolve(self, name: str) -> Pose3D:
        try:
            return self._locations[name.casefold()]
        except KeyError as exc:
            raise ValueError(f"unknown semantic location: {name}") from exc

    def describe(self, name: str) -> Json:
        key = name.casefold()
        pose = self.resolve(name)
        return {
            "name": self._display_names[key],
            "pose": pose.as_json(),
            **self._metadata.get(key, {}),
        }

    def resolve_navigation_goal(self, name: str) -> Pose3D:
        key = name.casefold()
        pose = self.resolve(name)
        if self._metadata.get(key, {}).get("navigation_target", True) is not True:
            raise ValueError(
                f"semantic location {name!r} is not a navigation target; use its approach pose"
            )
        return pose

    def tools(self) -> Sequence[Tool]:
        return (
            Tool(
                name="list_locations",
                description="List language-addressable destinations in the semantic map.",
                parameters=object_schema({}),
                handler=lambda _: {"locations": sorted(self._locations)},
            ),
            Tool(
                name="resolve_location",
                description="Resolve a semantic destination name to its map-frame pose.",
                parameters=object_schema(
                    {"location": {"type": "string"}}, ["location"]
                ),
                handler=lambda args: self.resolve(str(args["location"])).as_json(),
            ),
            Tool(
                name="describe_location",
                description=(
                    "Read a semantic location's map pose plus room, object, affordance, "
                    "approach, and interactivity metadata when available."
                ),
                parameters=object_schema(
                    {"location": {"type": "string"}}, ["location"]
                ),
                handler=lambda args: self.describe(str(args["location"])),
            ),
        )

    def snapshot(self) -> Json:
        return {"known_locations": sorted(self._locations)}
