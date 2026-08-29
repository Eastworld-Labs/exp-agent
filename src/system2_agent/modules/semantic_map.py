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

    def __init__(self, locations: Mapping[str, Pose3D]) -> None:
        self._locations = {key.casefold(): value for key, value in locations.items()}

    @classmethod
    def from_json(cls, path: str | Path) -> "SemanticMapModule":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({name: Pose3D(**pose) for name, pose in raw["locations"].items()})

    def resolve(self, name: str) -> Pose3D:
        try:
            return self._locations[name.casefold()]
        except KeyError as exc:
            raise ValueError(f"unknown semantic location: {name}") from exc

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
        )

    def snapshot(self) -> Json:
        return {"known_locations": sorted(self._locations)}
