from __future__ import annotations

import json
import math
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


def _key(name: str) -> str:
    """The lookup key for a destination name.

    Casefolded, and spaces and dashes folded to underscores, so "Kitchen Table",
    "kitchen table" and "kitchen_table" are one place. The dashboard's own
    `whereIs` does exactly this, and the two have to agree or an operator
    labels somewhere the model then cannot address. There is deliberately NO
    fuzzy or prefix matching beyond that: a near-miss must be a refusal.
    """
    return name.strip().casefold().replace(" ", "_").replace("-", "_")


class SemanticMapModule:
    """Language names on top of poses supplied by SLAM/mapping infrastructure."""

    name = "semantic_map"

    def __init__(
        self,
        locations: Mapping[str, Pose3D],
        metadata: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        map_name: str = "",
        errors: Sequence[str] = (),
        fault: str = "",
    ) -> None:
        self._locations = {_key(name): pose for name, pose in locations.items()}
        self._display_names = {_key(name): name for name in locations}
        self._metadata = {
            _key(name): dict(value) for name, value in (metadata or {}).items()
        }
        #: Which map these labels belong to, when they came from a map-bound file.
        self.map_name = map_name
        #: Rows that were DROPPED while loading, each with why. Surfaced in the
        #: snapshot rather than raised: one malformed label must not make the
        #: other nine unusable, and silently having fewer destinations than the
        #: file lists is the version of this that nobody notices.
        self.errors = tuple(errors)
        #: Non-empty when the whole document is unusable -- currently only when
        #: it describes a different map than the one the robot is localized
        #: against. ⚠️ THAT IS NOT A WARNING: navigating to a label from another
        #: building's map is a confident walk to the wrong place, and nothing
        #: downstream could detect it. Every resolve raises while this is set.
        self.fault = fault

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

    def names(self) -> list[str]:
        """Every destination, as the model should spell them. Sorted."""
        return sorted(self._display_names.values())

    @classmethod
    def from_places_json(
        cls, path: str | Path, *, expect_map: str = ""
    ) -> "SemanticMapModule":
        """Load a `maps/<map>.places.json` semantic map.

        The format is the one the dashboard's Places sheet authors, so the same
        labels serve the browser and this agent and cannot drift apart:

            {"map": "...", "updated": "...", "places": [
              {"name": "kitchen", "x": 1.7, "y": -15.8, "yaw": 15,
               "tags": [], "provenance": "measured", "note": ""}]}

        ⚠️ `yaw` IS IN DEGREES THERE AND RADIANS HERE. It is also REQUIRED and
        never defaulted: a waypoint in a doorway means *through* the doorway,
        and yaw is the only thing that says which way. A row without one is
        dropped rather than pointed north.

        ⚠️ A ROW IS DROPPED, NOT COERCED. Duplicate names are refused rather
        than last-wins. Every drop lands in `.errors` and in the snapshot the
        model sees, because a destination that silently does not exist is
        indistinguishable from one the operator forgot to add.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        doc_map = str(raw.get("map") or "")
        locations: dict[str, Pose3D] = {}
        metadata: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        for index, place in enumerate(raw.get("places") or []):
            name = str(place.get("name") or "").strip()
            if not name:
                errors.append(f"place #{index}: no name")
                continue
            if _key(name) in locations:
                errors.append(f"{name!r}: duplicate name, dropped")
                continue
            try:
                x = float(place["x"])
                y = float(place["y"])
                yaw_deg = float(place["yaw"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{name!r}: needs numeric x, y and yaw (degrees)")
                continue
            locations[name] = Pose3D(x=x, y=y, yaw=math.radians(yaw_deg))
            metadata[name] = {
                "tags": list(place.get("tags") or []),
                "provenance": str(place.get("provenance") or "assumed"),
                "note": str(place.get("note") or ""),
                "yaw_deg": yaw_deg,
                "map": doc_map,
            }

        fault = ""
        if expect_map and doc_map and doc_map != expect_map:
            fault = (
                f"the semantic map describes {doc_map!r} but the robot is "
                f"localized against {expect_map!r}. Refusing every destination: "
                f"a label from another map is a confident walk to the wrong place."
            )
        return cls(locations, metadata, map_name=doc_map, errors=errors, fault=fault)

    @classmethod
    def load(cls, path: str | Path, *, expect_map: str = "") -> "SemanticMapModule":
        """Read either supported layout, chosen by what the file contains."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "places" in raw:
            return cls.from_places_json(path, expect_map=expect_map)
        return cls.from_json(path)

    def resolve(self, name: str) -> Pose3D:
        if self.fault:
            raise ValueError(self.fault)
        try:
            return self._locations[_key(name)]
        except KeyError as exc:
            # ⚠️ REFUSE, NEVER RELOCATE, AND ALWAYS CARRY THE REAL LIST. A
            # nearest-match here would send a robot somewhere nobody named.
            known = ", ".join(self.names()) or "(nothing is labelled on this map)"
            raise ValueError(
                f"unknown semantic location: {name}. Known destinations: {known}"
            ) from exc

    def describe(self, name: str) -> Json:
        key = _key(name)
        pose = self.resolve(name)
        return {
            "name": self._display_names[key],
            "pose": pose.as_json(),
            **self._metadata.get(key, {}),
        }

    def resolve_navigation_goal(self, name: str) -> Pose3D:
        key = _key(name)
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
        snapshot: Json = {"known_locations": self.names()}
        if self.map_name:
            snapshot["map"] = self.map_name
        if self.errors:
            snapshot["unusable_entries"] = list(self.errors)
        if self.fault:
            snapshot["unusable"] = self.fault
        return snapshot
