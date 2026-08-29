from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from ..tools import Tool, object_schema
from ..types import Json


class ManipulationBackend(Protocol):
    """Adapter for a VLA, motion planner, or learned whole-body skill server."""

    def observe(self) -> Mapping[str, Any]: ...

    def pick(self, object_name: str) -> Mapping[str, Any]: ...

    def place(self, target: str) -> Mapping[str, Any]: ...


class DryRunManipulationBackend:
    def __init__(self, objects: Sequence[str] = ("red cup",), surfaces: Sequence[str] = ("table",)) -> None:
        self.objects = list(objects)
        self.surfaces = list(surfaces)
        self.holding: str | None = None

    def observe(self) -> Mapping[str, Any]:
        return {
            "objects": list(self.objects),
            "surfaces": list(self.surfaces),
            "holding": self.holding,
            "dry_run": True,
        }

    def pick(self, object_name: str) -> Mapping[str, Any]:
        if self.holding is not None:
            raise ValueError(f"gripper already holds {self.holding}")
        if object_name not in self.objects:
            raise ValueError(f"object not visible: {object_name}")
        self.objects.remove(object_name)
        self.holding = object_name
        return {"state": "succeeded", "holding": self.holding, "dry_run": True}

    def place(self, target: str) -> Mapping[str, Any]:
        if self.holding is None:
            raise ValueError("gripper is empty")
        if target not in self.surfaces:
            raise ValueError(f"placement target not visible: {target}")
        placed = self.holding
        self.holding = None
        return {"state": "succeeded", "placed": placed, "target": target, "dry_run": True}


class ManipulationModule:
    name = "manipulation"

    def __init__(self, backend: ManipulationBackend, *, requires_approval: bool = True) -> None:
        self.backend = backend
        self.requires_approval = requires_approval

    def tools(self) -> Sequence[Tool]:
        return (
            Tool(
                name="inspect_workspace",
                description="Observe manipulation-relevant objects, surfaces, and gripper state.",
                parameters=object_schema({}),
                handler=lambda _: dict(self.backend.observe()),
            ),
            Tool(
                name="pick_object",
                description=(
                    "Ask the manipulation backend to pick a named visible object. The backend "
                    "owns grasp selection, collision checking, whole-body motion, and verification."
                ),
                parameters=object_schema(
                    {"object": {"type": "string"}, "reason": {"type": "string"}},
                    ["object", "reason"],
                ),
                handler=lambda args: self.backend.pick(str(args["object"])),
                kind="action",
                requires_approval=self.requires_approval,
            ),
            Tool(
                name="place_object",
                description="Place the currently held object on or in a named visible target.",
                parameters=object_schema(
                    {"target": {"type": "string"}, "reason": {"type": "string"}},
                    ["target", "reason"],
                ),
                handler=lambda args: self.backend.place(str(args["target"])),
                kind="action",
                requires_approval=self.requires_approval,
            ),
        )

    def snapshot(self) -> Json:
        return dict(self.backend.observe())
