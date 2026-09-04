from __future__ import annotations

import math
from typing import Any, Mapping, Protocol, Sequence

from ..tools import Tool, object_schema
from ..types import Json
from .semantic_map import Pose3D, SemanticMapModule


class NavigationBackend(Protocol):
    """Adapter for Nav2, a 3-D planner, or a vendor navigation stack."""

    def navigate(self, goal: Pose3D) -> Mapping[str, Any]: ...

    def status(self) -> Mapping[str, Any]: ...


class DryRunNavigationBackend:
    """Safe demonstration backend. It never publishes robot commands."""

    def __init__(self) -> None:
        self.pose = Pose3D(0.0, 0.0)
        self._status: Json = {"state": "idle", "pose": self.pose.as_json()}

    def navigate(self, goal: Pose3D) -> Mapping[str, Any]:
        distance = math.hypot(goal.x - self.pose.x, goal.y - self.pose.y)
        self.pose = goal
        self._status = {
            "state": "succeeded",
            "pose": goal.as_json(),
            "planned_distance_m": round(distance, 3),
            "dry_run": True,
        }
        return {
            **self._status,
            "pipeline": [
                "global planner produced a map-frame path",
                "local controller would produce vx/vy/yaw_rate",
                "locomotion controller/WBC would track those references",
            ],
        }

    def status(self) -> Mapping[str, Any]:
        return self._status


class NavigationModule:
    name = "navigation"

    def __init__(
        self,
        semantic_map: SemanticMapModule,
        backend: NavigationBackend,
        *,
        requires_approval: bool = True,
    ) -> None:
        self.map = semantic_map
        self.backend = backend
        self.requires_approval = requires_approval

    def tools(self) -> Sequence[Tool]:
        return (
            Tool(
                name="navigate_to",
                description=(
                    "WALKS THE ROBOT to a named semantic location, and does not return "
                    "until it has arrived or failed. The navigation backend -- not you -- "
                    "plans the path, avoids obstacles and tracks it; you choose the "
                    "destination. Read the whole result before deciding anything: it says "
                    "whether the robot arrived, and whether that is the planner's own "
                    "verdict or an inference from where the robot ended up."
                ),
                parameters=object_schema(
                    {
                        # ⚠️ THE ENUM IS THE REFUSAL, MADE STRUCTURAL. Built from
                        # the loaded semantic map at schema time, so a
                        # destination nobody labelled cannot be emitted rather
                        # than being caught after the fact -- and on a provider
                        # with strict tool schemas it is impossible, not merely
                        # invalid. An empty map yields an empty enum, which is
                        # the honest shape: there is nowhere to go.
                        "location": {
                            "type": "string",
                            "enum": self.map.names(),
                            "description": "A destination from the semantic map.",
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "Why the robot should go there now, in one sentence. "
                                "An operator reads this before approving the step."
                            ),
                        },
                    },
                    ["location", "reason"],
                ),
                handler=self._navigate,
                kind="action",
                requires_approval=self.requires_approval,
            ),
            Tool(
                name="navigation_status",
                description="Read the latest navigation state and localization pose.",
                parameters=object_schema({}),
                handler=lambda _: dict(self.backend.status()),
            ),
        )

    def snapshot(self) -> Json:
        return dict(self.backend.status())

    def _navigate(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        goal = self.map.resolve_navigation_goal(str(arguments["location"]))
        return self.backend.navigate(goal)
