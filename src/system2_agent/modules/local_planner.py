"""The `local_planner` tool: walk up to something the robot can see.

##### THIS WALKS THE ROBOT. ##### It is the only tool besides `navigate_to`
that does, and unlike that one its destination is not on any map -- it comes out
of a photograph taken a moment ago.

## Why it is one tool and not two

Grounding, ranging, placing a goal and walking to it could each be a tool, and a
model would then be choosing coordinates -- which is the one thing this whole
stack is built to keep it from doing. So the model names a THING and a reason,
an operator approves that intent, and everything metric happens inside one call
where it can be checked against the costmap before anything moves.

## Why it can be called repeatedly

A local plan is frequently NOT the whole approach: the costmap is a rolling
window a few metres across, while the camera sees across a room. So a call may
return `reached_standoff: false` with the metres still to go, and the right
response is to look at the fresh frame and call again. Each leg re-grounds on a
new picture and a new costmap, which is also how the approach survives the
target being half-occluded at the start.

That loop needs a brake, and there are two: a budget on how many legs one
mission may walk, and a check that each leg actually got closer. A model that
keeps calling while the range does not shrink is not approaching anything --
something is blocking it, or the grounding is landing on a different object each
time -- and the honest answer is to stop and say so.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..local_planner import LocalPlan, LocalPlanner
from ..tools import Tool, object_schema
from ..types import Json
from .navigation import NavigationBackend

#: How many legs one mission may walk in total, across all targets. Twelve is
#: three or four full approaches; a mission that needs more is not approaching,
#: it is wandering, and the wall-clock and model-call budgets should not be the
#: only thing that notices.
DEFAULT_MAX_APPROACHES = 12

#: A leg must close at least this much of the remaining distance to count as
#: progress. Below it, Nav2's own goal tolerance (0.10 m) plus the depth
#: ranger's own scatter could account for the whole difference. Left at 0.3
#: rather than tracked down to the tolerance: this is a "did the robot actually
#: move" check, and it wants slack, not precision.
MIN_PROGRESS_M = 0.3


class LocalPlannerModule:
    name = "local_planner"

    def __init__(
        self,
        planner: LocalPlanner,
        backend: NavigationBackend,
        *,
        requires_approval: bool = True,
        max_approaches: int = DEFAULT_MAX_APPROACHES,
        min_progress_m: float = MIN_PROGRESS_M,
    ) -> None:
        self.planner = planner
        self.backend = backend
        self.requires_approval = requires_approval
        self.max_approaches = max_approaches
        self.min_progress_m = min_progress_m
        self.legs: list[Json] = []

    # ----------------------------------------------------------------- tool --
    def tools(self) -> Sequence[Tool]:
        return (
            Tool(
                name="local_planner",
                description=(
                    "WALKS THE ROBOT up to something it can SEE RIGHT NOW in the head "
                    "camera -- a sink, a chair, a box -- without using the map. Name "
                    "the thing; the robot's own depth camera measures where it is and "
                    "its local costmap decides whether the way is clear. Use this for "
                    "the last few metres, AFTER navigate_to has taken the robot to the "
                    "labelled place the thing is in, and only when the thing is in the "
                    "current camera frame. Read the whole result: `reached_standoff` "
                    "false means this was one leg of a longer approach and you should "
                    "look at the fresh frame and CALL THIS AGAIN with the same target. "
                    "A refusal that the object is not visible, or that nothing solid is "
                    "there, means move or turn with navigate_to first -- calling again "
                    "from the same spot will refuse again."
                ),
                parameters=object_schema(
                    {
                        "target": {
                            "type": "string",
                            "description": (
                                "The thing to walk up to, as a person would name it: "
                                "'sink', 'blue chair', 'cardboard box'. One object, "
                                "not a place and not a direction."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "Why the robot should approach it now, in one sentence. "
                                "An operator reads this before approving the step."
                            ),
                        },
                        # ⚠️ CLEARANCE IS OFFERED FIRST, AND ON PURPOSE. `standoff_m`
                        # is measured from the robot's CENTRE, and its body reaches
                        # ~0.31 m forward -- so a model told "0.65 m minimum" and
                        # asked to get 0.3 m from something concludes the robot
                        # cannot do it and gives up, when 0.3 m of actual clearance
                        # is a 0.61 m standoff and perfectly legal. That exact
                        # failure is why this parameter exists.
                        "clearance_m": {
                            "type": "number",
                            "description": (
                                "Optional. How much clear air to leave between the "
                                "FRONT OF THE ROBOT'S BODY and the object, in metres, "
                                f"between {self.planner.min_clearance_m:.2f} and "
                                f"{self.planner.max_clearance_m:.2f}. This is the "
                                "number to use when the mission needs the robot close "
                                "enough to reach or inspect something -- it is the gap "
                                "a person would measure. Defaults to "
                                f"{self.planner.clearance_for_standoff(self.planner.standoff_m):.2f}. "
                                f"The floor of {self.planner.min_clearance_m:.2f} m is the "
                                "closest this robot can STAND without tripping its own "
                                "collision stop, and it is a property of the robot's body "
                                "rather than a cautious setting: asking for less is "
                                "refused, not clamped."
                            ),
                        },
                        "standoff_m": {
                            "type": "number",
                            "description": (
                                "Optional, and usually the wrong one to reach for: "
                                "prefer clearance_m. Distance from the robot's CENTRE "
                                "to the object, in metres, between "
                                f"{self.planner.standoff_min_m:g} and "
                                f"{self.planner.standoff_max_m:g}. Defaults to "
                                f"{self.planner.standoff_m:g}. Because the body reaches "
                                f"{self.planner.nose_reach_m:.2f} m forward, this is "
                                f"about {self.planner.nose_reach_m:.2f} m MORE than the "
                                "gap you would see. Pass this or clearance_m, not both."
                            ),
                        },
                    },
                    ["target", "reason"],
                ),
                handler=self._run,
                kind="action",
                requires_approval=self.requires_approval,
            ),
        )

    # --------------------------------------------------------------- running --
    def _run(self, arguments: Mapping[str, Any]) -> Json:
        target = str(arguments["target"]).strip()
        if not target:
            raise ValueError("target must name something to walk up to")
        if len(self.legs) >= self.max_approaches:
            raise ValueError(
                f"local_planner has already walked {len(self.legs)} legs this mission, "
                f"which is the budget. If the robot still is not where it needs to be, "
                "something is wrong with the approach rather than with the last step: "
                "report what you have seen, or call request_human."
            )

        standoff = arguments.get("standoff_m")
        clearance = arguments.get("clearance_m")
        plan = self.planner.plan(
            target,
            standoff_m=None if standoff is None else float(standoff),
            clearance_m=None if clearance is None else float(clearance),
        )
        self._check_progress(target, plan)

        result = dict(self.backend.navigate(plan.standoff_map))
        self.legs.append(
            {
                "target": target,
                "range_m": round(plan.range.range_m, 2),
                "leg_m": round(plan.leg_m, 2),
                "remaining_m": round(plan.remaining_m, 2),
                "reached_standoff": plan.final,
                # What the robot actually ended up with, which the costmap may
                # have made larger than requested. Kept per-leg so a mission
                # that asked to get close can be audited on whether it did.
                "body_clearance_m": round(plan.body_clearance[0], 2),
                "state": result.get("state"),
            }
        )
        return {**result, "local_plan": plan.as_json()}

    def _check_progress(self, target: str, plan: LocalPlan) -> None:
        """Refuse a leg that repeats one which did not work.

        ⚠️ MEASURED AGAINST THE RANGE, NOT AGAINST THE PLAN. The previous call
        planned a leg; whether the robot walked it is a different question, and
        the only evidence that it did is that the thing is now nearer. A goal
        Nav2 aborted, a robot the collision monitor stopped and a target the
        grounder found somewhere new all look identical from the plan side and
        all show up here as a range that did not shrink.
        """
        previous = [leg for leg in self.legs if leg["target"] == target]
        if not previous:
            return
        last = previous[-1]["range_m"]
        if plan.range.range_m > last - self.min_progress_m:
            raise ValueError(
                f"the last approach to {target!r} did not get the robot closer: it "
                f"measured {last:.1f} m away then and {plan.range.range_m:.1f} m now. "
                "Walking the same leg again will do the same thing. Check the camera "
                "frame -- something may be blocking the way, the last walk may have "
                "been stopped, or this may be a different object of the same name. "
                "Try approaching from elsewhere with navigate_to, or call request_human."
            )

    # -------------------------------------------------------------- snapshot --
    def snapshot(self) -> Json:
        """Cheap and never raising: a snapshot is built on EVERY turn."""
        state: Json = {
            "legs_walked": len(self.legs),
            "legs_left": max(0, self.max_approaches - len(self.legs)),
            "last": self.legs[-1] if self.legs else None,
        }
        try:
            state["costmap"] = self.planner.grid_source.status()
        except Exception as exc:  # noqa: BLE001
            state["costmap"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        depth = self.planner.depth_source
        if depth is not None:
            try:
                state["depth"] = depth.status()
            except Exception as exc:  # noqa: BLE001
                state["depth"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        # What the camera can physically see, which on a pitched sensor is much
        # less than a reader assumes. Belongs in the snapshot rather than the
        # system prompt: it differs per robot and the prompt must stay static.
        geometry = self.planner.geometry
        state["view"] = {
            "pitch_down_deg": round(float(getattr(geometry, "pitch_down_deg", 0.0)), 2),
            "hfov_deg": round(float(getattr(geometry, "horizontal_fov_deg", 0.0)), 1),
            "max_range_m": self.planner.max_range_m,
            "max_leg_m": self.planner.max_leg_m,
            "standoff_m": self.planner.standoff_m,
            # Stated every turn so a model deciding how close it can get reads a
            # number rather than inferring one from a refusal it has not hit yet.
            "closest_body_clearance_m": round(self.planner.min_clearance_m, 2),
        }
        pitch = state["view"]["pitch_down_deg"]
        if pitch > 20:
            state["view"]["note"] = (
                f"this camera is aimed {pitch:.0f} degrees DOWN, so it sees the floor "
                "a few metres ahead and not much above knee height beyond about a "
                "metre: things on tables and shelves are out of frame, not missing"
            )
        return state
