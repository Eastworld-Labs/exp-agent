"""The `find_object` tool: go and look for something that is not in frame.

##### THIS WALKS THE ROBOT, SEVERAL TIMES, UNDER ONE APPROVAL. #####

## Why one tool and not a tool per leg

`navigate_to` goes to a labelled place. `local_planner` walks up to a thing
already in the picture. Between them is the case neither covers: the robot is in
the right room and the thing is not in frame -- behind a counter, round a
corner, on the other side of the island.

The obvious shape is two more tools (`look_around`, `search_nearby`) and one
motion each, letting the mission model drive. That was tried on paper and it is
worse. A four-leg search costs four reasoning turns, four operator approvals and
four images kept in the transcript for ever, which is minutes of wall clock and
enough clicking that the realistic operator response is to switch the gate to
auto -- and an auto gate is strictly less safe than one approval a person
actually reads. So the loop runs HERE, on the host, and the model sees one
result.

The precedent is not new: `navigate_to` already blocks for an entire walk that
Nav2 loops internally, and `NestedManipulationAgent` is already a bounded
sub-episode the outer loop sees as one result.

## What keeps it honest

- **The approval carries the budget.** The operator is not approving "a search",
  they are approving at most N legs inside R metres for at most T seconds, and
  the tool refuses to exceed any of them.
- **Every leg is an ordinary Nav2 goal.** Costmaps, MPPI, the recovery
  behaviours and the collision monitor are all still between this and the
  motors. The link still publishes one topic.
- **It never picks a coordinate the model chose.** The model names a THING and
  optionally a HINT in words. Candidate standpoints come from geometry
  (`search.standpoints`), and the only thing a model does is choose among them.
- **Stopping stops the robot.** ⚠️ Everywhere else in this stack, cancelling a
  mission does NOT stop a goal Nav2 already accepted -- `Nav2MqttBackend` says
  so on every cancelled result. A tool that walks six legs cannot live with
  that, so cancelling here publishes the robot's CURRENT POSE as a new goal,
  which is a goal Nav2 replaces the running one with and arrives at instantly.
  That is still inside the one-entry publish table; nothing new crosses the link.
- **It hands off rather than finishing the job.** Found means "grounded, and the
  robot is facing it". The last few metres stay with `local_planner`, which owns
  the clearance arithmetic and must not be duplicated here.
"""
from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

from ..local_planner import LocalPlanner, wrap_angle
from ..modules.semantic_map import Pose3D
from ..search import (
    Standpoint,
    VisibilityMap,
    standpoints,
    sweep_yaws,
)
from ..tools import Tool, object_schema
from ..types import Json
from .navigation import NavigationBackend

#: Legs one search may walk. Six is two or three times round a piece of
#: furniture; a search that needs more is not searching, it is wandering, and
#: the wall-clock budget should not be the only thing that notices.
DEFAULT_MAX_LEGS = 6

#: How far from where the search began it may roam. This is also a LOCALISATION
#: budget, not only a politeness one: on the real robot, scan-to-map matching
#: needs the prior point cloud to have described the space, and a search that
#: wanders into a room nobody surveyed is running on odometry alone.
DEFAULT_RADIUS_M = 4.0

#: Wall clock. The binding limit in practice -- a G1 walks about 0.5 m/s and
#: Nav2's own goal timeout is 120 s per leg.
DEFAULT_MAX_SECONDS = 180.0

#: How far the robot must have moved before a search for a target it has ALREADY
#: found is worth running again.
#:
#: ⚠️ THIS BRAKE IS NOT DEFENSIVE PROGRAMMING, IT IS A MEASURED FAILURE. Without
#: it a mission loops: `find_object` reports "found, facing it, nothing moved",
#: `local_planner` then refuses -- because the approach is blocked, or because
#: re-grounding on the next frame disagreed -- and the model, reading a refusal
#: that says "the object is not visible", reasonably calls `find_object` again.
#: It finds it again, instantly, having moved nothing. Observed in the SONIC sim
#: on 2026-09-05: five find_object/local_planner pairs, two of them zero-motion,
#: burning the model-call budget without the robot taking a step.
#:
#: 0.3 m matches `modules/local_planner.MIN_PROGRESS_M` for the same reason it
#: was chosen there: it is a "did the robot actually move" check, and it wants
#: slack rather than precision.
MIN_PROGRESS_M = 0.3

#: Quarter turns in the opening sweep. Four covers the circle at an 87-degree
#: field with overlap.
SWEEP_QUARTERS = 4


class SearchModule:
    """`find_object`: sweep, then walk to the best place to look, until found."""

    name = "search"

    def __init__(
        self,
        planner: LocalPlanner,
        backend: NavigationBackend,
        *,
        picker: Any = None,
        requires_approval: bool = True,
        max_legs: int = DEFAULT_MAX_LEGS,
        radius_m: float = DEFAULT_RADIUS_M,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        min_progress_m: float = MIN_PROGRESS_M,
        cancel: Any = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.planner = planner
        self.backend = backend
        #: Optional. Without one the highest-scoring candidate wins, which is
        #: pure geometry and perfectly serviceable; with one, a cheap vision
        #: model gets to say "the bucket is more likely round the left end".
        self.picker = picker
        self.requires_approval = requires_approval
        self.max_legs = max_legs
        self.radius_m = radius_m
        self.max_seconds = max_seconds
        self.min_progress_m = min_progress_m
        self.cancel = cancel
        self._clock = clock
        #: target -> where the robot stood when a search last FOUND it. The
        #: brake reads this; see MIN_PROGRESS_M.
        self._found_at: dict[str, tuple[float, float]] = {}
        #: Across the whole mission, not one call: a model that calls this five
        #: times has spent five searches' worth of walking either way.
        self.searches: list[Json] = []
        self.legs_walked = 0
        self._visibility: VisibilityMap | None = None
        self._visited: list[tuple[float, float]] = []

    # ----------------------------------------------------------------- tool --
    def tools(self) -> Sequence[Tool]:
        return (
            Tool(
                name="find_object",
                description=(
                    "WALKS THE ROBOT AROUND to find something that is NOT in the camera "
                    "frame right now -- behind a counter, round a corner, on the far side "
                    "of the room. Use this AFTER navigate_to has taken the robot to the "
                    "labelled place the thing should be in, when local_planner has refused "
                    "because the object is not visible. It turns on the spot first, then "
                    "walks to places its own obstacle map says would reveal floor nobody "
                    "has looked at, grounding on a fresh picture each time. It stops as "
                    "soon as it finds the thing, and leaves the robot FACING it -- then "
                    "call local_planner to close the last few metres. Read the whole "
                    "result: `outcome` says whether it found it, ran out of places to "
                    "look, or ran out of budget, and those mean different things. "
                    "'exhausted' means the thing is not in this area and no further "
                    "search here will help."
                ),
                parameters=object_schema(
                    {
                        "target": {
                            "type": "string",
                            "description": (
                                "The thing to look for, as a person would name it: "
                                "'red bucket', 'cardboard box', 'chair'. One object, "
                                "not a place and not a direction."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "Why the robot should search here now, in one sentence. "
                                "An operator reads this before approving, and approving "
                                "this ONE step authorises every leg of the search."
                            ),
                        },
                        "hint": {
                            "type": "string",
                            "description": (
                                "Optional, and worth giving. Where you think it is, in "
                                "plain words -- 'probably behind the kitchen island', "
                                "'likely along the wall on the left'. This does NOT set a "
                                "coordinate; it only breaks ties between places the "
                                "robot's own map already says are worth looking. Say what "
                                "you can see in the picture that makes you think so."
                            ),
                        },
                        "max_legs": {
                            "type": "integer",
                            "description": (
                                f"Optional. How many walks this search may make, 1 to "
                                f"{DEFAULT_MAX_LEGS}. Defaults to {DEFAULT_MAX_LEGS}. "
                                "Lower it when the thing should be very close by."
                            ),
                        },
                        "radius_m": {
                            "type": "number",
                            "description": (
                                "Optional. How far from HERE the robot may roam while "
                                f"searching, in metres, up to {DEFAULT_RADIUS_M}. "
                                f"Defaults to {DEFAULT_RADIUS_M}."
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

    # -------------------------------------------------------------- running --
    def _run(self, arguments: Mapping[str, Any]) -> Json:
        target = str(arguments["target"]).strip()
        if not target:
            raise ValueError("target must name something to look for")
        max_legs = self._bounded_int(arguments.get("max_legs"), self.max_legs, "max_legs")
        radius_m = self._bounded_float(arguments.get("radius_m"), self.radius_m, "radius_m")
        hint = str(arguments.get("hint") or "").strip()

        if self.legs_walked >= self.max_legs:
            raise ValueError(
                f"searching has already walked {self.legs_walked} legs this mission, "
                f"which is the budget. Report what the robot has seen, or call "
                "request_human -- another search will do the same thing again."
            )

        deadline = self._clock() + self.max_seconds
        init = self.planner.init_pose_source.init_pose(self.planner.expected_grid_frame)
        anchor = (init.odom.x, init.odom.y)
        self._check_progress(target, anchor)
        self._visibility = VisibilityMap.around(
            anchor[0], anchor[1], radius_m, frame=init.odom.frame
        )
        self._visited = []
        legs: list[Json] = []
        offered: list[Json] = []

        # ---- 0. it may simply be in front of the robot already --------------
        seen = self._look(target)
        if seen is not None:
            return self._result(
                "found", target, legs, offered, radius_m,
                detail=(
                    f"{target!r} was already in the current camera frame; nothing moved. "
                    "Call local_planner to walk up to it."
                ),
                grounding=seen,
            )

        # ---- 1. turn on the spot --------------------------------------------
        # Cheapest thing that can work, and it settles the common case: the
        # object is beside or behind the robot rather than hidden by anything.
        for index, yaw in enumerate(sweep_yaws(init.odom.yaw, SWEEP_QUARTERS), start=1):
            stop = self._should_stop(deadline)
            if stop:
                return self._result(stop, target, legs, offered, radius_m,
                                    detail=self._stop_detail(stop))
            turn = self._go(Pose3D(init.map.x, init.map.y, yaw=self._to_map_yaw(init, yaw),
                                   frame=init.map.frame), turn_only=True)
            legs.append({"kind": "turn", "quarter": index, **turn})
            seen = self._look(target)
            if seen is not None:
                return self._result(
                    "found", target, legs, offered, radius_m,
                    detail=(
                        f"{target!r} came into frame after turning on the spot. The robot "
                        "is facing it. Call local_planner to walk up to it."
                    ),
                    grounding=seen,
                )

        # ---- 2. walk to somewhere that reveals more --------------------------
        while len(legs) - SWEEP_QUARTERS < max_legs:
            stop = self._should_stop(deadline)
            if stop:
                return self._result(stop, target, legs, offered, radius_m,
                                    detail=self._stop_detail(stop))
            try:
                found = self._candidates(anchor, radius_m)
            except ValueError as exc:
                return self._result(
                    "blocked", target, legs, offered, radius_m, detail=str(exc)
                )
            if not found:
                return self._result(
                    "exhausted", target, legs, offered, radius_m,
                    detail=(
                        f"there is nowhere left within {radius_m:.1f} m that would show the "
                        "robot any floor it has not already looked at, and "
                        f"{target!r} was not in any of those views. It is not in this part "
                        "of the room. Try a different labelled place, or report that."
                    ),
                )
            resolution = self._visibility.resolution if self._visibility else 0.1
            offered.append({
                "leg": len(legs) - SWEEP_QUARTERS + 1,
                "candidates": [c.as_json(resolution) for c in found],
            })
            choice, why = self._pick(target, hint, found)
            if choice is None:
                return self._result(
                    "exhausted", target, legs, offered, radius_m,
                    detail=(
                        f"the robot has places it could look, but none of them plausibly "
                        f"holds {target!r}: {why}"
                    ),
                )
            walk = self._walk_to(init, choice)
            self.legs_walked += 1
            self._visited.append((choice.x, choice.y))
            legs.append({
                "kind": "walk",
                "to": choice.as_json(resolution),
                "why": why,
                **walk,
            })
            seen = self._look(target)
            if seen is not None:
                return self._result(
                    "found", target, legs, offered, radius_m,
                    detail=(
                        f"{target!r} came into frame after walking to a place that could "
                        "see round what was blocking it. The robot is facing it. Call "
                        "local_planner to walk up to it."
                    ),
                    grounding=seen,
                )

        return self._result(
            "budget", target, legs, offered, radius_m,
            detail=(
                f"{max_legs} legs walked without finding {target!r}, which is this "
                "search's budget. The robot is somewhere it has not been before, so "
                "looking at the fresh picture and deciding is worth more than another "
                "identical search."
            ),
        )

    def _check_progress(self, target: str, here: tuple[float, float]) -> None:
        """Refuse a search that repeats one which already succeeded.

        ⚠️ MEASURED AGAINST THE ROBOT'S POSITION, NOT AGAINST THE LAST RESULT.
        Whether the last search reported "found" is not the question; whether
        the robot has since MOVED is. A `local_planner` that walked part way, a
        `navigate_to` somewhere else and an approach that was refused before
        anything moved all look identical from the result side, and only the
        pose tells them apart.

        The message has to name the real problem, because the refusal the model
        just read from `local_planner` -- "not visible in the current camera
        frame" -- points it straight back here.
        """
        previous = self._found_at.get(target)
        if previous is None:
            return
        moved = math.hypot(here[0] - previous[0], here[1] - previous[1])
        if moved >= self.min_progress_m:
            return
        raise ValueError(
            f"a search already found {target!r} from this exact spot and the robot has "
            f"not moved since ({moved:.2f} m). Searching again will point the robot at "
            "it again and change nothing. THE PROBLEM IS THE APPROACH, NOT THE FINDING: "
            "read what local_planner actually said. If it refused because there is no "
            "free place to stand, the way is blocked and the robot must come at the "
            "object from somewhere else -- navigate_to a different labelled place "
            "first. If it refused because the object is not visible, the two cameras "
            "disagree about a thing at the edge of the frame, and getting the robot to "
            "a different spot is what settles it. If neither is possible, say so and "
            "call request_human."
        )

    # ------------------------------------------------------------- the parts --
    def _look(self, target: str) -> Json | None:
        """Ground on a fresh frame and record what the camera just saw.

        Returns the grounding when the object is there, None when it is not.
        ⚠️ A REFUSAL IS THE ORDINARY CASE HERE and must never end the search:
        `VisionGrounder.ground` raises for "not visible", which is exactly the
        answer this loop is built around.
        """
        try:
            frame = self.planner._colour_frame()
        except ValueError:
            # No fresh picture. Nothing to ground and nothing to record; the
            # loop will try again from the next standpoint.
            return None
        try:
            init = self.planner.init_pose_source.init_pose(self.planner.expected_grid_frame)
            grid = self.planner.grid_source.grid()
        except ValueError:
            init = grid = None
        if init is not None and grid is not None and self._visibility is not None:
            self._visibility.observe(
                grid,
                init.odom.x,
                init.odom.y,
                init.odom.yaw,
                hfov_rad=math.radians(float(self.planner.geometry.horizontal_fov_deg)),
                max_range_m=self.planner.max_range_m,
            )
        try:
            grounding = self.planner.grounder.ground(target, frame)
        except ValueError:
            return None
        except Exception:  # noqa: BLE001 - a transport failure is not "absent"
            return None
        if not grounding.found or grounding.box is None:
            return None
        return grounding.as_json()

    def _candidates(self, anchor: tuple[float, float], radius_m: float) -> list[Standpoint]:
        grid = self.planner.grid_source.grid()
        init = self.planner.init_pose_source.init_pose(self.planner.expected_grid_frame)
        if grid.frame != init.odom.frame:
            raise ValueError(
                f"the local costmap is in frame {grid.frame!r} but the robot's odometry "
                f"is in {init.odom.frame!r}; they must match to search over them"
            )
        assert self._visibility is not None
        free = grid.to_gridmap().inflated(self.planner.footprint_radius_m)
        return standpoints(
            self._visibility,
            grid,
            free,
            (init.odom.x, init.odom.y, init.odom.yaw),
            anchor=anchor,
            radius_m=radius_m,
            hfov_rad=math.radians(float(self.planner.geometry.horizontal_fov_deg)),
            max_range_m=self.planner.max_range_m,
            visited=self._visited,
        )

    def _pick(
        self, target: str, hint: str, found: Sequence[Standpoint]
    ) -> tuple[Standpoint | None, str]:
        """Which candidate to walk to, and why in one sentence."""
        if self.picker is None:
            return (
                found[0],
                "the place its obstacle map says reveals the most unseen floor",
            )
        try:
            frame = self.planner._colour_frame()
        except ValueError:
            return (found[0], "no fresh picture to choose on; took the most revealing place")
        try:
            index, why = self.picker.pick(target, hint, frame, found)
        except Exception as exc:  # noqa: BLE001
            # A picker that fell over must not end a search: geometry alone is
            # a complete answer, just a less informed one.
            return (
                found[0],
                f"the picker could not answer ({type(exc).__name__}); took the most "
                "revealing place",
            )
        if index is None:
            return (None, why or "none of them looked likely")
        if not 0 <= index < len(found):
            return (found[0], "the picker named a place that was not offered; took the best")
        return (found[index], why or "chosen from the picture")

    def _walk_to(self, init: Any, choice: Standpoint) -> Json:
        """One leg, as an ordinary map-frame goal through the navigation backend."""
        from ..local_planner import local_to_map, odom_to_local

        local_xy = odom_to_local(choice.x, choice.y, init.odom)
        local_yaw = wrap_angle(choice.yaw - init.odom.yaw)
        goal = local_to_map(local_xy[0], local_xy[1], local_yaw, init.map)
        return self._go(goal)

    def _go(self, goal: Pose3D, *, turn_only: bool = False) -> Json:
        result = dict(self.backend.navigate(goal))
        return {
            "state": result.get("state"),
            "verdict_source": result.get("verdict_source"),
            "turn_only": turn_only,
        }

    def _to_map_yaw(self, init: Any, odom_yaw: float) -> float:
        """An odometry heading as a map heading, through the sampled offset."""
        return wrap_angle(init.map.yaw + wrap_angle(odom_yaw - init.odom.yaw))

    # -------------------------------------------------------------- stopping --
    def _should_stop(self, deadline: float) -> str:
        if self.cancel is not None and self.cancel.is_set():
            return "cancelled"
        if self._clock() >= deadline:
            return "budget"
        return ""

    def _stop_detail(self, stop: str) -> str:
        if stop != "cancelled":
            return (
                f"the {self.max_seconds:.0f}s this search may take ran out before "
                "the object was found."
            )
        self._halt()
        return (
            "the operator stopped the mission mid-search. A goal Nav2 has already "
            "accepted keeps running, so this published the robot's own current "
            "position as a new goal, which replaces it -- the robot should have "
            "stopped where it stood. Only the emergency stop is a guarantee."
        )

    def _halt(self) -> None:
        """Replace the running goal with one at the robot's current pose.

        ⚠️ THE ONLY STOP THIS TOOL HAS, and it is a goal rather than a brake.
        Publishing where the robot already is makes Nav2 supersede the leg it
        was walking and arrive immediately. It goes through the same one-entry
        publish table as every other goal; nothing new crosses the link, and the
        emergency stop remains the only real guarantee.

        ⚠️ THE POSE COMES FROM THE INIT-POSE SOURCE, NOT FROM `backend.status()`,
        AND THE DIFFERENCE IS A FRAME. The simulator's pose arrives on `/odom`
        with `header.frame_id: "odom"` in a frame that IS the map frame by
        construction; `MqttInitPose` normalises that to `"map"` and the backend's
        status does not. A halt built from the status would therefore publish a
        goal stamped `odom` on the one target where stopping matters most, and
        every other goal this tool sends is already stamped `map`.
        """
        here: Pose3D | None = None
        try:
            here = self.planner.init_pose_source.init_pose(
                self.planner.expected_grid_frame
            ).map
        except Exception:  # noqa: BLE001
            # No fresh pose pair. Fall back to the backend's own reading rather
            # than not stopping at all, and force the goal frame: a stop in the
            # wrong frame is worse than the frame being assumed.
            try:
                pose = dict(self.backend.status()).get("pose") or {}
                here = Pose3D(
                    float(pose["x"]), float(pose["y"]),
                    yaw=float(pose.get("yaw", 0.0)), frame="map",
                )
            except (KeyError, TypeError, ValueError):
                return
        if here is None:
            return
        try:
            self.backend.navigate(here)
        except Exception:  # noqa: BLE001 - a failed halt must not mask the stop
            pass

    # -------------------------------------------------------------- reporting --
    def _bounded_int(self, value: Any, ceiling: int, name: str) -> int:
        if value is None:
            return ceiling
        number = int(value)
        if number < 1:
            raise ValueError(f"{name} must be at least 1")
        if number > ceiling:
            raise ValueError(
                f"{name}={number} is more than the {ceiling} this robot's search is "
                f"configured for. Ask for {ceiling} or fewer."
            )
        return number

    def _bounded_float(self, value: Any, ceiling: float, name: str) -> float:
        if value is None:
            return ceiling
        number = float(value)
        if number <= 0:
            raise ValueError(f"{name} must be positive")
        if number > ceiling:
            raise ValueError(
                f"{name}={number:g} is more than the {ceiling:g} m this robot's search "
                f"is configured for. Ask for {ceiling:g} or less."
            )
        return number

    def _result(
        self,
        outcome: str,
        target: str,
        legs: list[Json],
        offered: list[Json],
        radius_m: float,
        *,
        detail: str,
        grounding: Json | None = None,
    ) -> Json:
        walked = sum(1 for leg in legs if leg.get("kind") == "walk")
        record: Json = {
            "outcome": outcome,
            "target": target,
            "found": outcome == "found",
            "legs_walked": walked,
            "turns": sum(1 for leg in legs if leg.get("kind") == "turn"),
            "radius_m": radius_m,
            "detail": detail,
            "legs": legs,
            "candidates_offered": offered,
            "grounding": grounding,
            "looked_at": self._visibility.as_json() if self._visibility else None,
            "legs_left_this_mission": max(0, self.max_legs - self.legs_walked),
        }
        if outcome == "found":
            record["next_step"] = (
                "call local_planner with the same target: it is in frame now, and that "
                "tool owns the metric approach and the clearance the body needs."
            )
            # WHERE the robot was standing when it found it, so a second search
            # from the same spot can be refused rather than re-run. Read here
            # rather than reused from the top of the call because a search that
            # WALKED has moved since. A pose the source cannot give is simply
            # not recorded: the brake is worth having, not worth raising over.
            try:
                found_at = self.planner.init_pose_source.init_pose(
                    self.planner.expected_grid_frame
                )
                self._found_at[target] = (found_at.odom.x, found_at.odom.y)
            except Exception:  # noqa: BLE001
                pass
        self.searches.append({
            "target": target, "outcome": outcome, "legs_walked": walked,
        })
        return record

    # -------------------------------------------------------------- snapshot --
    def snapshot(self) -> Json:
        """Cheap and never raising: a snapshot is built on EVERY turn."""
        state: Json = {
            "searches": len(self.searches),
            "legs_walked": self.legs_walked,
            "legs_left": max(0, self.max_legs - self.legs_walked),
            "last": self.searches[-1] if self.searches else None,
            "radius_m": self.radius_m,
        }
        if self._visibility is not None:
            try:
                state["looked_at"] = self._visibility.as_json()
            except Exception as exc:  # noqa: BLE001
                state["looked_at"] = {"error": f"{type(exc).__name__}: {exc}"}
        return state


__all__ = [
    "DEFAULT_MAX_LEGS",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_RADIUS_M",
    "SWEEP_QUARTERS",
    "SearchModule",
]
