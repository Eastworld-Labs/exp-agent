"""A NavigationBackend that drives Nav2 on the robot over the fleet link.

    navigate_to("kitchen")
      -> a map-frame /goal_pose, CBOR over MQTT
      -> the on-robot fleet agent republishes it as ROS
      -> g1_bridge/goal_relay -> the NavigateToPose action -> Nav2
      -> Nav2's global planner + MPPI -> /cmd_vel -> the collision monitor
      -> whichever locomotion backend is up -> the robot walks

##### THE AGENT CHOOSES A DESTINATION. IT NEVER CHOOSES A VELOCITY. #####
Everything between the goal and the motors -- the costmaps, the obstacle
avoidance, the recovery behaviours, the collision monitor's stop zones, the
backend's own caps -- is on the robot and stays there. That boundary is the
reason a language model is allowed anywhere near this at all.

## Waiting, and the two kinds of "it arrived"

`navigate()` BLOCKS until the walk is over, because that is the contract
NavigationModule expects and because between tool calls is exactly when the
robot is not moving. While it blocks it watches two things:

  `/goal_status`      Nav2's OWN verdict, via g1_bridge/goal_relay. This is the
                      real answer: it distinguishes an abort from a slow walk.
  the pose topic      where the robot says it is (/localization_3d on the real
                      robot, /odom in the simulator). Converging on the goal is
                      an INFERENCE, and it cannot tell those two apart -- a robot
                      the planner gave up on and a robot picking its way round a
                      chair both look like "not there yet" until the timeout.

Every result says which of the two it is, in `verdict_source`, on the result
itself rather than in the system prompt -- a model that read a caveat forty
turns ago and is now looking at `state: arrived` will treat it as the planner's
word.

## Refusing

Pre-flight refusals raise ValueError, which `Tool.run` turns into an ordinary
`ok: false` the model reads as evidence and plans around. They exist because
publishing a goal at a robot that cannot act on it is worse than not
publishing: nothing moves, nothing says why, and the mission burns its budget
waiting for a walk that was never going to start.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

from ..modules.semantic_map import Pose3D
from ..types import Json
from .link import Link
from .wire import decode_bool, decode_goal_status, decode_pose, pose_stamped

#: How the arrival was decided, said on every result.
PLANNER_VERDICT = (
    "the planner's own verdict, reported by Nav2 on /goal_status"
)
DERIVED_VERDICT = (
    "DERIVED from the robot's reported position converging on the goal, not the "
    "planner's own verdict -- this robot's /goal_status did not arrive. An "
    "aborted goal and a slow walk are indistinguishable this way until the timeout."
)


class Nav2MqttBackend:
    """Implements NavigationBackend against the g1_auto_navigation stack."""

    def __init__(
        self,
        link: Link,
        *,
        cancel: threading.Event | None = None,
        pose_topic: str = "/localization_3d",
        arrive_timeout_s: float = 120.0,
        arrive_tol_m: float = 0.6,
        poll_s: float = 0.5,
        pose_stale_s: float = 3.0,
        heartbeat_stale_s: float = 3.0,
        goal_match_m: float = 0.1,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.link = link
        self.cancel = cancel or threading.Event()
        # Where this robot's map-frame pose arrives. The real robot's localizer
        # publishes a PoseStamped on /localization_3d; the simulator publishes
        # ground-truth Odometry on /odom in a frame that IS the map frame.
        # wire.decode_pose reads either shape, so a target is just a topic name.
        self.pose_topic = pose_topic
        self.arrive_timeout_s = arrive_timeout_s
        # Deliberately LOOSER than Nav2's own xy_goal_tolerance (0.10 m under
        # Unitree's gait, 0.25 m under the SONIC overlay): this watches a
        # localizer rather than the planner, so a tighter value would report
        # "did not arrive" for a robot Nav2 is perfectly happy with. 0.6 m stays
        # clear of both, which is why tightening that box did not move it.
        self.arrive_tol_m = arrive_tol_m
        self.poll_s = poll_s
        self.pose_stale_s = pose_stale_s
        # ⚠️ /estop_state AND /sonic/enabled ARE RETAINED AND HEARTBEATED (2 Hz).
        # Retained means the broker hands a late subscriber the LAST value the
        # robot ever published -- and when that publisher is gone, the value
        # outlives it. A `/sonic/enabled: false` left behind by a SONIC session
        # last week would otherwise refuse every goal on a robot now walking
        # happily on Unitree's gait, and a stale `/estop_state: true` would
        # report a stop nobody is enforcing. So both are read through the same
        # arrival-age gate the pose uses: six missed beats and the reading is
        # "nobody is saying", which is the honest state.
        self.heartbeat_stale_s = heartbeat_stale_s
        self.goal_match_m = goal_match_m
        self._now = now
        self._sleep = sleep
        self._last: Json = {"state": "idle"}
        for topic in (pose_topic, "/estop_state", "/goal_status", "/sonic/enabled"):
            link.subscribe(topic)

    # ------------------------------------------------------------- reading --
    def _fresh(self, topic: str, max_age_s: float) -> "tuple[Json, float] | None":
        entry = self.link.latest(topic)
        if entry is None:
            return None
        msg, arrived = entry
        age = self._now() - arrived
        return (msg, age) if age <= max_age_s else None

    def _pose(self) -> "Json | None":
        """Where the robot says it is, IF that is recent enough to act on.

        ⚠️ /localization_3d IS RETAINED BY THE BROKER, so connecting to a robot
        that has been off for an hour hands us its last known pose instantly and
        it looks live. Everything here that reads a pose reads it through this.
        """
        entry = self._fresh(self.pose_topic, self.pose_stale_s)
        if entry is None:
            return None
        pose = decode_pose(entry[0])
        if pose is None:
            return None
        pose["age_ms"] = round(entry[1] * 1000)
        pose["yaw_deg"] = round(math.degrees(pose["yaw"]), 1)
        return pose

    def _estop(self) -> "bool | None":
        """The latch AS THE ROBOT'S MOTION BACKEND REPORTS IT, if it is
        reporting. `None` means nobody fresh is saying -- no motion backend is
        up, or the link is dead -- which is not the same fact as `False`."""
        entry = self._fresh("/estop_state", self.heartbeat_stale_s)
        return None if entry is None else decode_bool(entry[0])

    def _armed(self) -> "bool | None":
        """Whether SONIC is armed, if SONIC is the backend that is up. `None`
        means nothing fresh on the topic: the ordinary state under Unitree's
        built-in gait (it publishes no such topic) AND the state after a SONIC
        session has ended (its last value is retained but no longer heartbeated).
        Neither is a refusal."""
        entry = self._fresh("/sonic/enabled", self.heartbeat_stale_s)
        return None if entry is None else decode_bool(entry[0])

    # ----------------------------------------------------------- the goal ---
    def navigate(self, goal: Pose3D) -> Json:
        try:
            return self._navigate(goal)
        except ValueError:
            # A pre-flight refusal. Let it out as a refusal -- Tool.run turns it
            # into ok:false with the text, which is what the model should read.
            raise
        except Exception as exc:  # noqa: BLE001
            # ⚠️ EVERYTHING ELSE BECOMES A RESULT, NOT AN EXCEPTION. Tool.run
            # only catches (KeyError, TypeError, ValueError), so a broker
            # timeout or a paho error raised from here would propagate out of
            # the agent loop and end the mission with a traceback instead of a
            # step the model could react to.
            self._last = {"state": "failed", "error": f"{type(exc).__name__}: {exc}"}
            return {
                **self._last,
                "goal": _goal_json(goal),
                "pose": self._pose(),
                "verdict_source": "none",
            }

    def _navigate(self, goal: Pose3D) -> Json:
        if not self.link.connected():
            raise ValueError(
                "no link to the robot: the mission service is not connected to the "
                "broker, so a goal cannot reach it. Nothing was published."
            )
        if self._estop() is True:
            raise ValueError(
                "the emergency stop is ENGAGED, as the robot itself reports it. The "
                "robot will ignore every velocity command until a person releases it "
                "(./g1 estop go, or the dashboard). Nothing was published."
            )
        start_pose = self._pose()
        if start_pose is None:
            raise ValueError(
                f"the robot is not reporting a position (nothing on {self.pose_topic} "
                f"within {self.pose_stale_s:.0f}s). Either the localizer is down or the "
                f"robot is off. Navigating without localization is not possible, and "
                f"Nav2 would refuse the goal anyway. Nothing was published."
            )
        if self._armed() is False:
            raise ValueError(
                "the SONIC locomotion backend is present but NOT ARMED, so the robot is "
                "holding a pose and will not walk. Arming starts a 35 kg biped balancing "
                "and is a person's decision (./g1 sonic arm). Nothing was published."
            )

        goal_json = _goal_json(goal)
        sent_at = self._now()
        self.link.publish_cmd("/goal_pose", pose_stamped(goal.x, goal.y, goal.yaw, goal.frame))

        closest = _distance(start_pose, goal_json)
        verdict_source = "derived"
        detail = ""
        state = "unknown"

        while True:
            if self.cancel.is_set():
                state = "cancelled"
                verdict_source = "none"
                detail = (
                    "the operator stopped the mission while the robot was moving. "
                    "##### THIS DID NOT STOP THE ROBOT ##### -- a goal the planner has "
                    "already accepted keeps running. Only the emergency stop or a new "
                    "goal halts it."
                )
                break

            status = self._bound_status(goal_json, sent_at)
            if status is not None:
                state = _state_from_status(status)
                verdict_source = "planner"
                remaining = status.get("distance_remaining")
                if isinstance(remaining, (int, float)):
                    closest = float(remaining)
                detail = str(status.get("reason") or "")
                if status.get("terminal"):
                    break

            pose = self._pose()
            if pose is None:
                state = "lost_localization"
                verdict_source = "none"
                detail = (
                    "the robot stopped reporting a position while it was moving. "
                    "Nothing can be said about where it is."
                )
                break
            distance = _distance(pose, goal_json)
            closest = min(closest, distance)
            if status is None and distance <= self.arrive_tol_m:
                state = "arrived"
                verdict_source = "derived"
                detail = f"the robot's reported position is {distance:.2f} m from the goal."
                break

            if self._now() - sent_at >= self.arrive_timeout_s:
                state = "timed_out"
                detail = (
                    f"after {self.arrive_timeout_s:.0f}s the robot is still {closest:.2f} m "
                    f"from the goal (the closest it came)."
                )
                if verdict_source != "planner":
                    detail += (
                        " No planner verdict arrived, so it is NOT known whether Nav2 "
                        "gave up, is still working, or never accepted the goal."
                    )
                break

            self._sleep(self.poll_s)

        self._last = {
            "state": state,
            "goal": goal_json,
            "elapsed_s": round(self._now() - sent_at, 2),
            "remaining_m": None if closest is None else round(closest, 2),
            "verdict_source": verdict_source,
        }
        return {
            **self._last,
            "pose": self._pose(),
            # ⚠️ ON EVERY RESULT, not in the system prompt. See the module header.
            "arrival_is": PLANNER_VERDICT if verdict_source == "planner" else DERIVED_VERDICT,
            "detail": detail,
        }

    def _bound_status(self, goal_json: Json, sent_at: float) -> "Json | None":
        """The /goal_status message that belongs to OUR goal, or None.

        ⚠️ THE TOPIC IS LATCHED, so a `succeeded` from an hour ago is sitting
        there waiting for a late subscriber -- reading `state` off it without
        this check would report the PREVIOUS goal's verdict as this one's. Bound
        two ways at once: it must have ARRIVED after we published, and its goal
        must be the goal we published.
        """
        entry = self.link.latest("/goal_status")
        if entry is None:
            return None
        msg, arrived = entry
        if arrived < sent_at:
            return None
        status = decode_goal_status(msg)
        if status is None:
            return None
        theirs = status.get("goal") or {}
        try:
            if math.hypot(
                float(theirs["x"]) - goal_json["x"], float(theirs["y"]) - goal_json["y"]
            ) > self.goal_match_m:
                return None
        except (KeyError, TypeError, ValueError):
            return None
        return status

    # ---------------------------------------------------------------- state --
    def status(self) -> Json:
        """Cheap, cached, and it never raises.

        Called on EVERY world snapshot the model is shown, so it must not block
        and must not be able to end a mission by throwing.
        """
        try:
            pose = self._pose()
            estop = self._estop()
            return {
                **self._last,
                "linked": self.link.connected(),
                "pose_topic": self.pose_topic,
                "estop_latched": estop,
                # Whether a motion backend is talking at all. `estop_latched`
                # is None both when nothing is up and when the link is dead;
                # this names the fact so the model does not read an absence
                # as "not stopped".
                "motion_backend_reporting": estop is not None,
                "pose": pose,
                "motion_backend": {True: "armed", False: "disarmed", None: "unknown"}[
                    self._armed()
                ],
                "goal_status_available": self.link.latest("/goal_status") is not None,
            }
        except Exception as exc:  # noqa: BLE001
            return {"state": "unknown", "error": f"{type(exc).__name__}: {exc}"}


def _goal_json(goal: Pose3D) -> Json:
    return {
        "x": round(float(goal.x), 3),
        "y": round(float(goal.y), 3),
        "yaw_deg": round(math.degrees(float(goal.yaw)), 1),
        "frame": goal.frame,
    }


def _distance(pose: Any, goal_json: Json) -> float:
    return math.hypot(float(pose["x"]) - goal_json["x"], float(pose["y"]) - goal_json["y"])


_STATUS_STATES = {
    "succeeded": "arrived",
    "aborted": "aborted",
    "canceled": "canceled",
    "rejected": "rejected",
    "no_server": "no_planner",
    "accepted": "walking",
    "running": "walking",
}


def _state_from_status(status: Json) -> str:
    """Nav2's word for what happened -> the word the model sees.

    An unfamiliar state reads as `aborted`: the goal is over and the robot is
    not known to be at it, which is the safe way to be wrong.
    """
    state = str(status.get("state") or "")
    if state in _STATUS_STATES:
        return _STATUS_STATES[state]
    return "walking" if not status.get("terminal") else "aborted"
