from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .model import ChatModel
from .tools import Tool, ToolRegistry, object_schema
from .types import Json, ToolCall

if TYPE_CHECKING:
    from .modules.camera import CameraFrame


MANIPULATION_SYSTEM_PROMPT = """You are a closed-loop manipulation controller.
Use fresh head and wrist images plus robot state to complete one manipulation instruction.

Rules:
- Make exactly one tool call per turn.
- Prefer small end-effector deltas and observe after every action.
- Never invent success. Call finish_manipulation only when the current images and state
  visibly verify the requested final arrangement.
- Stop with fail_manipulation if the task is unsafe, unreachable, or cannot be verified.
- The whole-body controller owns balance; you own only bounded arm and gripper intent.
"""


class ManipulationEmbodiment(Protocol):
    """Robot/simulator boundary used by the nested visual manipulation agent."""

    def observe(self) -> Mapping[str, Any]: ...

    def camera_frames(self) -> Sequence["CameraFrame"]: ...

    def move_end_effector(
        self,
        arm: str,
        translation_m: Sequence[float],
        rotation_rpy_rad: Sequence[float],
        duration_s: float,
    ) -> Mapping[str, Any]: ...

    def set_gripper(self, arm: str, aperture: float) -> Mapping[str, Any]: ...

    def verify(self, instruction: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ManipulationOutcome:
    status: str
    summary: str
    model_calls: int
    events: tuple[Json, ...] = field(default_factory=tuple)

    def as_json(self) -> Json:
        return {
            "status": self.status,
            "summary": self.summary,
            "model_calls": self.model_calls,
            "events": list(self.events),
        }


class NestedManipulationAgent:
    """Blocking Inspect-style VLM policy that returns only after task termination.

    The outer mission agent sees one ``manipulate`` tool call. Internally this agent
    repeatedly observes head/wrist cameras and issues bounded Cartesian/gripper
    actions. A WBC adapter behind ``ManipulationEmbodiment`` converts those intents
    into joint targets while retaining balance control.
    """

    def __init__(
        self,
        model: ChatModel,
        embodiment: ManipulationEmbodiment,
        *,
        max_model_calls: int = 60,
        max_translation_m: float = 0.08,
        max_rotation_rad: float = 0.35,
        max_duration_s: float = 3.0,
        system_prompt: str = MANIPULATION_SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.embodiment = embodiment
        self.max_model_calls = max_model_calls
        self.max_translation_m = max_translation_m
        self.max_rotation_rad = max_rotation_rad
        self.max_duration_s = max_duration_s
        self.system_prompt = system_prompt
        self.registry = ToolRegistry((), self._tools())

    def run(self, instruction: str) -> ManipulationOutcome:
        events: list[Json] = []
        messages: list[Json] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self._observation_content(
                    f"Manipulation instruction: {instruction}\nInitial observation:"
                ),
            },
        ]
        for model_call in range(1, self.max_model_calls + 1):
            turn = self.model.complete(messages, self.registry.schemas())
            messages.append(self._assistant_message(turn))
            if len(turn.tool_calls) != 1:
                messages.append({"role": "user", "content": "Make exactly one tool call."})
                continue
            call = turn.tool_calls[0]
            if call.name == "finish_manipulation":
                verification = dict(self.embodiment.verify(instruction))
                event = {"type": "verification", "result": verification}
                events.append(event)
                if verification.get("succeeded") is True:
                    return ManipulationOutcome(
                        "completed",
                        str(call.arguments["summary"]),
                        model_call,
                        tuple(events),
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(
                            {"ok": False, "error": "verification failed", "data": verification}
                        ),
                    }
                )
            elif call.name == "fail_manipulation":
                return ManipulationOutcome(
                    "failed", str(call.arguments["reason"]), model_call, tuple(events)
                )
            else:
                tool = self.registry.get(call.name)
                result = (
                    {"ok": False, "error": f"unknown tool: {call.name}"}
                    if tool is None
                    else tool.run(call.arguments).as_json()
                )
                events.append(
                    {
                        "type": "tool",
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(result, separators=(",", ":")),
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": self._observation_content("Fresh observation after action:"),
                }
            )
        return ManipulationOutcome(
            "budget_exhausted",
            f"stopped after {self.max_model_calls} manipulation model calls",
            self.max_model_calls,
            tuple(events),
        )

    def _tools(self) -> tuple[Tool, ...]:
        arm = {"type": "string", "enum": ["left", "right"]}
        vector = {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        }
        return (
            Tool(
                "observe_manipulation",
                "Take fresh head and wrist images without moving.",
                object_schema({}),
                lambda _: dict(self.embodiment.observe()),
            ),
            Tool(
                "move_end_effector",
                "Move one hand by a small Cartesian delta in the robot base frame.",
                object_schema(
                    {
                        "arm": arm,
                        "translation_m": vector,
                        "rotation_rpy_rad": vector,
                        "duration_s": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    ["arm", "translation_m", "rotation_rpy_rad", "duration_s", "reason"],
                ),
                self._move,
                kind="action",
            ),
            Tool(
                "set_gripper",
                "Set Dex1 aperture: 0 is closed and 1 is fully open.",
                object_schema(
                    {"arm": arm, "aperture": {"type": "number"}, "reason": {"type": "string"}},
                    ["arm", "aperture", "reason"],
                ),
                self._grip,
                kind="action",
            ),
            Tool(
                "finish_manipulation",
                "Request completion; the embodiment independently verifies the final state.",
                object_schema({"summary": {"type": "string"}}, ["summary"]),
                lambda _: {},
                kind="terminal",
            ),
            Tool(
                "fail_manipulation",
                "Return control safely when completion is impossible or unsafe.",
                object_schema({"reason": {"type": "string"}}, ["reason"]),
                lambda _: {},
                kind="terminal",
            ),
        )

    def _move(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        arm = self._arm(arguments["arm"])
        translation = self._bounded_vector(
            arguments["translation_m"], self.max_translation_m, "translation_m"
        )
        rotation = self._bounded_vector(
            arguments["rotation_rpy_rad"], self.max_rotation_rad, "rotation_rpy_rad"
        )
        duration = float(arguments["duration_s"])
        if not 0.05 <= duration <= self.max_duration_s:
            raise ValueError(f"duration_s must be in [0.05, {self.max_duration_s}]")
        return dict(self.embodiment.move_end_effector(arm, translation, rotation, duration))

    def _grip(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        arm = self._arm(arguments["arm"])
        aperture = float(arguments["aperture"])
        if not 0.0 <= aperture <= 1.0:
            raise ValueError("aperture must be in [0, 1]")
        return dict(self.embodiment.set_gripper(arm, aperture))

    @staticmethod
    def _arm(value: object) -> str:
        arm = str(value)
        if arm not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'")
        return arm

    @staticmethod
    def _bounded_vector(value: object, limit: float, name: str) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"{name} must contain exactly 3 numbers")
        vector = tuple(float(component) for component in value)
        if any(abs(component) > limit for component in vector):
            raise ValueError(f"each {name} component must be within +/-{limit}")
        return vector  # type: ignore[return-value]

    def _observation_content(self, heading: str) -> list[Json]:
        content: list[Json] = [
            {
                "type": "text",
                "text": f"{heading}\n{json.dumps(dict(self.embodiment.observe()), default=str)}",
            }
        ]
        for frame in self.embodiment.camera_frames():
            content.extend(
                [
                    {"type": "text", "text": f"Fresh camera frame: {frame.label}"},
                    {"type": "image_url", "image_url": {"url": frame.url}},
                ]
            )
        return content

    @staticmethod
    def _assistant_message(turn: Any) -> Json:
        message: Json = {"role": "assistant", "content": turn.content or None}
        if turn.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, separators=(",", ":")),
                    },
                }
                for call in turn.tool_calls
            ]
        return message


class AgenticManipulationBackend:
    """Outer-agent backend exposing a nested manipulation episode as one call."""

    def __init__(self, agent: NestedManipulationAgent) -> None:
        self.agent = agent
        self.last_outcome: ManipulationOutcome | None = None

    def observe(self) -> Mapping[str, Any]:
        state = dict(self.agent.embodiment.observe())
        state["nested_agent"] = (
            None if self.last_outcome is None else self.last_outcome.as_json()
        )
        return state

    def manipulate(self, instruction: str) -> Mapping[str, Any]:
        self.last_outcome = self.agent.run(instruction)
        return self.last_outcome.as_json()


class WbcCartesianManipulationEmbodiment:
    """Adapt a CAP-X/robot-class Cartesian API to the nested-agent contract.

    The wrapped API is expected to expose ``get_current_wrist_pose``,
    ``goto_pose`` and ``set_gripper``. It may use SONIC-compatible upper-body
    references or the masked manipulation WBC underneath; the VLM never owns
    the high-rate stabilization loop.
    """

    def __init__(
        self,
        control_api: object,
        camera_backend: object,
        verifier: Callable[[str], Mapping[str, Any]],
    ) -> None:
        self.control_api = control_api
        self.camera_backend = camera_backend
        self.verifier = verifier

    def observe(self) -> Mapping[str, Any]:
        state: dict[str, Any] = {}
        wrist = getattr(self.control_api, "get_current_wrist_pose")
        for arm in ("left", "right"):
            try:
                position, quaternion = wrist(arm=arm)
                state[f"{arm}_wrist"] = {
                    "position": [float(value) for value in position],
                    "quaternion_wxyz": [float(value) for value in quaternion],
                }
            except (RuntimeError, ValueError, NotImplementedError):
                continue
        get_gripper = getattr(self.control_api, "get_current_gripper_position", None)
        if get_gripper is not None:
            for arm in ("left", "right"):
                try:
                    state[f"{arm}_gripper"] = float(get_gripper(arm=arm))
                except (RuntimeError, ValueError, NotImplementedError):
                    continue
        return state

    def camera_frames(self) -> Sequence["CameraFrame"]:
        return tuple(getattr(self.camera_backend, "capture")())

    def move_end_effector(
        self,
        arm: str,
        translation_m: Sequence[float],
        rotation_rpy_rad: Sequence[float],
        duration_s: float,
    ) -> Mapping[str, Any]:
        position, current = getattr(self.control_api, "get_current_wrist_pose")(arm=arm)
        target_position = [float(position[index]) + float(translation_m[index]) for index in range(3)]
        delta = self._quaternion_from_rpy(*[float(value) for value in rotation_rpy_rad])
        target_quaternion = self._quaternion_multiply(delta, tuple(float(v) for v in current))
        getattr(self.control_api, "goto_pose")(
            position=target_position,
            quaternion_wxyz=target_quaternion,
            arm=arm,
            duration_s=duration_s,
        )
        return {
            "arm": arm,
            "target_position": target_position,
            "target_quaternion_wxyz": list(target_quaternion),
            "duration_s": duration_s,
        }

    def set_gripper(self, arm: str, aperture: float) -> Mapping[str, Any]:
        getattr(self.control_api, "set_gripper")(aperture, arm=arm)
        return {"arm": arm, "aperture": aperture}

    def verify(self, instruction: str) -> Mapping[str, Any]:
        return dict(self.verifier(instruction))

    @staticmethod
    def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
        import math

        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        return (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )

    @staticmethod
    def _quaternion_multiply(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )


class SonicUpperBodyControlApi:
    """Stream collision-gated IK solutions through SONIC while it balances."""

    def __init__(
        self,
        sonic_bridge: object,
        joint_state_provider: Callable[[], Sequence[float]],
        wrist_pose_provider: Callable[[str], tuple[Sequence[float], Sequence[float]]],
        ik_solver: Callable[..., Sequence[float]],
        dex1_sender: Callable[[str, float], None],
        *,
        control_hz: float = 10.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.sonic_bridge = sonic_bridge
        self.joint_state_provider = joint_state_provider
        self.wrist_pose_provider = wrist_pose_provider
        self.ik_solver = ik_solver
        self.dex1_sender = dex1_sender
        self.control_hz = control_hz
        self.sleeper = sleeper
        self._grippers = {"left": 1.0, "right": 1.0}

    def get_current_wrist_pose(self, arm: str) -> tuple[Sequence[float], Sequence[float]]:
        return self.wrist_pose_provider(arm)

    def get_current_gripper_position(self, arm: str) -> float:
        return self._grippers[arm]

    def goto_pose(
        self,
        position: Sequence[float],
        quaternion_wxyz: Sequence[float],
        arm: str,
        duration_s: float = 2.0,
    ) -> None:
        current = tuple(float(value) for value in self.joint_state_provider())
        if len(current) != 29:
            raise ValueError("joint_state_provider must return 29 robot-class G1 joints")
        target = tuple(
            float(value)
            for value in self.ik_solver(
                position=tuple(position),
                quaternion_wxyz=tuple(quaternion_wxyz),
                arm=arm,
                current_joint_positions=current,
            )
        )
        if len(target) != 29:
            raise ValueError("IK solver must return 29 robot-class G1 joints")
        steps = max(1, int(round(float(duration_s) * self.control_hz)))
        dt = float(duration_s) / steps
        velocity = tuple((end - start) / float(duration_s) for start, end in zip(current, target))
        for step in range(1, steps + 1):
            alpha = step / steps
            command = tuple(start + alpha * (end - start) for start, end in zip(current, target))
            getattr(self.sonic_bridge, "command_upper_body")(command, velocity)
            self.sleeper(dt)
        getattr(self.sonic_bridge, "command_upper_body")(target, (0.0,) * 29)

    def set_gripper(self, aperture: float, arm: str) -> None:
        value = float(aperture)
        if not 0.0 <= value <= 1.0:
            raise ValueError("Dex1 aperture must be in [0, 1]")
        self.dex1_sender(arm, value)
        self._grippers[arm] = value
