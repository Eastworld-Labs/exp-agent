"""Find ONE named thing in ONE camera frame, with a nested model call.

##### THIS MOVES NOTHING. It turns a word into a box, or refuses. #####

`local_planner("sink")` needs a pixel before it can need a distance. The
mission model already has the picture in its transcript, but asking IT for
coordinates mid-mission costs a whole reasoning turn at mission effort and
buries the answer in prose. So grounding is a separate, cheap, one-image call
with one tool and one job -- the same shape as the nested manipulation agent
(`manipulation_agent.NestedManipulationAgent`), for the same reason: a bounded
sub-episode the outer loop sees as one result.

⚠️ A REFUSAL IS THE COMMON CASE AND MUST STAY CHEAP TO PRODUCE. The camera is
pointed wherever the robot happens to face; most missions will name something
that is not in frame. `found: false` with a note is a first-class answer, not
an error path, and the outer tool turns it into an `ok: false` the mission model
reads as evidence. What must never happen is a guessed box: a confident
rectangle over a blank wall becomes a metric range, a standoff pose and a walk.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .agent import assistant_message, turn_fault
from .model import ChatModel
from .modules.camera import CameraFrame
from .tools import Tool, object_schema
from .types import Json


GROUNDING_SYSTEM_PROMPT = """You locate ONE named object in ONE photograph taken by a \
robot's forward camera. You do nothing else.

Reply by calling report_grounding exactly once.

- Coordinates are FRACTIONS of the image: x0/x1 are fractions of the width from the \
LEFT edge, y0/y1 fractions of the height from the TOP edge, each between 0 and 1, with \
x0 < x1 and y0 < y1. Box the whole visible extent of the object, tightly.
- If several things match, box the one nearest the camera (largest, lowest in frame).
- confidence is your own probability, 0 to 1, that the box contains the named object.
- ##### IF IT IS NOT CLEARLY VISIBLE, SAY found=false. ##### A guessed box becomes a \
distance and then a robot walking at it. "I can see a counter but no sink", "the image \
is too dark", "only part of it is in frame at the left edge" are all useful answers. \
Never box a thing you are inferring from context rather than seeing."""


@dataclass(frozen=True)
class Box:
    """A bounding box in normalised image coordinates, origin top-left.

    Normalised rather than pixels ON PURPOSE: the model sees a JPEG preview
    (480 px wide) while the range comes from a depth image (320 px wide) and the
    intrinsics describe a third size. Fractions are the one description all
    three agree on, so nothing downstream has to know which resize happened.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "x1", "y1"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value != value:
                raise ValueError(f"box {name} must be a number")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"box {name}={value} is outside the image (0..1)")
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError(
                f"box is inverted or empty: x0={self.x0} x1={self.x1} "
                f"y0={self.y0} y1={self.y1}"
            )

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def shrunk(self, keep: float) -> "Box":
        """Shrink toward the centre, keeping `keep` of each side length.

        ⚠️ THE BORDER OF A BOX IS MOSTLY NOT THE OBJECT. A tight box still
        contains silhouette and background at its edges, and those pixels are at
        the range of whatever is BEHIND the target -- which is exactly the
        reading that would pull an approach too far. g1_vision's target_point.py
        shrinks to 0.6 for this reason and this mirrors it.
        """
        if keep >= 1.0:
            return self
        keep = max(0.05, float(keep))
        cx, cy = self.centre
        half_w = (self.x1 - self.x0) * keep / 2.0
        half_h = (self.y1 - self.y0) * keep / 2.0
        return Box(
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(1.0, cx + half_w),
            min(1.0, cy + half_h),
        )

    def as_json(self) -> list[float]:
        return [round(self.x0, 4), round(self.y0, 4), round(self.x1, 4), round(self.y1, 4)]


@dataclass(frozen=True)
class Grounding:
    found: bool
    box: Box | None
    confidence: float
    label: str
    note: str
    model_calls: int
    usage: Json | None = None

    def as_json(self) -> Json:
        return {
            "found": self.found,
            "box": None if self.box is None else self.box.as_json(),
            "confidence": round(float(self.confidence), 3),
            "label": self.label,
            "note": self.note,
            "model_calls": self.model_calls,
            "usage": self.usage,
        }


class VisionGrounder:
    """Ask a vision model for one box, and refuse anything less than one."""

    def __init__(
        self,
        model: ChatModel,
        *,
        min_confidence: float = 0.5,
        min_box_area: float = 0.002,
        max_model_calls: int = 2,
        system_prompt: str = GROUNDING_SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.min_confidence = min_confidence
        # 0.2% of the frame. Below this there are too few depth pixels inside
        # the shrunk box to range it, so the refusal belongs here where it can
        # say "too small" rather than downstream as "no depth return".
        self.min_box_area = min_box_area
        self.max_model_calls = max_model_calls
        self.system_prompt = system_prompt

    # ----------------------------------------------------------------- API --
    def ground(self, target: str, frame: CameraFrame) -> Grounding:
        """A box for `target` in `frame`, or ValueError saying why not."""
        tool = self._report_tool()
        messages: list[Json] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Target: {target}\n"
                            f"Camera: {frame.label}\n"
                            "Report where it is, or that you cannot see it."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": frame.url}},
                ],
            },
        ]
        usage: Json | None = None
        last_fault = ""
        for attempt in range(1, self.max_model_calls + 1):
            turn = self.model.complete(messages, [tool.schema()])
            usage = _merge_usage(usage, turn.usage)
            fault = turn_fault(turn)
            if fault is None and turn.tool_calls[0].name != tool.name:
                fault = f"call {tool.name}, not {turn.tool_calls[0].name}"
            if fault is None:
                arguments: Any = turn.tool_calls[0].arguments
            else:
                # No usable tool call. A model that answered in prose still
                # answered; parse it before spending a second call on it.
                arguments = _json_object(turn.content)
                if arguments is None:
                    last_fault = fault
                    messages.append(assistant_message(turn))
                    messages.append({"role": "user", "content": fault})
                    continue
            result = tool.run(arguments)
            if not result.ok:
                last_fault = str(result.error)
                messages.append(assistant_message(turn))
                messages.append(
                    {"role": "user", "content": f"that was not usable: {last_fault}"}
                )
                continue
            return self._verdict(target, dict(result.data), attempt, usage)
        raise ValueError(
            f"the grounding model did not report a usable box for {target!r} in "
            f"{self.max_model_calls} attempts: {last_fault}"
        )

    # ------------------------------------------------------------ internal --
    def _verdict(self, target: str, data: Json, calls: int, usage: Json | None) -> Grounding:
        note = str(data.get("note") or "").strip()
        label = str(data.get("label") or target).strip()
        confidence = float(data.get("confidence") or 0.0)
        if not data.get("found"):
            raise ValueError(
                f"{target!r} is not visible in the current camera frame"
                + (f": {note}" if note else "")
                + ". Turn or walk to a labelled place that faces it, then look again."
            )
        raw = data.get("box")
        if not isinstance(raw, dict):
            raise ValueError(
                f"the grounding model said it found {target!r} but reported no box"
            )
        box = Box(
            float(raw.get("x0", 0.0)),
            float(raw.get("y0", 0.0)),
            float(raw.get("x1", 0.0)),
            float(raw.get("y1", 0.0)),
        )
        if not 0.0 <= confidence <= 1.0:
            # ⚠️ OUT OF CONTRACT, AND THE COORDINATES USE THE SAME SCALE. A model
            # that answered "5" to a field documented as 0 to 1 has misread the
            # convention, and the box it drew is described in fractions of the
            # image by exactly the same instruction. Refusing costs one retry;
            # trusting it costs a walk at whatever it actually meant.
            raise ValueError(
                f"the grounding model reported confidence {confidence} for {target!r}, "
                "which is outside 0 to 1. Its coordinates are on that same scale, so "
                "the box cannot be trusted either."
            )
        if confidence < self.min_confidence:
            raise ValueError(
                f"grounding confidence {confidence:.2f} for {target!r} is below the "
                f"{self.min_confidence:.2f} this tool acts on"
                + (f": {note}" if note else "")
                + ". Get closer or look from somewhere the object is unambiguous."
            )
        if box.area < self.min_box_area:
            raise ValueError(
                f"the box for {target!r} covers {box.area * 100:.2f}% of the frame, "
                "too small to measure a distance across. Walk closer with navigate_to "
                "and look again."
            )
        return Grounding(
            found=True,
            box=box,
            confidence=confidence,
            label=label,
            note=note,
            model_calls=calls,
            usage=usage,
        )

    def _report_tool(self) -> Tool:
        # ⚠️ A REAL `Tool`, NOT AN AD-HOC dict, so the arguments are validated by
        # the same `tools._validate` every physical tool goes through -- and so
        # the text fallback below is checked exactly as strictly as a tool call.
        return Tool(
            name="report_grounding",
            description="Report where the named object is in the image, or that it is not visible.",
            parameters=object_schema(
                {
                    "found": {
                        "type": "boolean",
                        "description": "True only if you can actually see the named object.",
                    },
                    "box": {
                        "type": "object",
                        "description": (
                            "Bounding box as fractions of the image: x from the left "
                            "edge, y from the top. Omit when found is false."
                        ),
                        "properties": {
                            "x0": {"type": "number"},
                            "y0": {"type": "number"},
                            "x1": {"type": "number"},
                            "y1": {"type": "number"},
                        },
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0 to 1: your probability that the box holds the object.",
                    },
                    "label": {
                        "type": "string",
                        "description": "What you actually see there, in a word or two.",
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "One line. When found is false, why not -- that sentence is "
                            "what the mission controller is told."
                        ),
                    },
                },
                ["found", "confidence", "note"],
            ),
            handler=lambda arguments: dict(arguments),
        )


def _json_object(text: str) -> Json | None:
    """The first JSON object in a prose reply, or None.

    Providers differ on whether `tool_choice: "auto"` ever forces a call, and
    `model.OpenAICompatibleModel` deliberately does not send `"required"`. A
    model that answered with the right fields in a fenced block has answered;
    throwing that away to spend a second call is waste, not rigour.
    """
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _merge_usage(total: Json | None, turn: Json | None) -> Json | None:
    """Add a provider usage block. `None` stays `None`: nobody said is not zero."""
    if not turn:
        return total
    merged: Json = dict(total or {})
    for key, value in turn.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            merged[key] = merged.get(key, 0) + value
        elif key not in merged:
            merged[key] = value
    return merged


__all__ = ["Box", "Grounding", "VisionGrounder", "GROUNDING_SYSTEM_PROMPT"]
