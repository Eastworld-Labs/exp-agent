from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..tools import Tool, object_schema
from ..types import Json


@dataclass(frozen=True)
class CameraFrame:
    label: str
    """Human-readable camera name such as ``head_rgb`` or ``left_wrist``."""

    url: str
    """HTTPS URL or base64 data URL accepted by the selected model endpoint."""


class CameraBackend(Protocol):
    def capture(self) -> Sequence[CameraFrame]: ...


class CameraModule:
    """Adds fresh camera frames to the initial prompt and every post-action turn."""

    name = "cameras"

    def __init__(self, backend: CameraBackend, *, max_looks: int | None = None) -> None:
        self.backend = backend
        self._last_labels: list[str] = []
        # ⚠️ A BUDGET, BECAUSE IMAGES ARE THE ONE THING THAT GROWS A
        # CONVERSATION WITHOUT BOUND. Every look adds a frame to the transcript
        # and every later turn re-sends all of them, so a model that answers
        # uncertainty by looking again turns a mission into an upload. The cap
        # is a refusal the model can read and plan around, not a silent drop.
        self.max_looks = max_looks
        self.looks = 0

    def tools(self) -> Sequence[Tool]:
        return (
            Tool(
                name="observe_surroundings",
                description=(
                    "Capture fresh head and wrist camera frames for the next reasoning turn. "
                    "Use this when local visual context is needed without moving the robot."
                ),
                parameters=object_schema({}),
                handler=self._observe,
                refresh_world=True,
            ),
        )

    def _observe(self, _arguments: object) -> Json:
        if self.max_looks is not None and self.looks >= self.max_looks:
            raise ValueError(
                f"no looks left: {self.max_looks} camera observations is the budget "
                f"for one mission. Decide from what you have already seen, or call "
                f"request_human if you cannot."
            )
        self.looks += 1
        return {"state": "fresh_frames_requested"}

    def snapshot(self) -> Json:
        snapshot: Json = {"last_frame_labels": self._last_labels}
        if self.max_looks is not None:
            snapshot["looks_left"] = max(0, self.max_looks - self.looks)
        return snapshot

    def prompt_content(self) -> list[Json]:
        frames = list(self.backend.capture())
        self._last_labels = [frame.label for frame in frames]
        content: list[Json] = []
        for frame in frames:
            content.extend(
                [
                    {"type": "text", "text": f"Fresh camera frame: {frame.label}"},
                    {"type": "image_url", "image_url": {"url": frame.url}},
                ]
            )
        return content
