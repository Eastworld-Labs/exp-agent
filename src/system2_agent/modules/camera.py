from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..tools import Tool
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

    def __init__(self, backend: CameraBackend) -> None:
        self.backend = backend
        self._last_labels: list[str] = []

    def tools(self) -> Sequence[Tool]:
        return ()

    def snapshot(self) -> Json:
        return {"last_frame_labels": self._last_labels}

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
