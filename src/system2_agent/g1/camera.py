"""Head-camera frames off the fleet link, as data URLs for the model."""
from __future__ import annotations

import base64
from typing import Sequence

from ..modules.camera import CameraFrame
from .link import Link
from .wire import image_mime

#: What the dashboard already publishes and the model can actually reason about.
#: ⚠️ COLOUR FIRST. Model attention follows order, and the range image is a false
#: colour map with its legend burned into its own pixels -- supporting evidence,
#: not the subject.
#:
#: ⚠️ THE `d435c` IN THESE NAMES IS A HISTORICAL LABEL, NOT A CLAIM ABOUT WHICH
#: CAMERA TOOK THE PICTURE. With `D455=1` on the robot the vision node runs on
#: the LEVEL D455 and publishes its frames here, because the dashboard, the
#: simulator, the fleet manifests and this file all agree on these two strings.
#: Renaming them is a change across two repositories and a running dashboard.
#: The camera that actually produced a frame is named in `/g1/head/depth_info`,
#: which is also where the metric depth for `local_planner` comes from -- see
#: g1/depth.py, and note that THOSE topics are deliberately camera-neutral.
DEFAULT_CAMERAS = (
    ("head_colour", "/g1/d435c/preview/compressed"),
    ("head_range", "/g1/d435c/preview_depth/compressed"),
)


class MqttHeadCamera:
    """A CameraBackend reading compressed frames the fleet agent forwards.

    ⚠️ FRESH FRAMES ONLY. A stale JPEG shown to the model is worse than no
    picture at all: it reasons confidently about a scene the robot left minutes
    ago. An old frame is dropped rather than captioned as old, because the
    caption is exactly the thing a model skims past.
    """

    def __init__(
        self,
        link: Link,
        *,
        cameras: Sequence[tuple[str, str]] = DEFAULT_CAMERAS,
        stale_s: float = 3.0,
        now=None,
    ) -> None:
        import time

        self.link = link
        self.cameras = tuple(cameras)
        self.stale_s = stale_s
        self._now = now or time.monotonic
        for _, topic in self.cameras:
            link.subscribe(topic)

    def capture(self) -> list[CameraFrame]:
        frames: list[CameraFrame] = []
        for label, topic in self.cameras:
            entry = self.link.latest(topic)
            if entry is None:
                continue
            msg, arrived = entry
            if self._now() - arrived > self.stale_s:
                continue
            mime = image_mime(msg.get("format"))
            data = msg.get("data")
            if mime is None or not isinstance(data, (bytes, bytearray)):
                continue
            encoded = base64.b64encode(bytes(data)).decode("ascii")
            frames.append(CameraFrame(label=label, url=f"data:{mime};base64,{encoded}"))
        return frames
