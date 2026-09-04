"""Metric depth off the fleet link: millimetres, and the lens that made them.

##### READ-ONLY. #####

The two preview topics the model already sees are PICTURES -- the range one is a
Turbo colour map with its legend burned into its own pixels, which is evidence a
reader can weigh and nothing a computer can measure. `local_planner` needs a
number, so the robot publishes two more topics:

    /g1/head/depth/compressed   sensor_msgs/CompressedImage, "16UC1; png"
                                16-bit PNG, MILLIMETRES, 0 = no return,
                                downsampled, in the COLOUR camera's pixel grid
    /g1/head/depth_info         std_msgs/String, JSON, RETAINED
                                {width,height,fx,fy,cx,cy,depth_scale,frame_id,...}

⚠️ THE INTRINSICS ARE A SEPARATE, RETAINED TOPIC AND BOTH ARE REQUIRED. A depth
image without the lens that produced it is not measurable: the publisher
downsamples before sending, so fx is not the sensor's fx, and the only party who
knows what it became is the party that resized it. A frame that arrives before
the info is dropped rather than ranged against an assumption -- the info is
retained, so that state lasts about one message.

⚠️ ALIGNED DEPTH, IN THE COLOUR GRID. The whole scheme rests on "the box is at
(u, v) in the colour frame, so the range is depth[v][u]", which is only true
because the driver resamples depth into the colour camera's pixel grid
(`align_depth.enable:=true`). Raw depth is a different imager, a different
principal point and a baseline away, and indexing it with a colour-image box is
wrong by a parallax that GROWS as the object gets closer -- which is exactly the
regime this tool works in, with nothing on screen to show it.
"""
from __future__ import annotations

import time
from typing import Callable

from ..local_planner import DepthImage
from ..png16 import decode_png16
from ..types import Json
from .link import Link
from .wire import decode_depth_info, image_mime

DEPTH_TOPIC = "/g1/head/depth/compressed"
DEPTH_INFO_TOPIC = "/g1/head/depth_info"


class MqttHeadDepth:
    """A `DepthSource` over the fleet link. Returns None rather than raising.

    Depth is an upgrade, not a precondition: without it `LocalPlanner` ranges on
    the costmap instead and says so in the result. So every failure here is a
    `None` plus a sentence in `status()`, never an exception that would end a
    mission over a camera the robot may simply not have.
    """

    def __init__(
        self,
        link: Link,
        *,
        depth_topic: str = DEPTH_TOPIC,
        info_topic: str = DEPTH_INFO_TOPIC,
        stale_s: float = 3.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.link = link
        self.depth_topic = depth_topic
        self.info_topic = info_topic
        # Tighter than the costmap's gate: this is a camera at ~3 Hz, so silence
        # means the camera stopped rather than the scene being quiet.
        self.stale_s = stale_s
        self._now = now
        self._reason = "no depth frame has arrived yet"
        link.subscribe(depth_topic)
        link.subscribe(info_topic)

    # ------------------------------------------------------------ DepthSource --
    def capture(self) -> DepthImage | None:
        try:
            return self._capture()
        except Exception as exc:  # noqa: BLE001
            self._reason = f"{type(exc).__name__}: {exc}"
            return None

    def status(self) -> Json:
        state: Json = {
            "topic": self.depth_topic,
            "info_topic": self.info_topic,
            "available": False,
        }
        try:
            frame = self.link.latest(self.depth_topic)
            info = self.link.latest(self.info_topic)
            if frame is not None:
                state["age_s"] = round(self._now() - frame[1], 1)
            if info is not None:
                state["info_age_s"] = round(self._now() - info[1], 1)
                parsed = decode_depth_info(info[0])
                if parsed is not None:
                    state["size"] = [parsed["width"], parsed["height"]]
                    state["camera"] = parsed["camera"]
                    state["source"] = parsed["source"]
            state["available"] = self._capture() is not None
        except Exception as exc:  # noqa: BLE001
            self._reason = f"{type(exc).__name__}: {exc}"
        if not state["available"] and self._reason:
            state["reason"] = self._reason
        return state

    # -------------------------------------------------------------- internal --
    def _capture(self) -> DepthImage | None:
        entry = self.link.latest(self.depth_topic)
        if entry is None:
            self._reason = f"nothing has arrived on {self.depth_topic}"
            return None
        msg, arrived = entry
        age = self._now() - arrived
        if age > self.stale_s:
            self._reason = f"the last depth frame arrived {age:.0f} s ago"
            return None

        info_entry = self.link.latest(self.info_topic)
        if info_entry is None:
            self._reason = (
                f"depth frames are arriving but {self.info_topic} is not; without the "
                "intrinsics the pixels cannot be turned into metres"
            )
            return None
        info = decode_depth_info(info_entry[0])
        if info is None:
            self._reason = f"the message on {self.info_topic} could not be read"
            return None

        mime = image_mime(msg.get("format"))
        data = msg.get("data")
        if mime != "image/png" or not isinstance(data, (bytes, bytearray)):
            self._reason = (
                f"depth frames must be PNG-encoded 16-bit; this one says "
                f"{msg.get('format')!r}"
            )
            return None

        pixels = decode_png16(bytes(data))
        if pixels is None:
            self._reason = "the depth PNG could not be decoded"
            return None
        values, width, height = pixels
        if (width, height) != (info["width"], info["height"]):
            # The info is retained and the frames are not, so a resolution change
            # shows up here first. Ranging the new pixels through the old lens
            # would be wrong by the ratio, silently.
            self._reason = (
                f"the depth frame is {width}x{height} but {self.info_topic} describes "
                f"{info['width']}x{info['height']}; waiting for them to agree"
            )
            return None
        self._reason = ""
        return DepthImage(
            width=width,
            height=height,
            fx=info["fx"],
            fy=info["fy"],
            cx=info["cx"],
            cy=info["cy"],
            depth_mm=values,
            scale=info["depth_scale"],
            frame_id=info["frame_id"],
            source=info["source"] or info["camera"],
            age_s=age,
            info_age_s=self._now() - info_entry[1],
        )
