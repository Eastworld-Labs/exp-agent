"""The bytes on the fleet link: ROS message shapes and topic names.

PURE. No sockets, no MQTT, no clock -- so the whole encoding is testable, and
the one thing that must never drift (what this service is allowed to publish)
is a constant rather than a habit.
"""
from __future__ import annotations

import json
import math
from typing import Any

from ..types import Json

#: MQTT topic layout, from src/g1_fleet/config/topics.json. Mechanical, so it
#: is derived rather than read from the manifest -- the manifest carries WHICH
#: topics and their QoS, not the shape.
def uplink(robot_id: str, ros_topic: str) -> str:
    return f"g1/{robot_id}/ros{ros_topic}"


def downlink(robot_id: str, ros_topic: str) -> str:
    return f"g1/{robot_id}/cmd{ros_topic}"


# ##### THE COMPLETE LIST OF WHAT THIS SERVICE MAY PUBLISH TO THE ROBOT. #####
#
# ⚠️ `/cmd_vel` AND `/estop` ARE ABSENT BY CONSTRUCTION, NOT BY PROMPT.
# `/cmd_vel` bypasses every protective thing in the stack -- the costmaps, the
# collision monitor's stop zones, and the backend's own stale-scan watchdog --
# so a model with access to it has access to none of the safety. `/estop` is
# the operator's latch, and releasing it is an operator's act. Both are in the
# downlink manifest and reachable; keeping them out of the tool set is the
# access control. A prompt is not one.
#
# The value is the MQTT v5 message-expiry in seconds: the line past which a
# command is better dropped by the broker than obeyed by the robot. A goal that
# took three seconds to cross is fine; one that took thirty is not a goal.
PUBLISHABLE: dict[str, int] = {
    "/goal_pose": 3,
}

#: Read-only. Everything the backend needs to refuse a goal it should not send,
#: and to say what became of one it did.
#: ⚠️ THESE ARE THE REAL ROBOT'S TOPICS. A target with a different pose
#: topic (the sim's /odom) subscribes its own in Nav2MqttBackend; the link
#: itself is per robot id, so nothing here leaks across targets.
SUBSCRIBED = (
    "/localization_3d",   # PoseStamped, retained -- where the robot says it is
    "/estop_state",       # Bool, retained -- the latch AS THE ROBOT REPORTS IT
    "/goal_status",       # String JSON -- Nav2's own verdict (g1_bridge)
    "/sonic/enabled",     # Bool -- armed, when SONIC is the locomotion backend
)


def quat_from_yaw(yaw_rad: float) -> Json:
    """Yaw about z -> a ROS quaternion."""
    half = float(yaw_rad) / 2.0
    return {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)}


def yaw_from_quat(quat: Any) -> float:
    """A ROS quaternion -> yaw about z, radians. Tolerant of missing fields."""
    q = quat or {}
    x = float(q.get("x") or 0.0)
    y = float(q.get("y") or 0.0)
    z = float(q.get("z") or 0.0)
    w = float(q.get("w") or 1.0)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pose_stamped(x: float, y: float, yaw_rad: float, frame: str = "map") -> Json:
    """A geometry_msgs/PoseStamped, as the fleet agent's CBOR decoder wants it.

    ⚠️ `stamp` IS ZERO, MEANING "use the latest transform". This pose is
    AUTHORED now, not measured at some earlier instant, and a host-clock stamp
    would be wrong by whatever the host/robot offset is -- which on this stack
    is real (the localizer re-stamps everything for exactly that reason). The
    dashboard's own goal publisher does the same.
    """
    return {
        "header": {"frame_id": frame, "stamp": {"sec": 0, "nanosec": 0}},
        "pose": {
            "position": {"x": round(float(x), 3), "y": round(float(y), 3), "z": 0.0},
            "orientation": quat_from_yaw(yaw_rad),
        },
    }


def decode_pose(msg: Any) -> Json | None:
    """A map-frame pose off the wire -> {x, y, yaw} in radians, or None.

    Accepts BOTH shapes a pose arrives in on this stack:

        geometry_msgs/PoseStamped   {"header", "pose": {"position", "orientation"}}
        nav_msgs/Odometry           {"header", "pose": {"pose": {...}, "covariance"}}

    The real robot's localizer publishes the first (/localization_3d); the
    simulator publishes ground truth as the second (/odom), in a frame that is
    the map frame by construction. One decoder, so a target is a topic name
    and nothing else has to know which robot it is looking at.
    """
    try:
        pose = (msg or {})["pose"]
        if isinstance(pose, dict) and "position" not in pose and "pose" in pose:
            pose = pose["pose"]          # nav_msgs/Odometry: pose.pose
        position = pose["position"]
        return {
            "x": float(position["x"]),
            "y": float(position["y"]),
            "yaw": yaw_from_quat(pose.get("orientation")),
            "frame": str((msg.get("header") or {}).get("frame_id") or "map"),
        }
    except (KeyError, TypeError, ValueError):
        return None


#: The original name, kept for callers that only ever see a PoseStamped.
decode_pose_stamped = decode_pose


def decode_bool(msg: Any) -> bool | None:
    """A std_msgs/Bool off the wire. `None` means the field was not there --
    which is NOT the same as False, and must not collapse into it: "nobody is
    saying whether the E-stop is engaged" is a different fact from "it is not"."""
    if not isinstance(msg, dict) or "data" not in msg:
        return None
    return bool(msg["data"])


def decode_goal_status(msg: Any) -> Json | None:
    """A /goal_status String off the wire -> the parsed verdict, or None.

    Tolerant: a consumer that hard-failed on an unfamiliar field would be a
    consumer that stops reporting arrivals the first time the format grows one.
    The schema lives in g1_bridge/goal_status.py.
    """
    try:
        parsed = json.loads((msg or {})["data"])
    except (KeyError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


_JPEG_HINTS = ("jpeg", "jpg")


def image_mime(compressed_format: Any) -> str | None:
    """sensor_msgs/CompressedImage `format` -> a data-URL mime type.

    The field is free text ("bgr8; jpeg compressed bgr8"), so this reads it
    rather than assuming. An unrecognised encoding returns None and the frame is
    skipped: a mislabelled data URL reaches the model as a broken image and it
    reasons about the absence rather than being told there was a problem.
    """
    text = str(compressed_format or "").lower()
    if any(hint in text for hint in _JPEG_HINTS):
        return "image/jpeg"
    if "png" in text:
        return "image/png"
    return None


def int8_values(data: Any) -> list[int] | None:
    """A ROS `int8[]` off the CBOR wire -> signed ints, or None if unreadable.

    ⚠️ int8 AND uint8 ARRAYS CROSS AS A PLAIN CBOR BYTE STRING, UNTAGGED.
    g1_fleet's cbor_codec tags everything wider (float32[] and friends get an
    RFC 8746 tag) but bytes are bytes, so an `int8[]` arrives as Python `bytes`
    with the sign already thrown away: a cost of -1 (unknown) reads as 255 and a
    lethal 100 reads as 100. Anything above 127 is the negative it was. The
    dashboard's own costmap reader does exactly this (webui/src/ros/costmap.ts).
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return [value - 256 if value > 127 else value for value in bytes(data)]
    if isinstance(data, (list, tuple)):
        try:
            return [int(value) for value in data]
        except (TypeError, ValueError):
            return None
    return None


def decode_occupancy_grid(msg: Any) -> Json | None:
    """A nav_msgs/OccupancyGrid -> {frame, width, height, resolution, origin_*, cost}.

    Returns None rather than raising on anything malformed: this is a picture of
    the world arriving several times a second, and a consumer that died on one
    bad frame would be a consumer that stops seeing obstacles.

    ⚠️ A ROTATED ORIGIN IS REFUSED (None), not silently ignored. Nav2's rolling
    local costmap is axis-aligned, so a quaternion that is not identity means
    this is some other grid -- and reading it as axis-aligned would place every
    obstacle somewhere plausible and wrong.
    """
    try:
        info = (msg or {})["info"]
        width = int(info["width"])
        height = int(info["height"])
        resolution = float(info["resolution"])
        origin = info["origin"]
        position = origin["position"]
        cost = int8_values(msg.get("data"))
    except (KeyError, TypeError, ValueError):
        return None
    if cost is None or width <= 0 or height <= 0 or resolution <= 0:
        return None
    if len(cost) != width * height:
        return None
    if abs(yaw_from_quat(origin.get("orientation"))) > 1e-3:
        return None
    return {
        "frame": str((msg.get("header") or {}).get("frame_id") or ""),
        "width": width,
        "height": height,
        "resolution": resolution,
        "origin_x": float(position["x"]),
        "origin_y": float(position["y"]),
        "cost": cost,
    }


def decode_grid_update(msg: Any) -> Json | None:
    """A map_msgs/OccupancyGridUpdate patch -> {x, y, width, height, cost}."""
    try:
        x = int((msg or {})["x"])
        y = int(msg["y"])
        width = int(msg["width"])
        height = int(msg["height"])
        cost = int8_values(msg.get("data"))
    except (KeyError, TypeError, ValueError):
        return None
    if cost is None or width <= 0 or height <= 0 or len(cost) != width * height:
        return None
    return {"x": x, "y": y, "width": width, "height": height, "cost": cost}


#: What a depth frame's intrinsics message must carry to be usable.
_DEPTH_INFO_KEYS = ("width", "height", "fx", "fy", "cx", "cy")


def decode_depth_info(msg: Any) -> Json | None:
    """The retained `/g1/head/depth_info` JSON -> the camera model, or None.

    ⚠️ THE INTRINSICS COME OVER THE WIRE RATHER THAN OFF A DATASHEET, and they
    describe THE SIZE THAT WAS PUBLISHED, not the sensor's native one. The
    publisher downsamples before sending, and it is the only party that knows
    what fx became. A field of view assumed here instead would be wrong by
    whatever crop and resize the driver applied, in a way nothing reports.
    """
    parsed = decode_goal_status(msg)
    if parsed is None:
        return None
    try:
        info = {key: float(parsed[key]) for key in _DEPTH_INFO_KEYS}
    except (KeyError, TypeError, ValueError):
        return None
    info["width"] = int(info["width"])
    info["height"] = int(info["height"])
    if info["width"] <= 0 or info["height"] <= 0 or info["fx"] <= 0 or info["fy"] <= 0:
        return None
    info["depth_scale"] = float(parsed.get("depth_scale") or 0.001)
    info["frame_id"] = str(parsed.get("frame_id") or "")
    info["camera"] = str(parsed.get("camera") or "")
    info["source"] = str(parsed.get("source") or "")
    return info
