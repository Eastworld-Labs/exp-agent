from __future__ import annotations

import math
import json
import struct
import time
from collections.abc import Callable
from typing import Any

from .modules.semantic_map import Pose3D
from .navigation_core import VelocityCommand


HEADER_SIZE = 1280
IDLE = 0
SLOW_WALK = 1


def _packed_message(topic: str, fields: list[tuple[str, str, list[int], bytes]]) -> bytes:
    header = {
        "v": 1,
        "endian": "le",
        "count": 1,
        "fields": [
            {"name": name, "dtype": dtype, "shape": shape}
            for name, dtype, shape, _ in fields
        ],
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) > HEADER_SIZE:
        raise ValueError("SONIC header exceeds 1280 bytes")
    return topic.encode("utf-8") + encoded.ljust(HEADER_SIZE, b"\0") + b"".join(
        value for _, _, _, value in fields
    )


def pack_sonic_command(*, start: bool, stop: bool, planner: bool = True) -> bytes:
    """Use robot_class's canonical command protocol implementation."""
    try:
        from robot.sonic_token import pack_sonic_command_message
    except ImportError as exc:
        raise ImportError(
            "SONIC integration requires robot_class; install the 'sonic' extra "
            "or place the robot_class checkout on PYTHONPATH"
        ) from exc
    return pack_sonic_command_message(start=start, stop=stop, planner=planner)


def pack_sonic_planner(
    *,
    mode: int,
    movement: tuple[float, float, float],
    facing: tuple[float, float, float],
    speed: float,
    height: float = -1.0,
) -> bytes:
    return _packed_message(
        "planner",
        [
            ("mode", "i32", [1], struct.pack("<i", mode)),
            ("movement", "f32", [3], struct.pack("<fff", *movement)),
            ("facing", "f32", [3], struct.pack("<fff", *facing)),
            ("speed", "f32", [1], struct.pack("<f", speed)),
            ("height", "f32", [1], struct.pack("<f", height)),
        ],
    )


class SonicZmqBase:
    """Navigation-to-SONIC bridge for the official target-velocity planner.

    The official deployer subscribes to this publisher, turns these bounded
    movement/facing commands into a kinematic reference trajectory, and SONIC
    tracks that reference at the joint level. Localization is supplied by a
    separate callback (SLAM in deployment, simulation truth in a harness).
    """

    name = "sonic-1.1-target-velocity-zmq"

    def __init__(
        self,
        pose_provider: Callable[[], Pose3D],
        *,
        endpoint: str = "tcp://*:5556",
        settle_s: float = 0.5,
        sonic_variant: str = "sonic_v1_1",
    ) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("SonicZmqBase needs pyzmq: pip install -e '.[sonic]'") from exc
        try:
            from robot.sonic_variants import normalize_sonic_variant
        except ImportError as exc:
            raise ImportError(
                "SONIC integration requires robot_class; install the 'sonic' extra "
                "or place the robot_class checkout on PYTHONPATH"
            ) from exc
        self.sonic_variant = normalize_sonic_variant(sonic_variant)
        if self.sonic_variant == "auto":
            raise ValueError("SonicZmqBase requires a resolved SONIC variant")
        self._pose_provider = pose_provider
        self._context = zmq.Context()
        self._publisher = self._context.socket(zmq.PUB)
        self._publisher.bind(endpoint)
        self._yaw_reference = pose_provider().yaw
        self._closed = False
        if settle_s:
            time.sleep(settle_s)
        self._publisher.send(pack_sonic_command(start=True, stop=False, planner=True))

    def pose(self) -> Pose3D:
        return self._pose_provider()

    def command_velocity(self, command: VelocityCommand, dt: float) -> None:
        speed = math.hypot(command.vx, command.vy)
        pose = self.pose()
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        world_vx = c * command.vx - s * command.vy
        world_vy = s * command.vx + c * command.vy
        self._yaw_reference += command.yaw_rate * dt
        facing = (math.cos(self._yaw_reference), math.sin(self._yaw_reference), 0.0)
        if speed > 1e-3:
            movement = (world_vx / speed, world_vy / speed, 0.0)
            mode = SLOW_WALK
        else:
            movement = (0.0, 0.0, 0.0)
            mode = IDLE
        self._publisher.send(
            pack_sonic_planner(
                mode=mode,
                movement=movement,
                facing=facing,
                speed=speed,
            )
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("SONIC bridge is closed")
        self._publisher.send(pack_sonic_command(start=True, stop=False, planner=True))

    def stop(self) -> None:
        if self._closed:
            return
        facing = (math.cos(self._yaw_reference), math.sin(self._yaw_reference), 0.0)
        self._publisher.send(
            pack_sonic_planner(
                mode=IDLE,
                movement=(0.0, 0.0, 0.0),
                facing=facing,
                speed=0.0,
            )
        )

    def close(self, *, stop_control: bool = True) -> None:
        if self._closed:
            return
        self.stop()
        if stop_control:
            self._publisher.send(pack_sonic_command(start=False, stop=True, planner=True))
        self._publisher.close(linger=0)
        self._context.term()
        self._closed = True
