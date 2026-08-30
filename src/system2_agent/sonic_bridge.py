from __future__ import annotations

import math
import json
import struct
import time
from collections.abc import Callable
from typing import Any, Sequence

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
    upper_body_position: Sequence[float] | None = None,
    upper_body_velocity: Sequence[float] | None = None,
) -> bytes:
    fields = [
            ("mode", "i32", [1], struct.pack("<i", mode)),
            ("movement", "f32", [3], struct.pack("<fff", *movement)),
            ("facing", "f32", [3], struct.pack("<fff", *facing)),
            ("speed", "f32", [1], struct.pack("<f", speed)),
            ("height", "f32", [1], struct.pack("<f", height)),
    ]
    for name, values in (
        ("upper_body_position", upper_body_position),
        ("upper_body_velocity", upper_body_velocity),
    ):
        if values is None:
            continue
        values = tuple(float(value) for value in values)
        if len(values) != 17:
            raise ValueError(f"{name} must contain SONIC's 17 upper-body joints")
        fields.append((name, "f32", [17], struct.pack("<17f", *values)))
    return _packed_message("planner", fields)


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
        planner_update_s: float = 1.0,
        max_facing_rate: float = 0.3,
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
        self._planner_update_s = planner_update_s
        self._max_facing_rate = max_facing_rate
        self._last_planner_update = 0.0
        self._latched_planner: tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            float,
        ] | None = None
        self._upper_body_position: tuple[float, ...] | None = None
        self._upper_body_velocity: tuple[float, ...] | None = None
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
        if command.facing_yaw is not None:
            # SONIC's planner accepts an absolute facing vector, not angular
            # velocity. Slew that vector instead of teleporting it across a
            # large heading change in one replan.
            requested_rate = abs(command.yaw_rate)
            rate = min(
                self._max_facing_rate,
                requested_rate if requested_rate > 1e-4 else self._max_facing_rate,
            )
            self._yaw_reference = slew_heading(
                self._yaw_reference, command.facing_yaw, rate, dt
            )
        else:
            self._yaw_reference += max(
                -self._max_facing_rate * dt,
                min(self._max_facing_rate * dt, command.yaw_rate * dt),
            )
        facing = (math.cos(self._yaw_reference), math.sin(self._yaw_reference), 0.0)
        if command.facing_yaw is not None and speed > 1e-3:
            # Movement and facing are independent SONIC planner inputs. After
            # rotate-before-walk has reduced the error, movement follows the
            # path bearing while facing finishes its bounded slew.
            world_vx = speed * math.cos(command.facing_yaw)
            world_vy = speed * math.sin(command.facing_yaw)
        else:
            world_vx = c * command.vx - s * command.vy
            world_vy = s * command.vx + c * command.vy
        if speed > 1e-3:
            movement = (world_vx / speed, world_vy / speed, 0.0)
            mode = SLOW_WALK
        elif abs(command.yaw_rate) > 1e-3:
            # Keep one locomotion mode while turning. Alternating IDLE/WALK at
            # a heading threshold forces discontinuous planner transitions.
            movement = (0.0, 0.0, 0.0)
            mode = SLOW_WALK
        else:
            movement = (0.0, 0.0, 0.0)
            mode = IDLE
        proposed = (mode, movement, facing, speed)
        now = time.monotonic()
        # ZMQ messages must continue at >1 Hz for the deployer's watchdog, but
        # changing any float makes the kinematic planner generate a fresh
        # trajectory. Latch values for its intended one-second replan interval.
        if (
            self._latched_planner is None
            or self._latched_planner[0] != mode
            or now - self._last_planner_update >= self._planner_update_s
        ):
            self._latched_planner = proposed
            self._last_planner_update = now
        latched_mode, latched_movement, latched_facing, latched_speed = self._latched_planner
        self._publisher.send(
            pack_sonic_planner(
                mode=latched_mode,
                movement=latched_movement,
                facing=latched_facing,
                speed=latched_speed,
                upper_body_position=self._upper_body_position,
                upper_body_velocity=self._upper_body_velocity,
            )
        )

    def command_upper_body(
        self,
        robot_class_joint_positions: Sequence[float],
        robot_class_joint_velocities: Sequence[float] | None = None,
    ) -> None:
        """Hold balance while applying a robot-class-ordered upper-body target.

        robot_class uses MuJoCo order (waist, left arm, right arm). SONIC's
        planner wire format expects its interleaved IsaacLab order.
        """
        positions = tuple(float(value) for value in robot_class_joint_positions)
        if len(positions) != 29:
            raise ValueError("robot_class_joint_positions must contain all 29 G1 joints")
        order = (12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28)
        self._upper_body_position = tuple(positions[index] for index in order)
        if robot_class_joint_velocities is None:
            self._upper_body_velocity = (0.0,) * 17
        else:
            velocities = tuple(float(value) for value in robot_class_joint_velocities)
            if len(velocities) != 29:
                raise ValueError("robot_class_joint_velocities must contain all 29 G1 joints")
            self._upper_body_velocity = tuple(velocities[index] for index in order)
        self._latched_planner = None
        self.command_velocity(VelocityCommand(0.0, 0.0, 0.0), 0.05)

    def release_upper_body(self) -> None:
        """Return arm-reference ownership to SONIC's generated motion."""
        self._upper_body_position = None
        self._upper_body_velocity = None
        self._latched_planner = None

    def start(self, *, planner: bool = True) -> None:
        if self._closed:
            raise RuntimeError("SONIC bridge is closed")
        self._publisher.send(pack_sonic_command(start=True, stop=False, planner=planner))

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
                upper_body_position=self._upper_body_position,
                upper_body_velocity=self._upper_body_velocity,
            )
        )
        self._latched_planner = None

    def close(self, *, stop_control: bool = True) -> None:
        if self._closed:
            return
        self.stop()
        if stop_control:
            self._publisher.send(pack_sonic_command(start=False, stop=True, planner=True))
        self._publisher.close(linger=0)
        self._context.term()
        self._closed = True


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def slew_heading(current: float, target: float, max_rate: float, dt: float) -> float:
    """Move an angle toward a target through the shortest bounded arc."""
    error = _wrap(target - current)
    step = max(0.0, max_rate) * max(0.0, dt)
    return _wrap(current + max(-step, min(step, error)))
