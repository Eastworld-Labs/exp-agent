from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np


SONIC_DOF = 29
STATE_FLOATS = 1 + 7 + 6 + 3 + 4 + 6 + 4 * SONIC_DOF
COMMAND_FLOATS = 5 * SONIC_DOF
_STATE = struct.Struct(f"<{STATE_FLOATS}f")
_COMMAND = struct.Struct(f"<{COMMAND_FLOATS}f")

# Selection vectors: ``mujoco[MJ_TO_ISAACLAB]`` produces IsaacLab order and
# ``isaaclab[ISAACLAB_TO_MUJOCO]`` produces SONIC/Unitree (MuJoCo) order.
MUJOCO_TO_ISAACLAB = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
ISAACLAB_TO_MUJOCO = np.argsort(MUJOCO_TO_ISAACLAB)


@dataclass(frozen=True)
class SonicState:
    time: float
    root_pose_wxyz: np.ndarray
    root_velocity: np.ndarray
    root_acceleration: np.ndarray
    torso_quaternion_wxyz: np.ndarray
    torso_velocity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_acceleration: np.ndarray
    joint_torque: np.ndarray


@dataclass(frozen=True)
class SonicMotorCommand:
    position: np.ndarray
    velocity: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    feedforward_torque: np.ndarray


def _array(values: Sequence[float], size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).reshape(-1)
    if result.size != size:
        raise ValueError(f"{label} must contain {size} values, got {result.size}")
    return result


def pack_state(state: SonicState) -> bytes:
    values = np.concatenate(
        (
            np.asarray([state.time], dtype=np.float32),
            _array(state.root_pose_wxyz, 7, "root_pose_wxyz"),
            _array(state.root_velocity, 6, "root_velocity"),
            _array(state.root_acceleration, 3, "root_acceleration"),
            _array(state.torso_quaternion_wxyz, 4, "torso_quaternion_wxyz"),
            _array(state.torso_velocity, 6, "torso_velocity"),
            _array(state.joint_position, SONIC_DOF, "joint_position"),
            _array(state.joint_velocity, SONIC_DOF, "joint_velocity"),
            _array(state.joint_acceleration, SONIC_DOF, "joint_acceleration"),
            _array(state.joint_torque, SONIC_DOF, "joint_torque"),
        )
    )
    return _STATE.pack(*values)


def unpack_state(packet: bytes) -> SonicState:
    if len(packet) != _STATE.size:
        raise ValueError(f"invalid SONIC state packet size {len(packet)}")
    values = np.asarray(_STATE.unpack(packet), dtype=np.float32)
    cursor = 1

    def take(size: int) -> np.ndarray:
        nonlocal cursor
        result = values[cursor : cursor + size].copy()
        cursor += size
        return result

    return SonicState(
        time=float(values[0]),
        root_pose_wxyz=take(7),
        root_velocity=take(6),
        root_acceleration=take(3),
        torso_quaternion_wxyz=take(4),
        torso_velocity=take(6),
        joint_position=take(SONIC_DOF),
        joint_velocity=take(SONIC_DOF),
        joint_acceleration=take(SONIC_DOF),
        joint_torque=take(SONIC_DOF),
    )


def pack_command(command: SonicMotorCommand) -> bytes:
    values = np.concatenate(
        (
            _array(command.position, SONIC_DOF, "position"),
            _array(command.velocity, SONIC_DOF, "velocity"),
            _array(command.stiffness, SONIC_DOF, "stiffness"),
            _array(command.damping, SONIC_DOF, "damping"),
            _array(command.feedforward_torque, SONIC_DOF, "feedforward_torque"),
        )
    )
    return _COMMAND.pack(*values)


def unpack_command(packet: bytes) -> SonicMotorCommand:
    if len(packet) != _COMMAND.size:
        raise ValueError(f"invalid SONIC command packet size {len(packet)}")
    values = np.asarray(_COMMAND.unpack(packet), dtype=np.float32).reshape(5, SONIC_DOF)
    return SonicMotorCommand(*(row.copy() for row in values))


def command_torque(
    command: SonicMotorCommand,
    position: Sequence[float],
    velocity: Sequence[float],
    effort_limit: Sequence[float],
) -> np.ndarray:
    """Apply the same clipped low-level PD law as SONIC's MuJoCo bridge."""
    q = _array(position, SONIC_DOF, "position")
    dq = _array(velocity, SONIC_DOF, "velocity")
    limit = _array(effort_limit, SONIC_DOF, "effort_limit")
    torque = (
        command.feedforward_torque
        + command.stiffness * (command.position - q)
        + command.damping * (command.velocity - dq)
    )
    return np.clip(torque, -limit, limit)


class SonicUdpClient:
    """Non-blocking Isaac-side transport to the Python 3.10 DDS sidecar."""

    def __init__(
        self,
        *,
        sidecar_address: tuple[str, int] = ("127.0.0.1", 17890),
        bind_address: tuple[str, int] = ("127.0.0.1", 17891),
    ) -> None:
        self._sidecar_address = sidecar_address
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(bind_address)
        self._socket.setblocking(False)
        self.latest_command: SonicMotorCommand | None = None
        self.last_command_received_s: float | None = None

    def exchange(self, state: SonicState) -> SonicMotorCommand | None:
        self._socket.sendto(pack_state(state), self._sidecar_address)
        while True:
            try:
                packet, _ = self._socket.recvfrom(_COMMAND.size)
            except BlockingIOError:
                break
            self.latest_command = unpack_command(packet)
            self.last_command_received_s = time.monotonic()
        return self.latest_command

    def close(self) -> None:
        self._socket.close()
