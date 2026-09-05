#!/usr/bin/env python3
"""Bridge Isaac's local UDP state stream to SONIC's existing Unitree DDS API."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WBC = Path(
    os.environ.get("SONIC_WBC_DIR", ROOT.parent / "GR00T-WholeBodyControl")
).expanduser().resolve()
sys.path[:0] = [str(ROOT / "src"), str(WBC)]

from system2_agent.sim.sonic_dds import (  # noqa: E402
    SONIC_DOF,
    SonicMotorCommand,
    pack_command,
    unpack_state,
)
from cyclonedds.domain import DomainParticipant  # noqa: F401,E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge  # noqa: E402


class FreshCommandBridge(UnitreeSdk2Bridge):
    """Do not make an old DDS command look fresh by retransmitting it over UDP."""

    def __init__(self, config):
        self.last_motor_command_s = 0.0
        super().__init__(config)

    def LowCmdHandler(self, msg):
        super().LowCmdHandler(msg)
        self.last_motor_command_s = time.monotonic()


def configuration() -> dict[str, object]:
    return {
        "ROBOT_TYPE": "g1_29dof",
        "NUM_MOTORS": SONIC_DOF,
        "NUM_HAND_MOTORS": 7,
        "USE_SENSOR": False,
    }


def command_from_bridge(bridge: UnitreeSdk2Bridge) -> SonicMotorCommand:
    motors = bridge.low_cmd.motor_cmd[:SONIC_DOF]
    return SonicMotorCommand(
        position=np.asarray([motor.q for motor in motors], dtype=np.float32),
        velocity=np.asarray([motor.dq for motor in motors], dtype=np.float32),
        stiffness=np.asarray([motor.kp for motor in motors], dtype=np.float32),
        damping=np.asarray([motor.kd for motor in motors], dtype=np.float32),
        feedforward_torque=np.asarray([motor.tau for motor in motors], dtype=np.float32),
    )


def publish(bridge: UnitreeSdk2Bridge, packet: bytes) -> None:
    state = unpack_state(packet)
    zeros = np.zeros(7, dtype=np.float32)
    bridge.PublishLowState(
        {
            "time": state.time,
            "floating_base_pose": state.root_pose_wxyz,
            "floating_base_vel": state.root_velocity,
            "floating_base_acc": state.root_acceleration,
            "secondary_imu_quat": state.torso_quaternion_wxyz,
            "secondary_imu_vel": state.torso_velocity,
            "body_q": state.joint_position,
            "body_dq": state.joint_velocity,
            "body_ddq": state.joint_acceleration,
            "body_tau_est": state.joint_torque,
            "left_hand_q": zeros,
            "left_hand_dq": zeros,
            "right_hand_q": zeros,
            "right_hand_dq": zeros,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, default=17890)
    parser.add_argument("--isaac-port", type=int, default=17891)
    args = parser.parse_args()
    ChannelFactoryInitialize(0, "lo")
    bridge = FreshCommandBridge(configuration())
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.listen_port))
    target = ("127.0.0.1", args.isaac_port)
    print("SONIC_DDS_SIDECAR_READY", flush=True)
    while True:
        packet, _ = sock.recvfrom(8192)
        try:
            publish(bridge, packet)
        except ValueError:
            continue
        if bridge.low_cmd_received and time.monotonic() - bridge.last_motor_command_s < 0.1:
            sock.sendto(pack_command(command_from_bridge(bridge)), target)


if __name__ == "__main__":
    main()
