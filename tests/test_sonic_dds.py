import numpy as np

from system2_agent.sim.sonic_dds import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    SONIC_DOF,
    SonicMotorCommand,
    SonicState,
    command_torque,
    pack_command,
    pack_state,
    unpack_command,
    unpack_state,
)


def test_udp_packets_round_trip() -> None:
    sequence = np.arange(SONIC_DOF, dtype=np.float32)
    state = SonicState(
        1.25,
        np.arange(7),
        np.arange(6),
        np.arange(3),
        np.arange(4),
        np.arange(6),
        sequence,
        sequence + 1,
        sequence + 2,
        sequence + 3,
    )
    restored = unpack_state(pack_state(state))
    assert restored.time == 1.25
    np.testing.assert_array_equal(restored.joint_torque, sequence + 3)

    command = SonicMotorCommand(*(sequence + index for index in range(5)))
    restored_command = unpack_command(pack_command(command))
    np.testing.assert_array_equal(restored_command.damping, sequence + 3)


def test_torque_matches_clipped_sonic_pd_law() -> None:
    ones = np.ones(SONIC_DOF, dtype=np.float32)
    command = SonicMotorCommand(2 * ones, ones, 10 * ones, 2 * ones, 3 * ones)
    torque = command_torque(command, ones, np.zeros(SONIC_DOF), 12 * ones)
    np.testing.assert_array_equal(torque, 12 * ones)


def test_joint_selection_vectors_are_inverses() -> None:
    mujoco = np.arange(SONIC_DOF)
    isaac = mujoco[MUJOCO_TO_ISAACLAB]
    np.testing.assert_array_equal(isaac[ISAACLAB_TO_MUJOCO], mujoco)
