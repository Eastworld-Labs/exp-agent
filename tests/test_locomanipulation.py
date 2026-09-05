import json
import math
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("zmq")
pytest.importorskip("PIL")

from system2_agent.experiments.locomanipulation.control import (
    Pose, Trajectory, body_pose, native_targets, rotation,
)
from system2_agent.experiments.locomanipulation.runtime import Runtime, compose_evidence_frame
from system2_agent.experiments.locomanipulation.scene import (
    BODY_JOINTS, CAMERAS, Evaluator, build_scene,
)
from system2_agent.sonic_bridge import HEADER_SIZE, pack_sonic_planner


def unpack(packet):
    assert packet.startswith(b"planner")
    header = json.loads(packet[7:7 + HEADER_SIZE].rstrip(b"\0"))
    offset = 7 + HEADER_SIZE
    fields = {}
    for field in header["fields"]:
        count = math.prod(field["shape"])
        code = "i" if field["dtype"] == "i32" else "f"
        fields[field["name"]] = struct.unpack_from(f"<{count}{code}", packet, offset)
        offset += count * 4
    assert offset == len(packet)
    return fields


def poses():
    root = Pose(np.zeros(3), np.array([1., 0, 0, 0]))
    wrists = {s: Pose(np.array([0.3, y, 0.1]), root.quaternion.copy()) for s, y in (("left", 0.2), ("right", -0.2))}
    return root, wrists, Pose(np.array([0., 0, 0.4]), root.quaternion.copy())


def trajectory(args):
    root, wrists, head = poses()
    return Trajectory(args, wrists, root, head, 0, {"left": 1., "right": 1.})


def test_native_packet_combines_body_and_three_points():
    packet = pack_sonic_planner(mode=4, movement=(0, 0, 0), facing=(1, 0, 0), speed=0,
                               height=0.6, vr_position=list(range(9)), vr_orientation=[1, 0, 0, 0] * 3)
    fields = unpack(packet)
    assert fields["mode"] == (4,)
    assert fields["height"] == pytest.approx((0.6,))
    assert fields["vr_position"] == tuple(range(9))
    assert len(fields["vr_orientation"]) == 12
    assert "upper_body_position" not in fields


@pytest.mark.parametrize("extra", [
    {"vr_position": [0] * 8}, {"vr_position": [float("nan")] * 9},
    {"vr_orientation": [1, 0, 0, 0] * 3},
    {"vr_position": [0] * 9, "upper_body_position": [0] * 17},
])
def test_invalid_native_fields(extra):
    with pytest.raises(ValueError):
        pack_sonic_planner(mode=0, movement=(0, 0, 0), facing=(1, 0, 0), speed=0, **extra)


def test_native_mapping_offsets_and_full_root_transform():
    root, wrists, head = poses()
    expected, orn = native_targets(wrists, head, root)
    assert expected == pytest.approx([0.48, 0.175, 0.1, 0.48, -0.175, 0.1, 0, 0, 0.4])
    # Arbitrary translated AND rolled/pitched/yawed root: not just yaw normalization.
    q = np.array([0.8, 0.2, -0.3, 0.4]); q /= np.linalg.norm(q)
    moved_root = Pose(np.array([2., -1, 0.7]), q)
    actual, actual_orn = native_targets({s: p.to_world(moved_root) for s, p in wrists.items()}, head, moved_root)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    np.testing.assert_allclose(actual_orn, orn, atol=1e-8)


def test_interpolation_and_quaternion_sign():
    t = trajectory({"duration_s": 2, "left_wrist": {"position_m": [0.4, 0.2, 0.1], "quaternion_wxyz": [-1, 0, 0, 0]}})
    start, _, _ = t.sample(0)
    middle, _, _ = t.sample(1)
    end, _, _ = t.sample(2)
    assert start["left"].position[0] == pytest.approx(0.3)
    assert middle["left"].position[0] == pytest.approx(0.35)
    assert end["left"].position[0] == pytest.approx(0.4)
    assert abs(end["left"].quaternion[0]) == pytest.approx(1)
    np.testing.assert_allclose(start["right"].position, end["right"].position)


def test_walking_stops_at_deadline_but_keeps_vr_targets():
    root, _, _ = poses()
    t = trajectory({"duration_s": 2, "body": {"mode": "slow_walk", "velocity_xy_m_s": [0.1, 0]}})
    before = unpack(t.packet(1, root)); after = unpack(t.packet(2, root))
    assert before["mode"] == (1,)
    assert before["speed"] == pytest.approx((0.1,))
    assert after["mode"] == (0,)
    assert after["speed"] == (0,)
    assert after["vr_position"] == before["vr_position"]


def test_head_target_supports_bounded_waist_lean():
    root, _, _ = poses()
    q = [math.cos(0.1), 0, math.sin(0.1), 0]
    t = trajectory({"duration_s": 2, "body": {"mode": "idle", "head_orientation_wxyz": q}})
    fields = unpack(t.packet(2, root))
    assert fields["vr_orientation"][-4:] == pytest.approx(q)
    assert fields["vr_position"][-3:] == pytest.approx([0.35 * math.sin(0.2), 0, 0.05 + 0.35 * math.cos(0.2)])
    with pytest.raises(ValueError, match="tilt"):
        trajectory({"body": {"head_orientation_wxyz": [0, 1, 0, 0]}})


@pytest.mark.parametrize("args", [
    {"duration_s": float("nan")}, {"duration_s": 0.1},
    {"left_gripper": 2}, {"left_gripper": float("nan")},
    {"body": {"mode": "run"}}, {"body": {"mode": "squat", "height_m": 0.2}},
    {"body": {"mode": "idle", "velocity_xy_m_s": [0.1, 0]}},
    {"body": {"mode": "slow_walk", "velocity_xy_m_s": [0.5, 0]}},
    {"left_wrist": {"position_m": [0.7, 0.2, 0.1], "quaternion_wxyz": [1, 0, 0, 0]}},
    {"left_wrist": {"position_m": [0.3, 0.2, 0.1], "quaternion_wxyz": [0, 0, 0, 0]}},
    {"left_wrist": {"position_m": [float("nan"), 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}},
])
def test_action_limits(args):
    with pytest.raises(ValueError):
        trajectory(args)


@pytest.fixture(params=["tabletop", "floor_basket"])
def scene(request):
    assets = Path(__file__).resolve().parents[2] / "g1_sim_pipeline/models/dex1_urdf"
    if not assets.exists():
        pytest.skip("local Dex-1 model not installed; see experiments/locomanipulation/README.md")
    return build_scene(assets, request.param)


def test_scene_is_dynamic_dex1_with_only_three_body_mounted_cameras(scene):
    m = scene.model
    assert m.ncam == 3
    assert tuple(m.camera(i).name for i in range(3)) == CAMERAS
    assert all(m.cam_bodyid > 0)
    assert m.joint("floating_base_joint").type == mujoco.mjtJoint.mjJNT_FREE
    assert m.joint("object_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert m.nu == 33
    assert m.neq == 0 and m.nmocap == 0
    assert tuple(m.actuator(i).name for i in range(29)) == BODY_JOINTS
    assert m.actuator("left_dex1_finger_joint_1").ctrlrange == pytest.approx([-0.02, 0.0245])


def test_proprioception_does_not_leak_object_truth(scene, tmp_path):
    runtime = Runtime(scene, output=tmp_path)
    first = runtime.proprioception()
    scene.data.joint("object_free").qpos[:3] += [0.2, 0.1, 0.1]
    mujoco.mj_forward(scene.model, scene.data)
    assert runtime.proprioception() == first
    assert "task_object" not in json.dumps(first)
    assert "success" not in json.dumps(first)
    assert len(first["joint_positions_rad"]) == 29
    runtime.close()


def test_renderer_uses_only_allowlisted_cameras(scene, tmp_path):
    class Renderer:
        calls = []
        def update_scene(self, data, camera):
            self.calls.append(camera)
        def render(self):
            return np.zeros((48, 64, 3), dtype=np.uint8)
        def close(self):
            pass
    runtime = Runtime(scene, output=tmp_path)
    runtime.renderer = Renderer()
    frames = runtime.capture()
    assert tuple(frame.label for frame in frames) == CAMERAS
    assert runtime.renderer.calls == list(CAMERAS)
    assert all(frame.url.startswith("data:image/jpeg;base64,") for frame in frames)
    runtime.close()


def test_evidence_frame_has_three_agent_views_observer_and_action_panel():
    from PIL import Image
    images = {
        "head_camera": Image.new("RGB", (64, 48), (255, 0, 0)),
        "cam_left_wrist": Image.new("RGB", (64, 48), (0, 255, 0)),
        "cam_right_wrist": Image.new("RGB", (64, 48), (0, 0, 255)),
    }
    observer = Image.new("RGB", (64, 48), (255, 255, 0))
    frame = compose_evidence_frame(images, observer, 3, "executing move_to",
                                   {"body": {"mode": "squat"}, "duration_s": 2})
    assert frame.size == (1280, 720)
    assert frame.getpixel((500, 300)) == (255, 255, 0)
    assert frame.getpixel((1000, 300)) == (255, 0, 0)
    assert frame.getpixel((800, 650)) == (0, 255, 0)
    assert frame.getpixel((1150, 650)) == (0, 0, 255)


def test_supported_or_untouched_scene_cannot_pass(scene):
    evaluator = Evaluator(scene.task)
    for i in range(100):
        scene.data.time = i * 0.05
        evaluator.update(scene.model, scene.data, supported=True)
    assert not evaluator.success
    evaluator.update(scene.model, scene.data)
    assert not evaluator.success


def test_preflight_required_for_actions(scene, tmp_path):
    runtime = Runtime(scene, output=tmp_path)
    with pytest.raises(RuntimeError, match="preflight"):
        runtime.execute({"duration_s": 1})
    runtime.close()


def test_actual_contact_dynamics_and_evaluator(scene):
    """Component test only: immobilized robot, gravity-driven object settling.

    This is NOT a SONIC rollout or a demonstrated pick-and-place. It verifies
    that the props have real dynamics and the private scorer detects release.
    """
    m, d = scene.model, scene.data
    robot_qpos = d.qpos.copy()
    object_adr = int(m.joint("object_free").qposadr[0])
    object_vel = int(m.joint("object_free").dofadr[0])
    evaluator = Evaluator(scene.task)
    if scene.task == "tabletop":
        d.joint("object_free").qpos[:3] = [0.5, 0.19, 0.85]
    else:
        d.joint("object_free").qpos[2] = 0.2
    initial_height = float(d.joint("object_free").qpos[2])
    for _ in range(1600):
        # Freeze ONLY the robot for this isolated object/contact unit test.
        d.qpos[:object_adr] = robot_qpos[:object_adr]
        d.qvel[:object_vel] = 0
        mujoco.mj_step(m, d)
        evaluator.update(m, d)
    assert d.body("task_object").xpos[2] < initial_height - 0.02
    assert np.isfinite(d.qpos).all()
    if scene.task == "tabletop":
        assert evaluator.success
        assert evaluator.hold_s >= 2
    else:
        assert not evaluator.success  # Ground contact is not a bimanual lift.
