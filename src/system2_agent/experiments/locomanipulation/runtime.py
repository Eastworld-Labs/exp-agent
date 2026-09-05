from __future__ import annotations

import base64
import io
import json
import math
import subprocess
import threading
import textwrap
import time
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ...modules.camera import CameraFrame
from ...sim.sonic_dds import SonicState, SonicUdpClient, command_torque
from ...sonic_bridge import sonic_planner_action
from .control import Pose, Trajectory, body_pose
from .scene import BODY_JOINTS, CAMERAS, Evaluator, Scene


def compose_evidence_frame(images: Mapping[str, Image.Image], observer_image: Image.Image,
                           action_index: int, phase: str, action: Mapping[str, Any] | None) -> Image.Image:
    """Compose experiment evidence without adding the observer to agent inputs."""
    canvas = Image.new("RGB", (1280, 720), "black")
    canvas.paste(observer_image.resize((640, 480), Image.Resampling.BILINEAR), (0, 0))
    canvas.paste(images["head_camera"].resize((640, 480), Image.Resampling.BILINEAR), (640, 0))
    canvas.paste(images["cam_left_wrist"].resize((320, 240), Image.Resampling.BILINEAR), (640, 480))
    canvas.paste(images["cam_right_wrist"].resize((320, 240), Image.Resampling.BILINEAR), (960, 480))
    labels = ImageDraw.Draw(canvas)
    for x, y, label in ((0, 0, "evidence-only third person"),
                        (640, 0, "AGENT: head camera"),
                        (640, 480, "AGENT: left wrist"),
                        (960, 480, "AGENT: right wrist")):
        labels.rectangle((x, y, x + 190, y + 22), fill="black")
        labels.text((x + 6, y + 5), label, fill="white")
    labels.rectangle((0, 480, 640, 720), fill=(8, 10, 13))
    labels.text((12, 492), f"ACTION STEP {action_index} | {phase}", fill=(80, 220, 255))
    action_text = ("No task action yet; SONIC is establishing unsupported balance."
                   if action is None else json.dumps(action, sort_keys=True))
    y = 520
    for line in textwrap.wrap(action_text, width=84)[:12]:
        labels.text((12, y), line, fill="white")
        y += 16
    return canvas


class Runtime:
    """Loopback-only MuJoCo <-> DDS sidecar <-> official SONIC deployment.

    The physics thread alone owns ZMQ/UDP. Rendering uses an atomic copy, not
    live data shared with the renderer. No IK, kinematic playback, or surrogate
    WBC is used for task execution. Preview can render without a controller.
    """

    def __init__(self, scene: Scene, *, output: Path, startup_s: float = 60,
                 support_s: float = 12, max_episode_s: float = 300,
                 record_video: bool = False, record_fps: float = 5) -> None:
        self.scene = scene
        self.model, self.data = scene.model, scene.data
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.startup_s, self.support_s = startup_s, support_s
        self.max_episode_s = max_episode_s
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: str | None = None
        self.evaluator = Evaluator(scene.task)
        self.trajectory: Trajectory | None = None
        self.apertures = {"left": 1.0, "right": 1.0}
        self.frame_index = 0
        self.actions: list[dict] = []
        self.telemetry: list[dict] = []
        self.renderer: mujoco.Renderer | None = None
        self.record_video = record_video
        self.record_fps = record_fps
        self.video_path = self.output / "camera_evidence.mp4"
        self.video_stop = threading.Event()
        self.video_thread: threading.Thread | None = None
        self.video_error: str | None = None
        self.video_action_index = 0
        self.video_action: dict[str, Any] | None = None
        self.video_phase = "SONIC startup / supported preflight"
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in BODY_JOINTS])
        self.vids = np.array([self.model.joint(n).dofadr[0] for n in BODY_JOINTS])
        self.aids = np.array([self.model.actuator(n).id for n in BODY_JOINTS])
        self.limits = self.model.actuator_ctrlrange[self.aids, 1].copy()
        self.initial_q = self.data.qpos[self.qids].copy()
        self.supported = True

    def _wrists(self) -> dict[str, Pose]:
        return {s: body_pose(self.data, f"{s}_wrist_yaw_link") for s in ("left", "right")}

    def _head(self) -> Pose:
        return Pose(self.data.site("vr_head").xpos.copy(), self.data.body("torso_link").xquat.copy())

    def _state(self) -> SonicState:
        velocity = np.zeros(6)
        torso = self.data.body("torso_link")
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, torso.id, velocity, 1)
        return SonicState(float(self.data.time), self.data.qpos[:7].copy(), self.data.qvel[:6].copy(),
                          self.data.qacc[:3].copy(), torso.xquat.copy(), np.r_[velocity[3:], velocity[:3]],
                          self.data.qpos[self.qids].copy(), self.data.qvel[self.vids].copy(),
                          self.data.qacc[self.vids].copy(), self.data.actuator_force[self.aids].copy())

    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("runtime already started")
        self.thread = threading.Thread(target=self._loop, name="locomanipulation-physics", daemon=True)
        self.thread.start()
        if self.record_video:
            self.video_thread = threading.Thread(target=self._record_loop, name="locomanipulation-video", daemon=True)
            self.video_thread.start()
        deadline = time.monotonic() + self.startup_s + self.support_s + 5
        while not self.ready.wait(0.1):
            self.check_health()
            if time.monotonic() > deadline:
                raise RuntimeError("SONIC startup timed out")
        self.check_health()

    def _record_loop(self) -> None:
        """Record public cameras plus a strictly evidence-only observer view."""
        renderer = None
        encoder = None
        try:
            renderer = mujoco.Renderer(self.model, width=640, height=480)
            observer = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(observer)
            observer.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            observer.trackbodyid = self.model.body("pelvis").id
            observer.distance = 2.8
            observer.azimuth = 145
            observer.elevation = -18
            encoder = subprocess.Popen([
                "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", "1280x720",
                "-framerate", str(self.record_fps), "-i", "-", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", str(self.video_path),
            ], stdin=subprocess.PIPE)
            period = 1.0 / self.record_fps
            deadline = time.monotonic()
            while not self.video_stop.is_set() and not self.stop_event.is_set():
                with self.lock:
                    snapshot = mujoco.MjData(self.model)
                    mujoco.mj_copyData(snapshot, self.model, self.data)
                    action_index = self.video_action_index
                    action = dict(self.video_action) if self.video_action is not None else None
                    phase = self.video_phase
                images = {}
                for name in CAMERAS:
                    renderer.update_scene(snapshot, camera=name)
                    images[name] = Image.fromarray(renderer.render().copy())
                renderer.update_scene(snapshot, camera=observer)
                observer_image = Image.fromarray(renderer.render().copy())
                canvas = compose_evidence_frame(images, observer_image, action_index, phase, action)
                assert encoder.stdin is not None
                encoder.stdin.write(canvas.tobytes())
                deadline += period
                if deadline < time.monotonic() - period:
                    deadline = time.monotonic()
                self.video_stop.wait(max(0, deadline - time.monotonic()))
        except Exception as exc:
            self.video_error = str(exc)
        finally:
            if renderer is not None:
                renderer.close()
            if encoder is not None:
                if encoder.stdin is not None:
                    encoder.stdin.close()
                return_code = encoder.wait(timeout=10)
                if return_code and self.video_error is None:
                    self.video_error = f"ffmpeg exited with status {return_code}"

    def check_health(self) -> None:
        if self.error:
            raise RuntimeError(self.error)

    def _loop(self) -> None:
        from robot import ZmqSonicPlannerBridge

        publisher = ZmqSonicPlannerBridge(sonic_variant="sonic_v1_1")
        client = None
        try:
            publisher.connect()
            client = SonicUdpClient()
            boot_time = time.monotonic()
            first_control = None
            next_publish = next_start = 0.0
            next_telemetry = 0.0
            next_step = time.monotonic()
            while not self.stop_event.is_set():
                wall = time.monotonic()
                with self.lock:
                    command = client.exchange(self._state())
                    if command is not None:
                        values = np.concatenate([command.position, command.velocity, command.stiffness,
                                                 command.damping, command.feedforward_torque])
                        if not np.isfinite(values).all() or (command.stiffness < 0).any() or (command.damping < 0).any():
                            raise RuntimeError("invalid SONIC motor command")
                        if first_control is None and (command.stiffness > 0).any():
                            first_control = wall
                    if first_control is None and wall - boot_time > self.startup_s:
                        raise RuntimeError("no active SONIC motor commands; start the DDS sidecar and official deployer")
                    if first_control is not None:
                        if client.last_command_received_s is None or wall - client.last_command_received_s > 0.25:
                            raise RuntimeError("SONIC motor-command watchdog expired (>250 ms)")
                    self.supported = first_control is None or wall - first_control < self.support_s
                    if self.supported and wall >= next_start:
                        publisher.start_control(planner=True)
                        next_start = wall + 0.5
                    if wall >= next_publish:
                        if self.trajectory is not None:
                            action = self.trajectory.planner_action(
                                float(self.data.time), body_pose(self.data, "pelvis")
                            )
                        else:
                            action = sonic_planner_action(
                                mode=0, movement=(0, 0, 0), facing=(1, 0, 0), speed=0
                            )
                        publisher.send_action(action)
                        next_publish = wall + 0.02
                    if command is not None and first_control is not None:
                        self.data.ctrl[self.aids] = command_torque(command, self.data.qpos[self.qids],
                                                                  self.data.qvel[self.vids], self.limits)
                    else:
                        # Startup only, while explicitly supported. Never a task controller.
                        self.data.ctrl[self.aids] = np.clip(40 * (self.initial_q - self.data.qpos[self.qids])
                                                           - 3 * self.data.qvel[self.vids], -self.limits, self.limits)
                    self.data.xfrc_applied[:] = 0
                    if self.supported:
                        pelvis = self.data.body("pelvis")
                        self.data.xfrc_applied[pelvis.id, :3] = (
                            600 * (np.array([0, 0, 1.0]) - pelvis.xpos) - 80 * self.data.qvel[:3])
                    if self.trajectory is not None:
                        _, self.apertures, _ = self.trajectory.sample(float(self.data.time))
                    for side, aperture in self.apertures.items():
                        for finger in (1, 2):
                            self.data.ctrl[self.model.actuator(f"{side}_dex1_finger_joint_{finger}").id] = -0.02 + 0.0445 * aperture
                    mujoco.mj_step(self.model, self.data)
                    if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                        raise RuntimeError("non-finite physics state")
                    self.evaluator.update(self.model, self.data, supported=self.supported)
                    if self.data.time >= next_telemetry:
                        self.telemetry.append({
                            "time_s": float(self.data.time), "supported": self.supported,
                            "root_pose_wxyz": self.data.qpos[:7].tolist(),
                            "joint_positions_rad": self.data.qpos[self.qids].tolist(),
                            "measured_wrists_world": {s: p.as_json() for s, p in self._wrists().items()},
                            "private_object_pose_world": body_pose(self.data, "task_object").as_json(),
                            "evaluation": self.evaluator.result(),
                        })
                        next_telemetry = float(self.data.time) + 0.1
                    if self.evaluator.failed:
                        raise RuntimeError("robot fell or exceeded the upright-body safety limit")
                    if first_control is not None and wall - first_control >= self.support_s + 2 and not self.ready.is_set():
                        if self.data.body("pelvis").xpos[2] < 0.70:
                            raise RuntimeError("SONIC failed unsupported standing preflight")
                        self.trajectory = Trajectory({"duration_s": 1, "body": {"mode": "idle"}}, self._wrists(),
                                                     body_pose(self.data, "pelvis"), self._head(),
                                                     float(self.data.time), self.apertures)
                        # Preflight motion must not satisfy the task's squat/lift history.
                        self.evaluator = Evaluator(self.scene.task)
                        self.video_phase = "unsupported standing ready / awaiting action"
                        self.ready.set()
                    if wall - boot_time > self.max_episode_s:
                        raise RuntimeError("episode wall-clock limit reached")
                next_step += float(self.model.opt.timestep)
                # Slow simulations must not cause an unbounded catch-up burst.
                if next_step < wall - 0.05:
                    next_step = wall
                self.stop_event.wait(max(0, next_step - time.monotonic()))
        except Exception as exc:
            self.error = str(exc)
        finally:
            # Loopback simulator only. Never a live robot transport.
            try:
                if publisher.is_connected:
                    publisher.stop_control(planner=True)
            finally:
                publisher.disconnect()
                if client is not None:
                    client.close()

    def proprioception(self) -> dict:
        self.check_health()
        with self.lock:
            root = body_pose(self.data, "pelvis")
            camera_calibration = {}
            for name in CAMERAS:
                camera = self.data.camera(name)
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, camera.xmat)
                f = 240 / math.tan(math.radians(float(self.model.camera(name).fovy[0])) / 2)
                camera_calibration[name] = {
                    "pose_in_pelvis": Pose(camera.xpos.copy(), q).relative_to(root).as_json(),
                    "fx": f, "fy": f, "cx": 320, "cy": 240,
                    "width": 640, "height": 480,
                    "axes": "MuJoCo/OpenGL: +X image-right, +Y image-up, -Z optical forward",
                    "status": "exact simulated mount; hardware calibration unmeasured",
                }
            return {"frame": "pelvis_at_action_start; X forward, Y left, Z up; quaternion wxyz",
                    "simulation_time_s": float(self.data.time),
                    "wrist_poses": {s: p.relative_to(root).as_json() for s, p in self._wrists().items()},
                    "joint_positions_rad": dict(zip(BODY_JOINTS, self.data.qpos[self.qids].tolist())),
                    "pelvis_orientation_wxyz": root.quaternion.tolist(),
                    "gripper_apertures": {
                        s: float(np.clip(np.mean([self.data.joint(f"{s}_dex1_finger_joint_{i}").qpos[0]
                                                  for i in (1, 2)]) + 0.02, 0, 0.0445) / 0.0445)
                        for s in ("left", "right")},
                    "controller": "SONIC 1.1 native planner + three-point teleop",
                    "requested_body": {"mode": self.trajectory.mode, "height_m": self.trajectory.height} if self.trajectory else None,
                    "camera_names": list(CAMERAS), "camera_calibration": camera_calibration}

    def execute(self, args: Mapping[str, Any]) -> dict:
        self.check_health()
        if not self.ready.is_set():
            raise RuntimeError("task actions require an unsupported SONIC preflight")
        with self.lock:
            previous_height = self.trajectory.height if self.trajectory is not None else -1
            trajectory = Trajectory(args, self._wrists(), body_pose(self.data, "pelvis"), self._head(),
                                    float(self.data.time), self.apertures, previous_height)
            self.trajectory = trajectory
            self.video_action_index += 1
            self.video_action = dict(args)
            self.video_phase = "executing move_to"
        deadline = time.monotonic() + trajectory.duration * 3 + 5
        while True:
            self.check_health()
            with self.lock:
                done = self.data.time >= trajectory.start_time + trajectory.duration
            if done:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("simulation did not finish the action in time")
            time.sleep(0.02)
        with self.lock:
            actual = self._wrists()
            errors = {s: {"position_m": float(np.linalg.norm(actual[s].position - trajectory.target[s].position)),
                          "orientation_rad": 2 * math.acos(min(1, abs(float(np.dot(actual[s].quaternion, trajectory.target[s].quaternion)))))}
                      for s in actual}
            result = {"tracking_error": errors,
                      "tracking_ok": all(e["position_m"] < 0.05 and e["orientation_rad"] < 0.3 for e in errors.values()),
                      "proprioception": self.proprioception()}
            self.actions.append({"action": dict(args), "result": result})
            self.video_phase = "action complete / holding targets"
            return result

    def capture(self) -> tuple[CameraFrame, ...]:
        self.check_health()
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        with self.lock:
            snapshot = mujoco.MjData(self.model)
            mujoco.mj_copyData(snapshot, self.model, self.data)
        frames = []
        directory = self.output / "frames"
        directory.mkdir(exist_ok=True)
        for name in CAMERAS:  # Strict allowlist: never overview/depth/segmentation.
            self.renderer.update_scene(snapshot, camera=name)
            image = Image.fromarray(self.renderer.render().copy())
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            payload = buffer.getvalue()
            (directory / f"{self.frame_index:04d}_{name}.jpg").write_bytes(payload)
            frames.append(CameraFrame(name, "data:image/jpeg;base64," + base64.b64encode(payload).decode()))
        self.frame_index += 1
        return tuple(frames)

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.video_stop.set()
        if self.video_thread is not None:
            self.video_thread.join(timeout=15)
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.telemetry:
            with (self.output / "private_telemetry.jsonl").open("w") as stream:
                for sample in self.telemetry:
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")

    def report(self) -> dict:
        with self.lock:
            return {"task": self.scene.task, "physics_evaluation": self.evaluator.result(),
                    "runtime_error": self.error, "actions": self.actions,
                    "observation_cameras": list(CAMERAS),
                    "video": str(self.video_path) if self.record_video else None,
                    "video_error": self.video_error,
                    "controller_ran": self.ready.is_set(),
                    "calibration_status": "experimental robot-root mapping; real camera/grasp calibration not measured"}
