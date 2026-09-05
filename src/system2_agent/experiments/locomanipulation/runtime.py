from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
import zmq
from PIL import Image

from ...modules.camera import CameraFrame
from ...sim.sonic_dds import SonicState, SonicUdpClient, command_torque
from ...sonic_bridge import pack_sonic_command, pack_sonic_planner
from .control import Pose, Trajectory, body_pose
from .scene import BODY_JOINTS, CAMERAS, Evaluator, Scene


class Runtime:
    """Loopback-only MuJoCo <-> DDS sidecar <-> official SONIC deployment.

    The physics thread alone owns ZMQ/UDP. Rendering uses an atomic copy, not
    live data shared with the renderer. No IK, kinematic playback, or surrogate
    WBC is used for task execution. Preview can render without a controller.
    """

    def __init__(self, scene: Scene, *, output: Path, startup_s: float = 60,
                 support_s: float = 12, max_episode_s: float = 300) -> None:
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
        deadline = time.monotonic() + self.startup_s + self.support_s + 5
        while not self.ready.wait(0.1):
            self.check_health()
            if time.monotonic() > deadline:
                raise RuntimeError("SONIC startup timed out")
        self.check_health()

    def check_health(self) -> None:
        if self.error:
            raise RuntimeError(self.error)

    def _loop(self) -> None:
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        client = None
        try:
            publisher.bind("tcp://127.0.0.1:5556")
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
                        publisher.send(pack_sonic_command(start=True, stop=False, planner=True))
                        next_start = wall + 0.5
                    if wall >= next_publish:
                        if self.trajectory is not None:
                            packet = self.trajectory.packet(float(self.data.time), body_pose(self.data, "pelvis"))
                        else:
                            packet = pack_sonic_planner(mode=0, movement=(0, 0, 0), facing=(1, 0, 0), speed=0)
                        publisher.send(packet)
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
                publisher.send(pack_sonic_command(start=False, stop=True, planner=True))
            finally:
                publisher.close(linger=0)
                context.term()
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
                    "controller_ran": self.ready.is_set(),
                    "calibration_status": "experimental robot-root mapping; real camera/grasp calibration not measured"}
