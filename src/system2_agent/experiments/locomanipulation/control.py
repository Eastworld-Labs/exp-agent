from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np

from ...sonic_bridge import pack_sonic_planner
from ...tools import Tool, object_schema
from .scene import VR_OFFSETS


def vector(value: Any, size: int, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return a


def quaternion(value: Any) -> np.ndarray:
    q = vector(value, 4, "quaternion_wxyz")
    norm = np.linalg.norm(q)
    if norm < 1e-8 or abs(norm - 1) > 0.05:
        raise ValueError("quaternion_wxyz must be a unit quaternion")
    return q / norm


def rotation(q: np.ndarray) -> np.ndarray:
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def inverse(q: np.ndarray) -> np.ndarray:
    return q * np.array([1, -1, -1, -1])


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = min(1, dot)
    if dot > 0.9995:
        q = (1 - t) * a + t * b
        return q / np.linalg.norm(q)
    angle = math.acos(dot)
    return (math.sin((1 - t) * angle) * a + math.sin(t * angle) * b) / math.sin(angle)


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    quaternion: np.ndarray

    def relative_to(self, root: "Pose") -> "Pose":
        return Pose(rotation(root.quaternion).T @ (self.position - root.position),
                    multiply(inverse(root.quaternion), self.quaternion))

    def to_world(self, root: "Pose") -> "Pose":
        return Pose(root.position + rotation(root.quaternion) @ self.position,
                    multiply(root.quaternion, self.quaternion))

    def as_json(self) -> dict:
        return {"position_m": self.position.tolist(), "quaternion_wxyz": self.quaternion.tolist()}


def body_pose(data: mujoco.MjData, name: str) -> Pose:
    return Pose(data.body(name).xpos.copy(), data.body(name).xquat.copy())


def native_targets(wrists: dict[str, Pose], head_local: Pose, root: Pose) -> tuple[list, list]:
    """Match upstream GatherVR3PointPosition/Orientation fallback exactly.

    Wrist body poses -> SONIC virtual landmark offsets -> inverse full root
    transform. The separate 1.1 heading-anchor observation is deploy-owned.
    No Dex-1 pinch-centre or human tracker offsets are silently substituted.
    """
    poses = []
    for side in ("left", "right"):
        wrist = wrists[side]
        point = Pose(wrist.position + rotation(wrist.quaternion) @ np.array(VR_OFFSETS[side]),
                     wrist.quaternion).relative_to(root)
        poses.append(point)
    poses.append(head_local)
    return ([float(v) for pose in poses for v in pose.position],
            [float(v) for pose in poses for v in pose.quaternion])


class Trajectory:
    """Bounded root-at-action-start wrist targets, held in world during motion."""

    def __init__(self, args: Mapping[str, Any], wrists: dict[str, Pose], root: Pose,
                 head: Pose, now: float, apertures: dict[str, float], previous_height: float = -1) -> None:
        allowed = {"left_wrist", "right_wrist", "left_gripper", "right_gripper", "body", "duration_s"}
        if set(args) - allowed:
            raise ValueError(f"unknown action fields: {set(args) - allowed}")
        self.duration = float(args.get("duration_s", 2))
        if not math.isfinite(self.duration) or not 0.5 <= self.duration <= 5:
            raise ValueError("duration_s must be between 0.5 and 5 seconds")
        self.start_time = now
        self.start = wrists
        self.target = dict(wrists)
        self.head_local = head.relative_to(root)
        self.gripper_start = dict(apertures)
        self.grippers = dict(apertures)
        for side in ("left", "right"):
            if f"{side}_wrist" in args:
                raw = args[f"{side}_wrist"]
                if not isinstance(raw, Mapping) or set(raw) != {"position_m", "quaternion_wxyz"}:
                    raise ValueError("wrist pose requires position_m and quaternion_wxyz only")
                local = Pose(vector(raw["position_m"], 3, "position_m"), quaternion(raw["quaternion_wxyz"]))
                if np.linalg.norm(local.position) > 0.9:
                    raise ValueError("wrist target exceeds the experimental 0.9 m root workspace")
                target = local.to_world(root)
                distance = np.linalg.norm(target.position - wrists[side].position)
                angle = 2 * math.acos(min(1, abs(float(np.dot(target.quaternion, wrists[side].quaternion)))))
                # Quintic easing peak velocity is 1.875 times mean velocity.
                if distance > 0.25 or 1.875 * distance / self.duration > 0.20:
                    raise ValueError("wrist move exceeds 0.25 m per call or 0.20 m/s peak; use shorter moves or more time")
                if 1.875 * angle / self.duration > 0.7:
                    raise ValueError("wrist rotation exceeds 0.7 rad/s peak")
                self.target[side] = target
            aperture = float(args.get(f"{side}_gripper", apertures[side]))
            if not math.isfinite(aperture) or not 0 <= aperture <= 1:
                raise ValueError("gripper aperture must be between 0 (closed) and 1 (open)")
            self.grippers[side] = aperture
        body = args.get("body", {})
        if not isinstance(body, Mapping) or set(body) - {"mode", "height_m", "velocity_xy_m_s", "turn_radians", "head_orientation_wxyz"}:
            raise ValueError("invalid body fields")
        self.head_target = self.head_local
        if "head_orientation_wxyz" in body:
            q = quaternion(body["head_orientation_wxyz"])
            angle = 2 * math.acos(min(1, abs(float(np.dot(q, self.head_local.quaternion)))))
            if rotation(q)[2, 2] < math.cos(math.radians(35)) or 1.875 * angle / self.duration > 0.7:
                raise ValueError("head/torso target exceeds 35 degree tilt or 0.7 rad/s peak")
            # Upstream ThreePointPose: root -> torso (+0.05 Z) -> neck (+0.35 local Z).
            self.head_target = Pose(np.array([0., 0, 0.05]) + rotation(q) @ np.array([0., 0, 0.35]), q)
        mode = body.get("mode", "idle")
        if mode not in {"idle", "slow_walk", "squat"}:
            raise ValueError("body mode must be idle, slow_walk or squat")
        self.mode = {"idle": 0, "slow_walk": 1, "squat": 4}[mode]
        self.height = float(body.get("height_m", -1))
        if not math.isfinite(self.height) or (mode == "squat" and not 0.40 <= self.height <= 0.80):
            raise ValueError("squat requires height_m in [0.40, 0.80]")
        if mode != "squat" and self.height != -1:
            raise ValueError("height_m is only supported for squat")
        self.start_height = previous_height if previous_height >= 0 else float(root.position[2])
        if mode == "squat" and abs(self.height - self.start_height) > 0.15:
            raise ValueError("lower/raise the squat height by at most 0.15 m per action")
        velocity = vector(body.get("velocity_xy_m_s", [0, 0]), 2, "velocity_xy_m_s")
        self.speed = float(np.linalg.norm(velocity))
        turn = float(body.get("turn_radians", 0))
        if not math.isfinite(turn) or abs(turn) > 0.25 or abs(turn) / self.duration > 0.15:
            raise ValueError("turn must be <= 0.25 rad and <= 0.15 rad/s")
        if self.speed > 0.15 or self.speed * self.duration > 0.25:
            raise ValueError("walking is limited to 0.15 m/s and 0.25 m per action")
        if mode != "slow_walk" and (self.speed > 0 or turn != 0):
            raise ValueError("velocity and turning require slow_walk mode")
        world_velocity = rotation(root.quaternion) @ np.array([*velocity, 0.0])
        norm = np.linalg.norm(world_velocity[:2])
        self.movement = (float(world_velocity[0] / norm), float(world_velocity[1] / norm), 0.0) if norm > 1e-8 else (0.0, 0.0, 0.0)
        self.yaw = math.atan2(rotation(root.quaternion)[1, 0], rotation(root.quaternion)[0, 0])
        self.turn = turn

    def sample(self, now: float) -> tuple[dict[str, Pose], dict[str, float], float]:
        t = float(np.clip((now - self.start_time) / self.duration, 0, 1))
        u = t * t * t * (10 + t * (-15 + 6 * t))
        wrists = {s: Pose((1 - u) * self.start[s].position + u * self.target[s].position,
                          slerp(self.start[s].quaternion, self.target[s].quaternion, u)) for s in self.start}
        apertures = {s: (1 - u) * self.gripper_start[s] + u * self.grippers[s] for s in self.grippers}
        return wrists, apertures, u

    def packet(self, now: float, root: Pose) -> bytes:
        wrists, _, u = self.sample(now)
        head = Pose((1 - u) * self.head_local.position + u * self.head_target.position,
                    slerp(self.head_local.quaternion, self.head_target.quaternion, u))
        positions, orientations = native_targets(wrists, head, root)
        # Stop translation at the action deadline, including while the VLM thinks.
        finished = now >= self.start_time + self.duration
        mode = 0 if finished and self.mode == 1 else self.mode
        yaw = self.yaw + self.turn * u
        return pack_sonic_planner(mode=mode, movement=(0, 0, 0) if finished else self.movement,
                                  facing=(math.cos(yaw), math.sin(yaw), 0), speed=0 if finished else self.speed,
                                  height=(self.start_height + u * (self.height - self.start_height)) if self.mode == 4 else -1,
                                  vr_position=positions, vr_orientation=orientations)


class ManipulationModule:
    name = "locomanipulation"

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def snapshot(self) -> dict:
        return self.runtime.proprioception()

    def tools(self) -> tuple[Tool, ...]:
        pose = object_schema({"position_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                              "quaternion_wxyz": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}},
                             required=("position_m", "quaternion_wxyz"))
        return (Tool("move_to", "Move both wrists and body together. Wrist poses are absolute in the full pelvis frame sampled at action start (X forward, Y left, Z up), NOT displacements or pinch centres. Omitted wrists hold their world pose. Grippers: 0 closed, 1 open. Observe after every move. Body mode must be explicit on every action.",
                     object_schema({"left_wrist": pose, "right_wrist": pose,
                                    "left_gripper": {"type": "number", "minimum": 0, "maximum": 1},
                                    "right_gripper": {"type": "number", "minimum": 0, "maximum": 1},
                                    "duration_s": {"type": "number", "minimum": 0.5, "maximum": 5},
                                    "body": object_schema({"mode": {"type": "string", "enum": ["idle", "slow_walk", "squat"]},
                                                           "height_m": {"type": "number"},
                                                           "velocity_xy_m_s": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                                           "head_orientation_wxyz": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                                                           "turn_radians": {"type": "number"}}, required=("mode",))},
                                   required=("body", "duration_s")),
                     self.runtime.execute, kind="action"),)


SYSTEM_PROMPT = """You control a simulated G1 Dex-1 humanoid through SONIC 1.1.
Your only scene observations are head_camera, cam_left_wrist and cam_right_wrist RGB.
Robot proprioception is provided; object locations, depth, segmentation and evaluator
truth are NOT available. Infer object locations from multiple views, move cautiously,
and visually recheck. You may choose numeric Cartesian wrist targets, body posture
and jaw apertures, but never joint/torque targets. Make exactly one tool call per turn.
move_to positions describe wrist_yaw_link, not the fingers: each pinch centre is
approximately 0.13 m along wrist-local +X (an uncalibrated mesh estimate). Jaws close
along wrist-local +/-Y. Quaternions use w,x,y,z. Body actions are not automatic IK:
if reaching low, lower in small squat-height increments and coordinate both wrists.
The head target follows the pelvis during squatting; there is no actuated neck.
Optional body.head_orientation_wxyz commands head/torso orientation relative to
the pelvis to request a small waist lean (<=35 degrees), not neck articulation.
On each call explicitly retain the desired body mode/height. Do not assume a requested
pose was reached: inspect returned measured tracking errors and fresh images. While
you think the executor holds targets and stops walking. Call finish only after visual
verification; finish is a claim, not the private physical task score. Request a human
if repeated tracking errors or falls prevent progress.
"""
