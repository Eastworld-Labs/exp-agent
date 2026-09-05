from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


CAMERAS = ("head_camera", "cam_left_wrist", "cam_right_wrist")
BODY_JOINTS = tuple(
    [f"{side}_{part}_joint" for side in ("left", "right")
     for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{part}_joint" for part in ("yaw", "roll", "pitch")]
    + [f"{side}_{part}_joint" for side in ("left", "right")
       for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                    "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
# Native SONIC virtual landmarks, NOT the Dex-1 pinch centres.
VR_OFFSETS = {"left": (0.18, -0.025, 0), "right": (0.18, 0.025, 0), "head": (0, 0, 0.35)}
TASKS = {
    "tabletop": "Pick up the red block from the table and place it on the green target. Release it and leave it resting stably.",
    "floor_basket": "Use both grippers to grasp the two blue basket handles near the floor. Lower your body to reach, then stand and lift the basket clear of the floor. Hold it level with both hands.",
}


@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    task: str
    xml: str


def build_scene(asset_dir: Path, task: str) -> Scene:
    """Compose the local Unitree Dex-1 conversion without modifying its assets.

    The conversion fused the pelvis into world. Restore its floating body and
    root-link inertia from the source URDF.
    Camera mounts are experiment approximations, not measured calibration.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    asset_dir = asset_dir.resolve()
    tree = ET.parse(asset_dir / "g1_dex1_converted.xml").getroot()
    compiler = tree.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(asset_dir / "meshes"))
    ET.SubElement(tree, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    visual = ET.SubElement(tree, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    world = tree.find("worldbody")
    assert world is not None
    contents = list(world)
    for child in contents:
        world.remove(child)
    pelvis = ET.SubElement(world, "body", name="pelvis", pos="0 0 0.79")
    ET.SubElement(pelvis, "freejoint", name="floating_base_joint")
    # Pelvis root-link inertia from Unitree's URDF.
    ET.SubElement(pelvis, "inertial", pos="0 0 -0.07605", mass="3.813",
                  fullinertia="0.010549 0.0093089 0.0079184 0 0.0000021 0")
    pelvis.extend(contents)
    for parent in tree.iter():
        for child in list(parent):
            if child.tag == "camera":
                parent.remove(child)
    torso = tree.find(".//body[@name='torso_link']")
    assert torso is not None
    # MuJoCo looks down local -Z. X forward, Y left, Z up on the robot.
    # Head: forward/down 35 degrees; wrists: forward/down 25 degrees.
    def camera(body: ET.Element, name: str, pos: str, tilt: float) -> None:
        c, s = math.cos(tilt), math.sin(tilt)
        ET.SubElement(body, "camera", name=name, pos=pos, fovy="90",
                      xyaxes=f"0 -1 0 {s} 0 {c}")
    camera(torso, CAMERAS[0], "0.08 0 0.42", math.radians(35))
    ET.SubElement(torso, "site", name="vr_head", pos="0 0 0.35", size="0.004", rgba="0 0 0 0")
    for side in ("left", "right"):
        wrist = tree.find(f".//body[@name='{side}_wrist_yaw_link']")
        assert wrist is not None
        camera(wrist, f"cam_{side}_wrist", "0.045 0 0.055", math.radians(25))
        ET.SubElement(wrist, "site", name=f"vr_{side}",
                      pos=" ".join(map(str, VR_OFFSETS[side])), size="0.004", rgba="0 0 0 0")
    for joint in tree.findall(".//joint"):
        joint.set("damping", "0.05" if "dex1" in joint.get("name", "") else "0.1")
        joint.set("armature", "0.01")
    # Contact-only grasping: no object weld, mocap attachment, or gravity compensation.
    for body in tree.findall(".//body"):
        if "dex1_finger" in body.get("name", ""):
            for geom in body.findall("geom"):
                geom.set("friction", "1.0 0.005 0.0001")
                geom.set("condim", "4")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.1", rgba="0.25 0.3 0.35 1")
    object_body = ET.SubElement(world, "body", name="task_object",
                               pos="0.50 -0.16 0.731" if task == "tabletop" else "0.52 0 0.13")
    ET.SubElement(object_body, "freejoint", name="object_free")
    if task == "tabletop":
        ET.SubElement(world, "geom", name="table", type="box", pos="0.62 0 0.35",
                      size="0.27 0.42 0.35", rgba="0.6 0.5 0.38 1")
        ET.SubElement(world, "geom", name="target_pad", type="box", pos="0.50 0.19 0.704",
                      size="0.09 0.09 0.004", rgba="0.1 0.75 0.2 1")
        ET.SubElement(object_body, "geom", name="block", type="box", size="0.022 0.022 0.03",
                      mass="0.08", rgba="0.9 0.08 0.06 1", friction="0.8 0.005 0.0001")
    else:
        ET.SubElement(object_body, "geom", name="basket_bottom", type="box", pos="0 0 -0.11",
                      size="0.13 0.18 0.012", mass="0.2", rgba="0.85 0.55 0.2 1")
        for sign in (-1, 1):
            ET.SubElement(object_body, "geom", type="box", pos=f"0 {sign * 0.174} -0.04",
                          size="0.13 0.006 0.07", mass="0.05", rgba="0.85 0.55 0.2 1")
            ET.SubElement(object_body, "geom", type="box", pos=f"{sign * 0.124} 0 -0.04",
                          size="0.006 0.17 0.07", mass="0.05", rgba="0.85 0.55 0.2 1")
            for x in (-0.07, 0.07):
                ET.SubElement(object_body, "geom", type="capsule",
                              fromto=f"{x} {sign * 0.174} 0.025 {x} {sign * 0.22} 0.10",
                              size="0.01", mass="0.02", rgba="0.1 0.3 0.9 1")
            ET.SubElement(object_body, "geom", name=f"handle_{'left' if sign > 0 else 'right'}",
                          type="capsule", fromto=f"-0.07 {sign * 0.22} 0.10 0.07 {sign * 0.22} 0.10",
                          size="0.012", mass="0.03", rgba="0.1 0.3 0.9 1", friction="1 0.005 0.0001")
    actuators = ET.SubElement(tree, "actuator")
    for name in BODY_JOINTS:
        joint = tree.find(f".//joint[@name='{name}']")
        assert joint is not None
        ET.SubElement(actuators, "motor", name=name, joint=name, ctrllimited="true",
                      ctrlrange=joint.get("actuatorfrcrange", "-25 25"))
    for side in ("left", "right"):
        for finger in (1, 2):
            name = f"{side}_dex1_finger_joint_{finger}"
            ET.SubElement(actuators, "position", name=name, joint=name, kp="500", kv="5",
                          ctrllimited="true", ctrlrange="-0.02 0.0245", forcelimited="true", forcerange="-20 20")
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for side in ("left", "right"):
        for part, angle in (("hip_pitch", -0.1), ("knee", 0.3), ("ankle_pitch", -0.2),
                            ("elbow", 0.6), ("shoulder_roll", 0.15 if side == "left" else -0.15)):
            data.joint(f"{side}_{part}_joint").qpos[0] = angle
        for finger in (1, 2):
            name = f"{side}_dex1_finger_joint_{finger}"
            data.joint(name).qpos[0] = 0.0245
            data.ctrl[model.actuator(name).id] = 0.0245
    mujoco.mj_forward(model, data)
    return Scene(model, data, task, xml)


class Evaluator:
    """Privileged task truth; never register as an agent tool or snapshot."""

    def __init__(self, task: str) -> None:
        self.task = task
        self.minimum_pelvis_height = float("inf")
        self.maximum_object_height = 0.0
        self.hold_s = 0.0
        self.success = False
        self.failed = False
        self.last_time: float | None = None

    def update(self, model: mujoco.MjModel, data: mujoco.MjData, *, supported: bool = False) -> None:
        dt = 0 if self.last_time is None else max(0, float(data.time) - self.last_time)
        self.last_time = float(data.time)
        if supported:
            self.hold_s = 0
            return
        pelvis_height = float(data.body("pelvis").xpos[2])
        self.minimum_pelvis_height = min(self.minimum_pelvis_height, pelvis_height)
        # Roll/pitch from body's up-axis; deep upright squats remain allowed.
        self.failed |= pelvis_height < 0.28 or float(data.body("pelvis").xmat[8]) < 0.5
        obj = data.body("task_object")
        x, y, z = obj.xpos
        self.maximum_object_height = max(self.maximum_object_height, float(z))
        touches: set[str] = set()
        supported_by_target = False
        for contact in data.contact:
            if contact.dist > 0:
                continue
            b1, b2 = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if obj.id not in (b1, b2):
                continue
            other = b2 if b1 == obj.id else b1
            name = model.body(other).name
            for side in ("left", "right"):
                if name.startswith(f"{side}_dex1_finger"):
                    touches.add(side)
            supported_by_target |= any(model.geom(g).name == "target_pad" for g in (contact.geom1, contact.geom2))
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, obj.id, velocity, 0)
        stable = np.linalg.norm(velocity[3:]) < 0.04 and np.linalg.norm(velocity[:3]) < 0.2
        if self.task == "tabletop":
            valid = (abs(x - 0.50) < 0.06 and abs(y - 0.19) < 0.06 and
                     0.72 < z < 0.77 and supported_by_target and not touches and stable and
                     self.maximum_object_height > 0.80)
        else:
            valid = (z > 0.50 and obj.xmat[8] > math.cos(math.radians(20)) and
                     touches == {"left", "right"} and pelvis_height > 0.70 and
                     self.minimum_pelvis_height < 0.64 and stable)
        self.hold_s = self.hold_s + dt if valid and not self.failed else 0.0
        self.success = self.hold_s >= 2.0

    def result(self) -> dict:
        return {"success": self.success and not self.failed, "failed": self.failed,
                "stable_hold_s": self.hold_s,
                "minimum_pelvis_height_m": self.minimum_pelvis_height if math.isfinite(self.minimum_pelvis_height) else None,
                "maximum_object_height_m": self.maximum_object_height}
