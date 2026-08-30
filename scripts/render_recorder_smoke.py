#!/usr/bin/env python3
"""Render a few static composite frames without launching SONIC or simulation."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco

from system2_agent.modules import Pose3D
from system2_agent.navigation_core import GridMap
from system2_agent.official_sonic_sim_cli import _VideoRecorder
from system2_agent.scene_bundle import SceneBundle
from system2_agent.scene_loader import SceneLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--compose",
        action="store_true",
        help="Treat model as the immutable robot MJCF and attach the bundle's physical scene",
    )
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--yaw", type=float, default=0.0)
    args = parser.parse_args()
    scene = SceneBundle.from_json(args.bundle)
    loaded = SceneLoader(args.model).load(scene) if args.compose else None
    model_path = loaded.model_path if loaded is not None else args.model.resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    requested_pose = None
    if args.x is not None or args.y is not None:
        if args.x is None or args.y is None:
            parser.error("--x and --y must be provided together")
        requested_pose = (args.x, args.y, args.yaw)
    elif scene.initial_pose is not None:
        requested_pose = scene.initial_pose
    if requested_pose is not None:
        pose_x, pose_y, pose_yaw = requested_pose
        data.qpos[0:2] = (pose_x, pose_y)
        data.qpos[3:7] = (
            math.cos(pose_yaw / 2),
            0.0,
            0.0,
            math.sin(pose_yaw / 2),
        )
    mujoco.mj_forward(model, data)

    def pose() -> Pose3D:
        w, x, y, z = [float(value) for value in data.qpos[3:7]]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return Pose3D(float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2]), yaw)

    start = pose()
    recorder = _VideoRecorder(
        args.output,
        model=model,
        data=data,
        fps=5,
        width=320,
        height=240,
        gaussian_splat=scene.gaussian_splat,
        gaussian_alignment=scene.gaussian_alignment,
        navigation_grid=GridMap.from_json(scene.navigation_grid),
        planned_path=[start, Pose3D(start.x + 4, start.y, start.z, start.yaw)],
        pose_provider=pose,
    )
    recorder.start()
    time.sleep(1.0)
    recorder.close()
    print(recorder.summary())
    if loaded is not None:
        loaded.close()


if __name__ == "__main__":
    main()
