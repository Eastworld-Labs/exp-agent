#!/usr/bin/env python3
"""Bounded free-base G1/PhysX contact test using SONIC's low-level gains."""

from __future__ import annotations

import argparse
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT.parent
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--sonic", action="store_true")
parser.add_argument("--vx", type=float, default=0.0)
parser.add_argument("--scene-bundle", type=Path)
parser.add_argument("--route-goal")
parser.add_argument("--record", type=Path)
parser.add_argument("--record-fps", type=float, default=10.0)
parser.add_argument("--preview-frame", action="store_true")
parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
parser.add_argument("--wbc-dir", type=Path)
parser.add_argument("--robot-class-dir", type=Path)
parser.add_argument("--g1-usd", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
WORKSPACE = args.workspace.expanduser().resolve()
WBC = (args.wbc_dir or WORKSPACE / "GR00T-WholeBodyControl").expanduser().resolve()
ROBOT_CLASS = (args.robot_class_dir or WORKSPACE / "robot_class").expanduser().resolve()
G1_USD = (
    args.g1_usd
    or WBC / "gear_sonic/data/robots/g1/g1_29dof_cylinder/main_nodex.usd"
).expanduser().resolve()
if args.scene_bundle:
    nurec_kit_args = (
        "--enable omni.rtx.spg "
        "--enable isaacsim.replicator.nurec_utils "
        "--/renderer/multiGpu/enabled=false"
    )
    args.kit_args = f"{args.kit_args} {nurec_kit_args}".strip()
launcher = AppLauncher(args)
simulation_app = launcher.app
print("CONTACT_SMOKE_IMPORT torch", flush=True)

import torch  # noqa: E402
print("CONTACT_SMOKE_IMPORT sim_utils", flush=True)
import isaaclab.sim as sim_utils  # noqa: E402
print("CONTACT_SMOKE_IMPORT actuators", flush=True)
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
print("CONTACT_SMOKE_IMPORT articulation", flush=True)
from isaaclab.assets import Articulation  # noqa: E402
print("CONTACT_SMOKE_IMPORT articulation_cfg", flush=True)
from isaaclab.assets.articulation import ArticulationCfg  # noqa: E402
print("CONTACT_SMOKE_IMPORT simulation_context", flush=True)
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse  # noqa: E402

sys.path[:0] = [str(ROOT / "src"), str(ROBOT_CLASS)]
from system2_agent.modules.semantic_map import Pose3D, SemanticMapModule  # noqa: E402
from system2_agent.navigation_core import (  # noqa: E402
    GridMap,
    RegulatedTrajectoryFollower,
    SmoothTrajectoryPlanner,
    VelocityCommand,
)
from system2_agent.scene_bundle import SceneBundle  # noqa: E402
from system2_agent.sim.isaac_scene import (  # noqa: E402
    IsaacRouteRecorder,
    compose_multiroom_stage,
    configure_nurec_rendering,
)
from system2_agent.sim.sonic_dds import (  # noqa: E402
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    SonicState,
    SonicUdpClient,
    command_torque,
)
from system2_agent.sonic_bridge import SonicZmqBase  # noqa: E402
print("CONTACT_SMOKE_IMPORT complete", flush=True)


DEFAULT_MJ = torch.tensor(
    [-0.312, 0, 0, 0.669, -0.363, 0,
     -0.312, 0, 0, 0.669, -0.363, 0,
     0, 0, 0, 0.2, 0.2, 0, 0.6, 0, 0, 0,
     0.2, -0.2, 0, 0.6, 0, 0, 0], dtype=torch.float32,
)
KP_MJ = torch.tensor(
    [150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40,
     250, 250, 250, 100, 100, 40, 40, 20, 20, 20,
     100, 100, 40, 40, 20, 20, 20], dtype=torch.float32,
)
KD_MJ = torch.tensor(
    [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5,
     5, 5, 2, 2, 2, 2, 2, 5, 5, 2, 2, 2, 2, 2], dtype=torch.float32,
)
LIMIT_MJ = torch.tensor(
    [139, 139, 88, 139, 50, 50, 139, 139, 88, 139, 50, 50, 88, 50, 50,
     25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], dtype=torch.float32,
)
MJ_TO_ISAAC = torch.tensor(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=torch.long,
)
MUJOCO_JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"
    )
] + [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
] + [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw",
    )
]


def main() -> None:
    scene = SceneBundle.from_json(args.scene_bundle) if args.scene_bundle else None
    spawn = scene.initial_pose if scene and scene.initial_pose else (0.0, 0.0, 0.0)
    spawn_x, spawn_y, spawn_yaw = (float(value) for value in spawn)
    if scene is not None:
        configure_nurec_rendering()
    print("CONTACT_SMOKE_CREATE_SIM", flush=True)
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.005,
            render_interval=4,
            device=args.device,
            physics=PhysxCfg(
                solver_type=1,
                min_position_iteration_count=8,
                min_velocity_iteration_count=4,
            ),
        )
    )
    print("CONTACT_SMOKE_CREATE_GROUND", flush=True)
    ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
        )
    )
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=1200.0)
    light.func("/World/Light", light)
    if scene is not None:
        nurec = scene.gaussian_splat.with_name(
            "interior_0184_840116.lod300000.gsplat.usdc"
        )
        composition = compose_multiroom_stage(
            nurec_usdz=nurec,
            gaussian_alignment=scene.gaussian_alignment,
            navigation_grid=scene.navigation_grid,
            simfoundry_assets=ROOT / "assets/simfoundry/assets",
        )
        print(f"ISAAC_SCENE_COMPOSED {composition['collision_rectangles']} collisions", flush=True)
        from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering

        import omni.usd

        ready, detected, _, problems = setup_for_rendering(
            omni.usd.get_context().get_stage()
        )
        if not ready or not detected:
            raise RuntimeError(f"NuRec render setup failed: {problems or 'scene not detected'}")
    route_cameras = None
    if args.record:
        route_cameras = (
            Camera(CameraCfg(
                prim_path="/World/ChaseCamera", update_period=0.0,
                height=540, width=960, data_types=["rgb"], spawn=None,
            )),
            Camera(CameraCfg(
                prim_path="/World/HeadCamera", update_period=0.0,
                height=270, width=480, data_types=["rgb"], spawn=None,
            )),
        )
    cfg = ArticulationCfg(
        prim_path="/World/G1",
        # Use SONIC's pre-authored cylinder USD. Isaac Sim 6 removed the URDF
        # importer's cylinder-to-capsule switch, so importing main.urdf here
        # silently changes the collision model used during SONIC training.
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(G1_USD),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(spawn_x, spawn_y, 0.76),
            rot=(0.0, 0.0, math.sin(spawn_yaw / 2.0), math.cos(spawn_yaw / 2.0)),
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint"],
                effort_limit_sim={
                    ".*_hip_(pitch|roll)_joint": 139.0,
                    ".*_hip_yaw_joint": 88.0,
                    ".*_knee_joint": 139.0,
                },
                velocity_limit_sim={
                    ".*_hip_(pitch|roll)_joint": 20.0,
                    ".*_hip_yaw_joint": 32.0,
                    ".*_knee_joint": 20.0,
                },
                stiffness=0.0,
                damping=0.0,
                armature={".*_hip_(pitch|roll)_joint": 0.025101925,
                          ".*_hip_yaw_joint": 0.010177520,
                          ".*_knee_joint": 0.025101925},
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_.*_joint"], effort_limit_sim=50.0,
                velocity_limit_sim=37.0, stiffness=0.0, damping=0.0,
                armature=0.00721945,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*_joint"], effort_limit_sim=88.0,
                velocity_limit_sim=32.0, stiffness=0.0, damping=0.0,
                armature={"waist_yaw_joint": 0.010177520,
                          "waist_(roll|pitch)_joint": 0.00721945},
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_(shoulder|elbow|wrist)_.*joint"],
                effort_limit_sim=25.0, velocity_limit_sim=37.0,
                stiffness=0.0, damping=0.0,
                armature={".*_(shoulder_.*|elbow|wrist_roll)_joint": 0.003609725,
                          ".*_wrist_(pitch|yaw)_joint": 0.00425},
            ),
        },
    )
    robot = Articulation(cfg)
    sim.reset()
    print("JOINT_NAMES=" + ",".join(robot.joint_names), flush=True)
    if len(robot.joint_names) != 29:
        raise RuntimeError(f"expected 29 G1 joints, found {len(robot.joint_names)}")
    missing = sorted(set(MUJOCO_JOINT_NAMES) - set(robot.joint_names))
    if missing:
        raise RuntimeError(f"G1 asset is missing SONIC joints: {missing}")
    # Resolve by name instead of assuming a USD/URDF traversal order. This is
    # important because PhysX tensor ordering can change with the source asset.
    order = torch.tensor(
        [MUJOCO_JOINT_NAMES.index(name) for name in robot.joint_names],
        dtype=torch.long,
        device=sim.device,
    )
    target = DEFAULT_MJ.to(sim.device)[order].unsqueeze(0)
    kp = KP_MJ.to(sim.device)[order].unsqueeze(0)
    kd = KD_MJ.to(sim.device)[order].unsqueeze(0)
    limit = LIMIT_MJ.to(sim.device)[order].unsqueeze(0)
    robot.write_joint_position_to_sim_index(position=target)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target))
    robot.reset()
    if args.preview_frame:
        if scene is None or not args.route_goal or not args.record or route_cameras is None:
            raise RuntimeError("preview requires --scene-bundle, --route-goal, and --record")
        preview_grid = GridMap.from_json(scene.navigation_grid)
        preview_goal = SemanticMapModule.from_json(scene.semantic_map).resolve(args.route_goal)
        preview_path = SmoothTrajectoryPlanner(
            preview_grid, footprint_radius_m=scene.navigation_footprint_radius_m
        ).plan(Pose3D(spawn_x, spawn_y, 0.76, spawn_yaw), preview_goal)
        preview = IsaacRouteRecorder(
            args.record, grid=preview_grid, path=preview_path,
            cameras=route_cameras, fps=args.record_fps,
        )
        preview.prepare(Pose3D(spawn_x, spawn_y, 0.76, spawn_yaw))
        for _ in range(35):
            sim.render()
        if not preview.write(Pose3D(spawn_x, spawn_y, 0.76, spawn_yaw), "NuRec preview"):
            raise RuntimeError("NuRec preview camera produced no frame")
        preview.close()
        print("ISAAC_NUREC_PREVIEW_COMPLETE", flush=True)
        return
    if args.sonic:
        run_sonic(sim, robot, target, kp, kd, limit, scene, route_cameras)
        return
    minimum_z = float("inf")
    maximum_z = 0.0
    for _ in range(args.steps):
        q = robot.data.joint_pos.torch
        dq = robot.data.joint_vel.torch
        effort = torch.clip(kp * (target - q) - kd * dq, -limit, limit)
        robot.set_joint_effort_target_index(target=effort)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(0.005)
        z = float(robot.data.root_link_pos_w.torch[0, 2])
        minimum_z = min(minimum_z, z)
        maximum_z = max(maximum_z, z)
    final_z = float(robot.data.root_link_pos_w.torch[0, 2])
    print(
        f"CONTACT_SMOKE final_z={final_z:.4f} min_z={minimum_z:.4f} "
        f"max_z={maximum_z:.4f} steps={args.steps}", flush=True
    )
    if final_z < 0.55 or minimum_z < 0.45:
        raise RuntimeError("free-base G1 failed the floor/contact stability check")


def run_sonic(
    sim, robot, target, kp, kd, limit, scene: SceneBundle | None,
    route_cameras: tuple[Camera, Camera] | None,
) -> None:
    """Validate unchanged SONIC 1.1 through the DDS sidecar, then release support."""
    wbc = WBC
    log_dir = ROOT / "artifacts"
    log_dir.mkdir(exist_ok=True)
    sidecar_log = open(log_dir / "isaac_sonic_sidecar.log", "w", encoding="utf-8")
    deploy_log = open(log_dir / "isaac_sonic_deploy.log", "w", encoding="utf-8")
    def allocate_udp_port() -> int:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
        probe.close()
        return port

    sidecar_port = allocate_udp_port()
    isaac_port = allocate_udp_port()
    sidecar = subprocess.Popen(
        [
            str(wbc / ".venv_sim/bin/python"), "-u",
            str(ROOT / "scripts/sonic_dds_sidecar.py"),
            "--listen-port", str(sidecar_port),
            "--isaac-port", str(isaac_port),
        ],
        cwd=ROOT,
        env={**os.environ, "SONIC_WBC_DIR": str(wbc)},
        stdout=sidecar_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    client = None
    bridge = None
    deploy = None
    try:
        time.sleep(1.0)
        if sidecar.poll() is not None:
            raise RuntimeError("SONIC DDS sidecar failed during startup; see its artifact log")

        def pose() -> Pose3D:
            value = robot.data.root_link_pose_w.torch[0].detach().cpu().numpy()
            x, y, z, w = (float(v) for v in value[3:7])
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            return Pose3D(float(value[0]), float(value[1]), float(value[2]), yaw)

        bridge = SonicZmqBase(pose, settle_s=0.0, sonic_variant="sonic_v1_1")
        deploy = subprocess.Popen(
            [
                sys.executable,
                str(ROBOT_CLASS / "scripts/setup_sonic_deploy.py"),
                "--wbc-dir", str(wbc),
                "--launch-deploy",
                "--variant", "sonic_v1_1",
                "--robot-interface", "sim",
                "--deploy-input-type", "zmq_manager",
                "--deploy-planner", "planner/target_vel/V2/planner_sonic.onnx",
            ],
            cwd=ROBOT_CLASS,
            env={**os.environ, "SONIC_WBC_DIR": str(wbc)},
            stdin=subprocess.PIPE,
            stdout=deploy_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        assert deploy.stdin is not None
        deploy.stdin.write("\n")
        deploy.stdin.close()
        client = SonicUdpClient(
            sidecar_address=("127.0.0.1", sidecar_port),
            bind_address=("127.0.0.1", isaac_port),
        )
        torso_id = robot.find_bodies("torso_link")[0][0]
        to_mj = torch.tensor(
            [robot.joint_names.index(name) for name in MUJOCO_JOINT_NAMES],
            dtype=torch.long,
            device=sim.device,
        )
        to_isaac = torch.tensor(
            [MUJOCO_JOINT_NAMES.index(name) for name in robot.joint_names],
            dtype=torch.long,
            device=sim.device,
        )
        limit_mj = limit[0, to_mj].detach().cpu().numpy()
        previous_root_velocity = torch.zeros(6, device=sim.device)
        pelvis_id = robot.find_bodies("pelvis")[0][0]
        pelvis_ids = torch.tensor([pelvis_id], device=sim.device, dtype=torch.long)
        # Preserve Isaac's initialized standing/contact height. The vendor
        # MuJoCo elastic-band demo anchors at z=1.0 for its own model; copying
        # that literal into this asset lifted both feet ~20 cm off the floor
        # and caused a free-fall impulse when support was released.
        initial_root = robot.data.root_link_pos_w.torch[0].clone()
        support_anchor = initial_root.clone()
        zero_torque = torch.zeros((1, 1, 3), device=sim.device)
        first_command_step = None
        release_start_step = None
        release_step = None
        minimum_free_z = float("inf")
        free_steps = 0
        report = log_dir / "isaac_sonic_stability.txt"
        telemetry = [
            "step,free,z,axis_angle,linear_speed,angular_speed,max_q_error,max_torque"
        ]
        # The vendor deployer can spend 20-30 seconds initializing its Unitree
        # service clients before publishing the first motor command.
        route_path = None
        route_grid = None
        route_follower = None
        route_waypoint = 1
        route_reached = False
        route_goal = None
        recorder = None
        if scene is not None and args.route_goal:
            route_grid = GridMap.from_json(scene.navigation_grid)
            route_goal = SemanticMapModule.from_json(scene.semantic_map).resolve(args.route_goal)
            route_planner = SmoothTrajectoryPlanner(
                route_grid, footprint_radius_m=scene.navigation_footprint_radius_m
            )
            route_path = route_planner.plan(pose(), route_goal)
            route_follower = RegulatedTrajectoryFollower(
                route_grid,
                max_speed=0.28,
                minimum_speed=0.20,
                max_yaw_rate=0.30,
                holonomic=False,
                turn_in_place=True,
                turn_in_place_threshold=0.50,
                align_final_yaw=False,
            )
            print(
                f"ISAAC_ROUTE goal={args.route_goal} points={len(route_path)} "
                f"length={route_planner.last_plan_metrics['length_m']}", flush=True
            )
            if args.record:
                if route_cameras is None:
                    raise RuntimeError("recording requested without initialized Isaac cameras")
                recorder = IsaacRouteRecorder(
                    args.record,
                    grid=route_grid,
                    path=route_path,
                    cameras=route_cameras,
                    fps=args.record_fps,
                )
                recorder.prepare(pose())
                for _ in range(35):
                    sim.render()
        total_steps = max(args.steps, 18000 if route_path else 7500)
        for step in range(total_steps):
            tick = time.monotonic()
            q_isaac = robot.data.joint_pos.torch[0]
            dq_isaac = robot.data.joint_vel.torch[0]
            q_mj = q_isaac[to_mj].detach().cpu().numpy()
            dq_mj = dq_isaac[to_mj].detach().cpu().numpy()
            root_pose = robot.data.root_link_pose_w.torch[0]
            root_velocity_w = robot.data.root_link_vel_w.torch[0]
            root_quat = root_pose[3:7]
            # IMU gyro measurements are in the sensor/body frame. Rotate the
            # world-frame PhysX angular velocities accordingly (wxyz quats).
            root_ang_b = quat_apply_inverse(root_quat.unsqueeze(0), root_velocity_w[3:].unsqueeze(0))[0]
            root_velocity = torch.cat((root_velocity_w[:3], root_ang_b))
            root_acceleration = ((root_velocity - previous_root_velocity) / 0.005)[:3]
            previous_root_velocity = root_velocity.clone()
            torso_quat = robot.data.body_link_quat_w.torch[0, torso_id]
            torso_velocity_w = robot.data.body_link_vel_w.torch[0, torso_id]
            torso_ang_b = quat_apply_inverse(
                torso_quat.unsqueeze(0), torso_velocity_w[3:].unsqueeze(0)
            )[0]
            torso_velocity = torch.cat((torso_velocity_w[:3], torso_ang_b))
            joint_acceleration = robot.data.joint_acc.torch[0, to_mj]
            applied = robot.data.applied_torque.torch[0, to_mj]
            # Isaac Lab 3 / PhysX tensors use xyzw. Unitree DDS and SONIC use
            # wxyz, as does MuJoCo's qpos. Convert only at this backend edge;
            # Isaac math helpers below intentionally continue using xyzw.
            root_pose_wxyz = torch.cat((root_pose[:3], root_quat[3:], root_quat[:3]))
            torso_quat_wxyz = torch.cat((torso_quat[3:], torso_quat[:3]))
            command = client.exchange(
                SonicState(
                    step * 0.005,
                    root_pose_wxyz.detach().cpu().numpy(),
                    root_velocity.detach().cpu().numpy(),
                    root_acceleration.detach().cpu().numpy(),
                    torso_quat_wxyz.detach().cpu().numpy(),
                    torso_velocity.detach().cpu().numpy(),
                    q_mj,
                    dq_mj,
                    joint_acceleration.detach().cpu().numpy(),
                    applied.detach().cpu().numpy(),
                )
            )
            # The DDS shim publishes a zero-valued placeholder before the
            # SONIC deployer has loaded its policy. Never interpret that
            # transport-ready packet as a locomotion command or begin support
            # handoff from it.
            command_ready = (
                command is not None
                and np.isfinite(command.position).all()
                and np.isfinite(command.stiffness).all()
                and float(np.max(command.stiffness)) > 1.0
                and float(np.max(np.abs(command.position))) > 0.05
            )
            if step % 10 == 0:
                bridge.start(planner=True)
                navigation_command = VelocityCommand(0.0, 0.0, 0.0)
                if release_step is not None and step >= release_step:
                    if route_path is not None and route_follower is not None and not route_reached:
                        navigation_command, route_reached, route_waypoint = route_follower.command_path(
                            pose(), route_path, route_waypoint
                        )
                    elif route_path is None:
                        navigation_command = VelocityCommand(args.vx, 0.0, 0.0)
                bridge.command_velocity(navigation_command, 0.05)
            if not command_ready:
                effort = torch.clip(kp * (target - q_isaac) - kd * dq_isaac, -limit, limit)
            else:
                if first_command_step is None:
                    first_command_step = step
                    release_start_step = step + 500
                    release_step = release_start_step + 400
                    print(
                        f"SONIC_FIRST_COMMAND step={step} "
                        f"q=[{command.position.min():.3f},{command.position.max():.3f}] "
                        f"kp=[{command.stiffness.min():.3f},{command.stiffness.max():.3f}] "
                        f"tau=[{command.feedforward_torque.min():.3f},"
                        f"{command.feedforward_torque.max():.3f}]",
                        flush=True,
                    )
                torque_mj = command_torque(command, q_mj, dq_mj, limit_mj)
                if not np.isfinite(torque_mj).all():
                    raise RuntimeError("SONIC emitted a non-finite motor command")
                effort = torch.as_tensor(torque_mj, device=sim.device)[to_isaac].unsqueeze(0)
            if first_command_step is not None and step % 10 == 0:
                attitude = float(torch.linalg.vector_norm(axis_angle_from_quat(root_quat.unsqueeze(0))[0]))
                telemetry.append(
                    f"{step},{int(release_step is not None and step >= release_step)},"
                    f"{float(root_pose[2]):.6f},{attitude:.6f},"
                    f"{float(torch.linalg.vector_norm(root_velocity_w[:3])):.6f},"
                    f"{float(torch.linalg.vector_norm(root_velocity_w[3:])):.6f},"
                    f"{float(torch.max(torch.abs(target - q_isaac))):.6f},"
                    f"{float(torch.max(torch.abs(effort))):.6f}"
                )
            # Match the official MuJoCo startup band's compliant 6-D wrench;
            # unlike a pose lock this lets contacts and joint motion evolve.
            if release_step is None or step < release_step:
                if release_start_step is None or step < release_start_step:
                    support_scale = 1.0
                else:
                    support_scale = (release_step - step) / (
                        release_step - release_start_step
                    )
                support_force = (
                    10000.0 * (support_anchor - root_pose[:3])
                    - 1000.0 * root_velocity_w[:3]
                ).mul(support_scale).reshape(1, 1, 3)
                support_torque = (
                    -1000.0 * axis_angle_from_quat(root_quat.unsqueeze(0))[0]
                    - 10.0 * root_velocity_w[3:]
                ).mul(support_scale).reshape(1, 1, 3)
                robot.permanent_wrench_composer.set_forces_and_torques_index(
                    forces=support_force,
                    torques=support_torque,
                    body_ids=pelvis_ids,
                    is_global=True,
                )
            else:
                if free_steps == 0:
                    robot.permanent_wrench_composer.set_forces_and_torques_index(
                        forces=torch.zeros((1, 1, 3), device=sim.device),
                        torques=zero_torque,
                        body_ids=pelvis_ids,
                        is_global=True,
                    )
                free_steps += 1
                z = float(root_pose[2])
                minimum_free_z = min(minimum_free_z, z)
                if free_steps % 100 == 0:
                    report.write_text(
                        f"free_steps={free_steps} z={z:.4f} min_z={minimum_free_z:.4f}\n",
                        encoding="utf-8",
                    )
            capture_period = max(1, round(200.0 / args.record_fps))
            do_capture = (
                recorder is not None
                and release_step is not None
                and step >= release_step
                and step % capture_period == 0
            )
            if do_capture:
                recorder.prepare(pose())
            robot.set_joint_effort_target_index(target=effort)
            robot.write_data_to_sim()
            sim.step(render=do_capture)
            robot.update(0.005)
            if do_capture:
                recorder.write(
                    pose(),
                    f"Isaac Sim + NuRec Interior 0184 | SONIC 1.1 | to {args.route_goal}",
                )
            if route_reached and free_steps > 400:
                if recorder is None or free_steps % 400 == 0:
                    break
            elapsed = time.monotonic() - tick
            if elapsed < 0.005:
                time.sleep(0.005 - elapsed)
        final = pose()
        (log_dir / "isaac_sonic_telemetry.csv").write_text(
            "\n".join(telemetry) + "\n", encoding="utf-8"
        )
        print(
            f"SONIC_ISAAC_SMOKE final=({final.x:.3f},{final.y:.3f},{final.z:.3f}) "
            f"min_free_z={minimum_free_z:.3f} free_steps={free_steps}", flush=True
        )
        if first_command_step is None:
            raise RuntimeError("SONIC deploy produced no low-level command")
        if free_steps < 400 or final.z < 0.55 or minimum_free_z < 0.45:
            raise RuntimeError("SONIC failed the untethered Isaac stability test")
        if abs(args.vx) > 0.01 and final.x * args.vx < 0.1:
            raise RuntimeError(
                "SONIC remained stable but did not track the requested forward velocity"
            )
        if route_goal is not None:
            error = math.hypot(final.x - route_goal.x, final.y - route_goal.y)
            if not route_reached or error > 0.5:
                raise RuntimeError(
                    f"SONIC did not complete the semantic route (error={error:.3f} m)"
                )
            print(
                f"ISAAC_ROUTE_COMPLETE goal={args.route_goal} error={error:.3f} "
                f"frames={recorder.frames if recorder else 0}", flush=True
            )
    finally:
        if 'recorder' in locals() and recorder is not None:
            recorder.close()
        if bridge is not None:
            bridge.close()
        if client is not None:
            client.close()
        for process in (deploy, sidecar):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, 15)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, 9)
        sidecar_log.close()
        deploy_log.close()


try:
    print("CONTACT_SMOKE_MAIN", flush=True)
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
