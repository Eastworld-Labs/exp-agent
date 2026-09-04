from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .agent import System2Agent
from .model import OpenAICompatibleModel
from .modules import CameraModule, NavigationModule, Pose3D, SemanticMapModule
from .navigation_core import (
    GridMap,
    PlannedNavigationBackend,
    RegulatedTrajectoryFollower,
    SmoothTrajectoryPlanner,
    VelocityCommand,
)
from .scene_bundle import SceneBundle
from .scene_loader import SceneLoader
from .sim import ZmqJpegCamera
from .sim.g1_mujoco import MuJoCoHeadCamera
from .sim.head_camera import HeadCameraBackend, HeadCameraStream
from .sim_cli import (
    add_head_camera_arguments,
    frame_publisher_from_args,
    head_camera_from_args,
)
from .sonic_bridge import SonicZmqBase


def main() -> None:
    package_root = Path(__file__).resolve().parents[2]
    workspace = package_root.parent
    parser = argparse.ArgumentParser(
        description="Run System-2 + NVIDIA SONIC v1.1 + official G1 MuJoCo sim2sim"
    )
    parser.add_argument("--groot-wbc", type=Path, default=workspace / "GR00T-WholeBodyControl")
    parser.add_argument("--robot-class", type=Path, default=workspace / "robot_class")
    parser.add_argument("--sonic-variant", default="sonic_v1_1")
    parser.add_argument("--scene", type=Path, default=package_root / "examples" / "sim_scene.json")
    parser.add_argument("--goal")
    parser.add_argument(
        "--route",
        help="Comma-separated semantic destinations to preplan and execute as one route",
    )
    parser.add_argument("--mission")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--with-vision", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--record", type=Path, help="Write an overview MP4 of the SONIC run")
    parser.add_argument("--record-fps", type=float, default=30.0)
    parser.add_argument("--record-width", type=int, default=1280)
    parser.add_argument("--record-height", type=int, default=720)
    parser.add_argument("--result-json", type=Path, help="Write measured navigation results")
    parser.add_argument("--startup-seconds", type=float, default=60.0)
    parser.add_argument("--startup-support-seconds", type=float, default=12.0)
    parser.add_argument("--navigation-timeout", type=float, default=180.0)
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--check", action="store_true")
    add_head_camera_arguments(parser)
    args = parser.parse_args()

    repo = args.groot_wbc.expanduser().resolve()
    robot_class = args.robot_class.expanduser().resolve()
    _add_robot_class(robot_class)
    try:
        from robot.sonic_variants import normalize_sonic_variant, sonic_variant_spec

        variant = normalize_sonic_variant(args.sonic_variant)
        if variant == "auto":
            raise ValueError("--sonic-variant must resolve to a concrete bundle")
        variant_spec = sonic_variant_spec(variant)
    except (ImportError, ValueError) as exc:
        if args.check:
            print(json.dumps({"ready": False, "problems": [str(exc)]}, indent=2))
            return
        raise SystemExit(f"robot_class SONIC integration is unavailable: {exc}") from exc
    problems = _check(repo, robot_class, variant_spec)
    if args.check:
        print(json.dumps({"ready": not problems, "problems": problems}, indent=2))
        return
    if problems:
        raise SystemExit("SONIC sim2sim is not ready:\n- " + "\n- ".join(problems))
    if sum(bool(value) for value in (args.goal, args.route, args.mission)) != 1:
        parser.error("provide exactly one of --goal, --route, or --mission")
    if args.mission and not args.model:
        parser.error("--mission requires --model")

    # Load the scene before constructing the upstream simulator.  SceneBundle's
    # optional MJCF is the physical scene (walls, doors, furniture); the grid is
    # its planning representation.  Previously the official SONIC path always
    # loaded the upstream empty-floor scene and used only the planning overlay.
    scene = SceneBundle.from_json(args.scene)
    scene_grid = GridMap.from_json(scene.navigation_grid)
    if scene.initial_pose is not None:
        initial_x, initial_y, _ = scene.initial_pose
        initial_clearance = scene_grid.clearance(initial_x, initial_y)
        if initial_clearance < scene.navigation_footprint_radius_m:
            raise SystemExit(
                "scene initial_pose is not collision-safe "
                f"(clearance={initial_clearance:.3f} m, required="
                f"{scene.navigation_footprint_radius_m:.3f} m)"
            )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.scripts.run_sim_loop import SimWrapper
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory

    cfg = SimLoopConfig(
        enable_onscreen=args.viewer,
        enable_offscreen=args.with_vision,
        enable_image_publish=args.with_vision,
    )
    wbc_config = cfg.load_wbc_yaml()
    wbc_config["ENV_NAME"] = cfg.env_name
    robot_scene = repo / wbc_config["ROBOT_SCENE"]
    head_camera = head_camera_from_args(args)
    loaded_physics = SceneLoader(robot_scene).load(scene, head_camera=head_camera)
    wbc_config["ROBOT_SCENE"] = str(loaded_physics.model_path)
    # Keep the upstream support available during deploy's three-second posture
    # ramp. It is released before navigation, so locomotion remains untethered.
    wbc_config["ENABLE_ELASTIC_BAND"] = True
    wrapper = SimWrapper(
        instantiate_g1_robot_model(),
        cfg.env_name,
        wbc_config,
        onscreen=args.viewer,
        offscreen=cfg.enable_offscreen,
        enable_image_publish=cfg.enable_image_publish,
    )
    support = wrapper.sim.sim_env.elastic_band
    if scene.initial_pose is not None:
        # Scene-defined spawn is applied to simulator state, not baked into the
        # robot or external MJCF. Keep the temporary startup support directly
        # above that same point so it cannot pull the robot back to map origin.
        import mujoco

        initial_x, initial_y, initial_yaw = scene.initial_pose
        sim_env = wrapper.sim.sim_env
        initial_quaternion = (
            math.cos(initial_yaw / 2.0),
            0.0,
            0.0,
            math.sin(initial_yaw / 2.0),
        )
        # Fall recovery uses mj_resetData(), which restores model.qpos0. Store
        # the scene spawn there as well as in live state so an early startup
        # reset cannot silently return the robot to the robot MJCF's (0, 0).
        sim_env.mj_model.qpos0[0:2] = (initial_x, initial_y)
        sim_env.mj_model.qpos0[3:7] = initial_quaternion
        sim_env.mj_data.qpos[0:2] = (initial_x, initial_y)
        sim_env.mj_data.qpos[3:7] = initial_quaternion
        sim_env.mj_data.qvel[:6] = 0.0
        support.point[0:2] = (initial_x, initial_y)
        mujoco.mj_forward(sim_env.mj_model, sim_env.mj_data)
    if args.record is not None:
        wrapper.sim.sim_env.mj_model.vis.global_.offwidth = args.record_width
        wrapper.sim.sim_env.mj_model.vis.global_.offheight = args.record_height
    if args.with_vision and head_camera is None:
        wrapper.sim.start_image_publish_subprocess(cfg.mp_start_method, cfg.camera_port)
    wrapper.sim.start_as_thread()
    head_stream = None
    publisher = frame_publisher_from_args(args)
    if publisher is not None:
        assert head_camera is not None
        # Renders the composed model's head camera from its own thread while
        # the upstream sim thread steps -- the same arrangement as the video
        # recorder below, which upstream already relies on.
        head_stream = HeadCameraStream(
            MuJoCoHeadCamera(
                wrapper.sim.sim_env.mj_model, wrapper.sim.sim_env.mj_data, spec=head_camera
            ),
            publisher,
            hz=args.stream_hz,
            source="sim:sonic",
        )
        head_stream.start()
        print(
            f"[head camera] streaming to {publisher.broker} as {publisher.robot_id}; "
            f"dashboard: ?robot={publisher.robot_id}",
            flush=True,
        )

    def sim_pose() -> Pose3D:
        qpos = wrapper.sim.sim_env.mj_data.qpos
        w, x, y, z = [float(v) for v in qpos[3:7]]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return Pose3D(float(qpos[0]), float(qpos[1]), float(qpos[2]), yaw)

    bridge = SonicZmqBase(sim_pose, settle_s=0.0, sonic_variant=variant)
    deploy = subprocess.Popen(
        [
            sys.executable,
            str(robot_class / "scripts" / "setup_sonic_deploy.py"),
            "--wbc-dir",
            str(repo),
            "--launch-deploy",
            "--variant",
            variant,
            "--robot-interface",
            "sim",
            "--deploy-input-type",
            "zmq_manager",
            "--deploy-planner",
            "planner/target_vel/V2/planner_sonic.onnx",
        ],
        cwd=robot_class,
        stdin=subprocess.PIPE,
        text=True,
    )
    assert deploy.stdin is not None
    deploy.stdin.write("\n")
    deploy.stdin.close()
    camera = None
    recorder = None
    try:
        _wait_for_deploy(deploy, timeout_s=args.startup_seconds)
        # Start directly in the ZMQ manager's default planner mode while the
        # upstream lift band holds the base at its intended 1 m anchor. Using
        # auto-start first and switching modes later initialized the planner
        # from a contacted/tilted state and produced an invalid base reference.
        support_deadline = time.monotonic() + args.startup_support_seconds
        while time.monotonic() < support_deadline:
            bridge.start(planner=True)
            bridge.command_velocity(VelocityCommand(0.0, 0.0, 0.0), 0.02)
            time.sleep(0.05)
        support.enable = False
        # Verify the learned planner/control pair can hold untethered before
        # handing it a route. This is the programmatic equivalent of pressing
        # `]` while lifted and then `9` in the official quick-start.
        planner_stable_deadline = time.monotonic() + 2.0
        minimum_preflight_height = float("inf")
        while time.monotonic() < planner_stable_deadline:
            bridge.command_velocity(VelocityCommand(0.0, 0.0, 0.0), 0.02)
            minimum_preflight_height = min(minimum_preflight_height, sim_pose().z)
            time.sleep(0.02)
        preflight_pose = sim_pose()
        if preflight_pose.z < 0.55:
            raise RuntimeError(
                "SONIC failed untethered planner preflight "
                f"(final_z={preflight_pose.z:.3f}, min_z={minimum_preflight_height:.3f})"
            )
        if scene.initial_pose is not None:
            initial_x, initial_y, _ = scene.initial_pose
            spawn_error = math.hypot(preflight_pose.x - initial_x, preflight_pose.y - initial_y)
            if spawn_error > 0.5:
                raise RuntimeError(
                    "simulator did not preserve the scene initial_pose through startup "
                    f"(position error={spawn_error:.3f} m)"
                )
        semantic = SemanticMapModule.from_json(scene.semantic_map)
        grid = scene_grid
        planner = SmoothTrajectoryPlanner(
            grid,
            footprint_radius_m=scene.navigation_footprint_radius_m,
        )
        navigation = PlannedNavigationBackend(
            planner,
            RegulatedTrajectoryFollower(
                planner.grid,
                # Rotate before translating through large heading errors. The
                # bridge slews SONIC's absolute facing vector at this bounded
                # yaw rate instead of making a discontinuous direction jump.
                max_speed=0.30,
                # SONIC's SLOW_WALK reference barely translates below ~0.2
                # m/s even though it remains dynamically active.
                minimum_speed=0.22,
                max_yaw_rate=0.3,
                holonomic=False,
                # Finish most of a large turn before translating. The
                # world-frame movement vector then stays tangent to the local
                # collision-checked look-ahead segment while facing converges.
                turn_in_place_threshold=0.50,
                turn_in_place=True,
                # A semantic navigation goal is a position, not a precision
                # docking pose. A separate docking skill should own the final
                # 180-degree alignment requested by some map annotations.
                align_final_yaw=False,
            ),
            bridge,
            realtime=args.realtime,
            timeout_s=args.navigation_timeout,
            stuck_timeout_s=12.0,
        )
        route_names = (
            [args.goal]
            if args.goal
            else [name.strip() for name in args.route.split(",") if name.strip()]
            if args.route
            else []
        )
        planned_path: list[Pose3D] = []
        planned_segments: list[list[Pose3D]] = []
        planned_metrics: list[dict[str, object]] = []
        if route_names:
            cursor = sim_pose()
            planned_path.append(cursor)
            for name in route_names:
                destination = semantic.resolve(name)
                destination_clearance = grid.clearance(destination.x, destination.y)
                required_destination_clearance = scene.navigation_footprint_radius_m + 0.15
                if destination_clearance < required_destination_clearance:
                    raise RuntimeError(
                        f"semantic destination {name!r} is too close to an obstacle "
                        f"for a dynamic humanoid stop (clearance={destination_clearance:.3f} m, "
                        f"required={required_destination_clearance:.3f} m)"
                    )
                segment = navigation.planner.plan(cursor, destination)
                planned_segments.append(segment)
                planned_metrics.append(dict(navigation.planner.last_plan_metrics))
                planned_path.extend(segment[1:])
                cursor = destination
        if args.record is not None:
            recorder = _VideoRecorder(
                args.record,
                model=wrapper.sim.sim_env.mj_model,
                data=wrapper.sim.sim_env.mj_data,
                fps=args.record_fps,
                width=args.record_width,
                height=args.record_height,
                gaussian_splat=scene.gaussian_splat,
                gaussian_alignment=scene.gaussian_alignment,
                navigation_grid=navigation.planner.grid,
                planned_path=planned_path,
                pose_provider=sim_pose,
            )
            # Initializing the EGL renderer can take longer than SONIC's
            # one-second planner watchdog. Keep publishing a stationary
            # target while the recorder creates its first frame; otherwise a
            # perfectly stable robot can drop to IDLE and fall before the
            # first navigation command is issued.
            recorder.start(
                on_wait=lambda: bridge.command_velocity(
                    VelocityCommand(0.0, 0.0, 0.0), 0.05
                )
            )
        if route_names:
            # Preserve semantic-stop ordering. A route can cross itself (this
            # demo returns through the foyer); treating every leg as one path
            # lets nearest-point lookahead jump to a later crossing.
            stop_results: list[dict[str, object]] = []
            total_commands = 0
            for name, segment, metrics in zip(route_names, planned_segments, planned_metrics):
                navigation.planner.last_plan_metrics = metrics
                stop_result = dict(navigation.navigate_path(segment))
                measured = sim_pose()
                destination = semantic.resolve(name)
                stop_result["semantic_destination"] = name
                stop_result["position_error_m"] = round(
                    math.hypot(measured.x - destination.x, measured.y - destination.y), 3
                )
                stop_results.append(stop_result)
                total_commands += int(stop_result.get("command_count", 0))
                if stop_result.get("state") != "succeeded":
                    break
            result = {
                "state": (
                    "succeeded"
                    if len(stop_results) == len(route_names)
                    and all(stop.get("state") == "succeeded" for stop in stop_results)
                    else "failed"
                ),
                "locomotion": bridge.name,
                "semantic_route": route_names,
                "completed_stops": sum(
                    stop.get("state") == "succeeded" for stop in stop_results
                ),
                "command_count": total_commands,
                "pose": sim_pose().as_json(),
                "stops": stop_results,
                "preplanned_path": [pose.as_json() for pose in planned_path],
            }
            if args.result_json is not None:
                args.result_json.parent.mkdir(parents=True, exist_ok=True)
                args.result_json.write_text(json.dumps(result, indent=2) + "\n")
            if recorder is not None:
                time.sleep(2.0)
            print(json.dumps(result, indent=2))
            return
        modules: list[object] = [
            semantic,
            NavigationModule(semantic, navigation, requires_approval=False),
        ]
        if args.with_vision:
            if head_camera is not None:
                # The robot's own two previews, colour then range, as the
                # mission service shows them on hardware.
                camera = HeadCameraBackend(
                    MuJoCoHeadCamera(
                        wrapper.sim.sim_env.mj_model,
                        wrapper.sim.sim_env.mj_data,
                        spec=head_camera,
                    )
                )
            else:
                camera = ZmqJpegCamera(camera="ego_view")
            modules.append(CameraModule(camera))
        model = OpenAICompatibleModel.from_env(
            args.model, base_url=args.base_url, api_key_env=args.api_key_env
        )
        outcome = System2Agent(model, modules).run(args.mission)
        print(json.dumps(outcome.__dict__, indent=2, default=list))
    finally:
        if recorder is not None:
            recorder.close()
            print(json.dumps(recorder.summary(), indent=2))
        if head_stream is not None:
            head_stream.close()
            print(json.dumps({"head_camera_stream": head_stream.summary()}, indent=2))
        if camera is not None:
            camera.close()
        bridge.close()
        deploy.terminate()
        try:
            deploy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            deploy.kill()
            deploy.wait()
        wrapper.sim.close()
        if wrapper.sim.sim_thread is not None:
            wrapper.sim.sim_thread.join(timeout=5)
        loaded_physics.close()


def _add_robot_class(path: Path) -> None:
    if not (path / "robot" / "sonic_variants.py").exists():
        return
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _wait_for_deploy(deploy: subprocess.Popen[str], *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if deploy.poll() is not None:
            raise RuntimeError(f"SONIC deploy exited early with code {deploy.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.1)
            if probe.connect_ex(("127.0.0.1", 5557)) == 0:
                return
        time.sleep(0.25)
    raise TimeoutError(f"SONIC deploy was not ready after {timeout_s:.1f}s")


class _VideoRecorder:
    def __init__(
        self,
        path: Path,
        *,
        model: object,
        data: object,
        fps: float,
        width: int,
        height: int,
        gaussian_splat: Path | None = None,
        gaussian_alignment: tuple[tuple[float, ...], ...] | None = None,
        navigation_grid: GridMap | None = None,
        planned_path: list[Pose3D] | None = None,
        pose_provider: Callable[[], Pose3D] | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.model = model
        self.data = data
        self.fps = fps
        self.width = width
        self.height = height
        self.gaussian_splat = gaussian_splat
        self.gaussian_alignment = gaussian_alignment
        self.navigation_grid = navigation_grid
        self.planned_path = planned_path or []
        self.pose_provider = pose_provider
        self.frames = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sonic-video", daemon=True)

    def start(self, on_wait: Callable[[], None] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        # The first gsplat use JIT-compiles its CUDA extension.  On a fresh
        # machine this can take several minutes; subsequent starts are instant.
        deadline = time.monotonic() + 600.0
        while not self._ready.wait(timeout=0.05):
            if time.monotonic() >= deadline:
                raise RuntimeError(self.error or "video recorder did not initialize")
            if on_wait is not None:
                on_wait()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def summary(self) -> dict[str, object]:
        return {
            "video": str(self.path),
            "frames": self.frames,
            "fps": self.fps,
            "error": self.error,
            "gaussian_splat": None if self.gaussian_splat is None else str(self.gaussian_splat),
        }

    def _run(self) -> None:
        import cv2
        import mujoco
        import numpy as np

        renderer = None
        writer = None
        try:
            # EGL contexts are thread-affine. Construct, render, and destroy the
            # recorder's context entirely inside this worker thread.
            self.model.vis.global_.fovy = 71.51
            renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            # Start near the route. During recording this camera tracks the
            # robot closely enough to show gait quality; the map overlay keeps
            # the entire collision-checked route visible at the same time.
            if self.planned_path:
                route_x = np.asarray([pose.x for pose in self.planned_path])
                route_y = np.asarray([pose.y for pose in self.planned_path])
                camera.lookat[:] = (
                    float((route_x.min() + route_x.max()) * 0.5),
                    float((route_y.min() + route_y.max()) * 0.5),
                    0.55,
                )
                route_span = max(float(np.ptp(route_x)), float(np.ptp(route_y)))
            else:
                camera.lookat[:] = (0.25, 0.45, 0.8)
                route_span = 1.0
            camera.distance = 1.1
            camera.azimuth = 0.0
            camera.elevation = -20.0
            gaussian = None
            world_t_gs = None
            gaussian_camera = "head_camera"
            foreground_geom_ids: np.ndarray | None = None
            if self.gaussian_splat is not None:
                cuda_root = (
                    Path(sys.prefix)
                    / "lib"
                    / f"python{sys.version_info.major}.{sys.version_info.minor}"
                    / "site-packages"
                    / "nvidia"
                    / "cu13"
                )
                if (cuda_root / "bin" / "nvcc").exists():
                    os.environ.setdefault("CUDA_HOME", str(cuda_root))
                    os.environ.setdefault(
                        "TORCH_EXTENSIONS_DIR", "/tmp/exp_agent_torch_extensions_cu130"
                    )
                    os.environ.setdefault("MAX_JOBS", "4")
                    os.environ["PATH"] = (
                        f"{Path(sys.prefix) / 'bin'}:{cuda_root / 'bin'}:"
                        f"{os.environ.get('PATH', '')}"
                    )
                from mugs.sensors import GaussianSensor, GaussianSensorConfig

                gaussian = GaussianSensor(
                    GaussianSensorConfig(
                        width=self.width,
                        height=self.height,
                        background_ply_path=self.gaussian_splat,
                        render_mode="3dgs_only",
                        cache_background=False,
                    )
                )
                world_t_gs = np.asarray(self.gaussian_alignment or np.eye(4), dtype=np.float32)
                # Composite every body-backed geom over the visual-only splat:
                # this includes the robot and SimFoundry's movable/fixed props.
                # Worldbody collision meshes remain invisible so they do not
                # double-render the photorealistic background.
                foreground_geom_ids = np.flatnonzero(
                    np.asarray(self.model.geom_bodyid) != 0
                ).astype(np.int32)

            def render_view(camera_spec: object) -> np.ndarray:
                renderer.update_scene(self.data, camera=camera_spec)
                foreground = renderer.render().copy()
                if gaussian is None:
                    return foreground
                gl_camera = renderer.scene.camera[0]
                eye = np.asarray(gl_camera.pos, dtype=np.float32).copy()
                forward = np.asarray(gl_camera.forward, dtype=np.float32).copy()
                forward /= np.linalg.norm(forward)
                camera_up = np.asarray(gl_camera.up, dtype=np.float32).copy()
                camera_up /= np.linalg.norm(camera_up)
                z_back = -forward
                x_right = np.cross(forward, camera_up)
                x_right /= np.linalg.norm(x_right)
                y_up = np.cross(z_back, x_right)
                camera_rotation = np.column_stack((x_right, y_up, z_back))
                # MuJoCo/OpenGL camera Y is up while gsplat/OpenCV camera Y is
                # down. MuGS flips Z internally, so pre-flip Y here.
                camera_rotation[:, 1] *= -1.0
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render()[:, :, 0].astype(np.int32)
                renderer.disable_segmentation_rendering()
                assert world_t_gs is not None and foreground_geom_ids is not None
                gs_rotation = world_t_gs[:3, :3]
                if isinstance(camera_spec, str):
                    camera_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_spec
                    )
                    fovy_degrees = float(self.model.cam_fovy[camera_id])
                else:
                    fovy_degrees = float(self.model.vis.global_.fovy)
                focal_px = 0.5 * self.height / math.tan(math.radians(fovy_degrees) * 0.5)
                params = {
                    "position": gs_rotation @ eye + world_t_gs[:3, 3],
                    "rotation": gs_rotation @ camera_rotation,
                    "fx": focal_px,
                    "fy": focal_px,
                    "width": self.width,
                    "height": self.height,
                }
                background = gaussian.render(
                    self.model, self.data, gaussian_camera, camera_params=params
                )
                mask = np.isin(segmentation, foreground_geom_ids)[:, :, None]
                return np.where(mask, foreground, background).astype(np.uint8)

            def add_map_overlay(frame: np.ndarray) -> None:
                grid = self.navigation_grid
                if grid is None:
                    return
                side = max(80, min(self.width, self.height) // 4)
                panel = np.full((side, side, 3), 28, dtype=np.uint8)
                sx, sy = side / grid.width, side / grid.height
                for cell_x, cell_y in grid.occupied:
                    x0, x1 = round(cell_x * sx), round((cell_x + 1) * sx)
                    y0, y1 = round(side - (cell_y + 1) * sy), round(side - cell_y * sy)
                    cv2.rectangle(panel, (x0, y0), (x1, y1), (85, 85, 85), -1)

                def pixel(pose: Pose3D) -> tuple[int, int]:
                    cell_x = (pose.x - grid.origin_x) / grid.resolution
                    cell_y = (pose.y - grid.origin_y) / grid.resolution
                    return round(cell_x * sx), round(side - cell_y * sy)

                if len(self.planned_path) > 1:
                    points = np.asarray([pixel(pose) for pose in self.planned_path], np.int32)
                    cv2.polylines(panel, [points], False, (255, 220, 30), 3, cv2.LINE_AA)
                    for point in points[1:]:
                        cv2.circle(panel, tuple(point), 4, (0, 210, 255), -1)
                if self.pose_provider is not None:
                    pose = self.pose_provider()
                    center = pixel(pose)
                    tip = (
                        round(center[0] + 13 * math.cos(pose.yaw)),
                        round(center[1] - 13 * math.sin(pose.yaw)),
                    )
                    cv2.circle(panel, center, 6, (30, 230, 30), -1)
                    cv2.arrowedLine(panel, center, tip, (30, 230, 30), 2, tipLength=0.35)
                cv2.putText(
                    panel,
                    "NAVIGATION COLLISION MAP",
                    (7, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y0 = self.height - side - 10
                frame[y0 : y0 + side, 10 : 10 + side] = panel
                cv2.rectangle(frame, (10, y0), (10 + side, y0 + side), (255, 255, 255), 2)

            writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"could not open video writer: {self.path}")
            next_frame = time.monotonic()
            while not self._stop.is_set():
                if self.pose_provider is not None:
                    tracked = self.pose_provider()
                    camera.lookat[:] = (tracked.x, tracked.y, max(0.55, tracked.z * 0.75))
                    # A splat is only valid from viewpoints inside its captured
                    # free space. Prefer a camera behind the walking direction,
                    # then try side/front alternatives if a wall occupies that
                    # point. This avoids placing an orbit camera through a wall.
                    preferred = tracked.yaw
                    for angle in (
                        preferred,
                        preferred + math.pi / 2,
                        preferred - math.pi / 2,
                        preferred + math.pi,
                    ):
                        horizontal_distance = camera.distance * math.cos(
                            math.radians(camera.elevation)
                        )
                        eye_x = tracked.x - horizontal_distance * math.cos(angle)
                        eye_y = tracked.y - horizontal_distance * math.sin(angle)
                        if self.navigation_grid is None or (
                            self.navigation_grid.segment_is_free(
                                (eye_x, eye_y), (tracked.x, tracked.y)
                            )
                            and self.navigation_grid.clearance(eye_x, eye_y) >= 0.15
                        ):
                            camera.azimuth = math.degrees(angle)
                            break
                image = cv2.cvtColor(render_view(camera), cv2.COLOR_RGB2BGR)
                ego = cv2.cvtColor(render_view("head_camera"), cv2.COLOR_RGB2BGR)
                inset_width = max(110, round(self.width * 0.30))
                inset_height = round(inset_width * self.height / self.width)
                ego = cv2.resize(ego, (inset_width, inset_height), interpolation=cv2.INTER_AREA)
                inset_x, inset_y = self.width - inset_width - 10, 10
                image[inset_y : inset_y + inset_height, inset_x : inset_x + inset_width] = ego
                cv2.rectangle(
                    image,
                    (inset_x, inset_y),
                    (inset_x + inset_width, inset_y + inset_height),
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    image,
                    "G1 HEAD CAMERA",
                    (inset_x + 8, inset_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                add_map_overlay(image)
                writer.write(image)
                self.frames += 1
                self._ready.set()
                next_frame += 1.0 / self.fps
                self._stop.wait(max(0.0, next_frame - time.monotonic()))
        except Exception as exc:
            self.error = str(exc)
            self._ready.set()
        finally:
            if writer is not None:
                writer.release()
            if renderer is not None:
                renderer.close()


def _check(repo: Path, robot_class: Path, variant_spec: object) -> list[str]:
    problems: list[str] = []
    if not sys.platform.startswith("linux"):
        problems.append("official SONIC deploy requires Ubuntu/Linux with NVIDIA CUDA + TensorRT")
    checkpoint = repo / "gear_sonic_deploy" / str(variant_spec.deploy_checkpoint)
    required = [
        robot_class / "scripts" / "setup_sonic_deploy.py",
        repo / "gear_sonic" / "scripts" / "run_sim_loop.py",
        repo / "gear_sonic_deploy" / "deploy.sh",
        Path(f"{checkpoint}_encoder.onnx"),
        Path(f"{checkpoint}_decoder.onnx"),
        repo / "gear_sonic_deploy" / str(variant_spec.deploy_observation_config),
        repo / "gear_sonic_deploy" / "planner" / "target_vel" / "V2" / "planner_sonic.onnx",
        repo / ".venv_sim" / "bin" / "python",
    ]
    problems.extend(f"missing {path}" for path in required if not path.exists())
    return problems


if __name__ == "__main__":
    main()
