from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from .agent import System2Agent
from .model import OpenAICompatibleModel
from .modules import CameraModule, NavigationModule, Pose3D, SemanticMapModule
from .navigation_core import AStarPlanner, GridMap, PathFollower, PlannedNavigationBackend
from .scene_bundle import SceneBundle
from .sim import ZmqJpegCamera
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
    parser.add_argument("--mission")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--with-vision", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--startup-seconds", type=float, default=8.0)
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--check", action="store_true")
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
    if bool(args.goal) == bool(args.mission):
        parser.error("provide exactly one of --goal or --mission")
    if args.mission and not args.model:
        parser.error("--mission requires --model")

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
    wrapper = SimWrapper(
        instantiate_g1_robot_model(),
        cfg.env_name,
        wbc_config,
        onscreen=args.viewer,
        offscreen=args.with_vision,
        enable_image_publish=args.with_vision,
    )
    if args.with_vision:
        wrapper.sim.start_image_publish_subprocess(cfg.mp_start_method, cfg.camera_port)
    wrapper.sim.start_as_thread()

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
            "--deploy-auto-start-control",
        ],
        cwd=robot_class,
    )
    camera = None
    try:
        time.sleep(args.startup_seconds)
        if deploy.poll() is not None:
            raise RuntimeError(f"SONIC deploy exited early with code {deploy.returncode}")
        bridge.start()
        scene = SceneBundle.from_json(args.scene)
        semantic = SemanticMapModule.from_json(scene.semantic_map)
        navigation = PlannedNavigationBackend(
            AStarPlanner(GridMap.from_json(scene.navigation_grid)),
            PathFollower(max_speed=0.35, max_yaw_rate=0.5),
            bridge,
            realtime=args.realtime,
            timeout_s=180.0,
        )
        if args.goal:
            print(json.dumps(navigation.navigate(semantic.resolve(args.goal)), indent=2))
            return
        modules: list[object] = [
            semantic,
            NavigationModule(semantic, navigation, requires_approval=False),
        ]
        if args.with_vision:
            camera = ZmqJpegCamera(camera="ego_view")
            modules.append(CameraModule(camera))
        model = OpenAICompatibleModel.from_env(
            args.model, base_url=args.base_url, api_key_env=args.api_key_env
        )
        outcome = System2Agent(model, modules).run(args.mission)
        print(json.dumps(outcome.__dict__, indent=2, default=list))
    finally:
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


def _add_robot_class(path: Path) -> None:
    if not (path / "robot" / "sonic_variants.py").exists():
        return
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


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
