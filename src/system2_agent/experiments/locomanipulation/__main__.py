from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from ...agent import System2Agent
from ...model import OpenAICompatibleModel
from ...modules.camera import CameraModule
from .control import ManipulationModule, SYSTEM_PROMPT
from .runtime import Runtime
from .scene import TASKS, build_scene


def main() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    workspace = Path(__file__).resolve().parents[5]
    parser = argparse.ArgumentParser(description="G1 Dex-1 native SONIC 1.1 manipulation experiments (simulation only)")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--dex1-assets", type=Path, default=workspace / "g1_sim_pipeline/models/dex1_urdf")
    parser.add_argument(
        "--sonic-g1-assets", type=Path,
        default=workspace / "GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1",
        help="SONIC authoritative G1 model directory containing g1_29dof_with_hand.xml",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/locomanipulation"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="Compile scene and render the three allowed views; no controller or task success claim")
    mode.add_argument("--check", action="store_true", help="Validate model, task, cameras and local SONIC prerequisites")
    mode.add_argument("--connect-sonic", action="store_true", help="Connect to local DDS sidecar + official SONIC 1.1 deploy; physics execution")
    agent = parser.add_mutually_exclusive_group()
    agent.add_argument("--model", help="Existing exp-agent model-provider route")
    agent.add_argument("--actions", type=Path, help="JSON list of scripted move_to actions (diagnostic, NOT a vision-agent result)")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--max-model-calls", type=int, default=30)
    parser.add_argument("--startup-seconds", type=float, default=60)
    parser.add_argument("--episode-seconds", type=float, default=300)
    parser.add_argument("--record-video", action="store_true",
                        help="Record a continuous head/left-wrist/right-wrist evidence video")
    parser.add_argument("--record-fps", type=float, default=5)
    args = parser.parse_args()
    if args.connect_sonic and args.model is None and args.actions is None:
        args.model = os.getenv("SYSTEM2_MODEL")
    if args.connect_sonic and not (args.model or args.actions):
        parser.error("--connect-sonic requires --model or --actions")
    if not args.connect_sonic and (args.model or args.actions):
        parser.error("--model and --actions require --connect-sonic")
    if args.startup_seconds <= 0 or args.episode_seconds <= 0 or args.max_model_calls < 1 or not 1 <= args.record_fps <= 15:
        parser.error("timeouts and model-call budget must be positive")
    if args.connect_sonic and platform.system() != "Linux":
        parser.error("official SONIC execution requires Linux/CUDA/TensorRT; use --preview on this machine")
    if not args.check and args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory to preserve previous experiment evidence")
    scene = build_scene(args.dex1_assets, args.task, args.sonic_g1_assets)
    if args.check:
        deploy = workspace / "GR00T-WholeBodyControl/gear_sonic_deploy"
        required = [deploy / p for p in ("policy/sonic_v1_1/model_encoder.onnx", "policy/sonic_v1_1/model_decoder.onnx",
                                         "policy/sonic_v1_1/observation_config.yaml", "planner/target_vel/V2/planner_sonic.onnx")]
        problems = [f"missing {p}" for p in required if not p.exists()]
        if platform.system() != "Linux":
            problems.append("official SONIC runtime requires Linux/CUDA/TensorRT")
        print(json.dumps({"scene_ready": True, "cameras": scene.model.ncam, "actuators": scene.model.nu,
                          "sonic_model_source": scene.sonic_model_source,
                          "sonic_files_present": not any(not p.exists() for p in required),
                          "problems": problems, "controller_execution_validated": False}, indent=2))
        return
    runtime = Runtime(scene, output=args.output, startup_s=args.startup_seconds,
                      max_episode_s=args.episode_seconds, record_video=args.record_video,
                      record_fps=args.record_fps)
    report = {}
    try:
        if args.preview:
            runtime.capture()
            report = {"mode": "preview_only", "task": args.task, "scene_compiled": True,
                      "controller_ran": False, "task_success": None,
                      "camera_names": runtime.proprioception()["camera_names"]}
        else:
            # Validate model credentials before starting physics, without an API call.
            model = OpenAICompatibleModel.from_env(args.model, base_url=args.base_url, api_key_env=args.api_key_env) if args.model else None
            actions = json.loads(args.actions.read_text()) if args.actions else None
            if actions is not None and not isinstance(actions, list):
                raise ValueError("--actions must contain a JSON list")
            runtime.start()
            if model is not None:
                outcome = System2Agent(model, [ManipulationModule(runtime), CameraModule(runtime)],
                                       max_model_calls=args.max_model_calls, system_prompt=SYSTEM_PROMPT).run(TASKS[args.task])
                report["agent_outcome"] = asdict(outcome)
                report["mode"] = "vision_agent"
            else:
                for action in actions:
                    runtime.execute(action)
                    runtime.capture()
                report["mode"] = "scripted_diagnostic"
            time.sleep(2.2)  # Score a final stable hold/release, not a transient pose.
            runtime.check_health()
            report.update(runtime.report())
    except Exception as exc:
        report.update(runtime.report())
        report["error"] = str(exc)
        raise
    finally:
        runtime.close()
        # Private scoring is written only for the experimenter, never fed to the VLM.
        (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"actions", "agent_outcome"}}, indent=2))


if __name__ == "__main__":
    main()
