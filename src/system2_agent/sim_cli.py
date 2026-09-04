from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import System2Agent
from .model import OpenAICompatibleModel
from .modules import CameraModule, NavigationModule, Pose3D, SemanticMapModule
from .navigation_core import (
    GridMap,
    PlannedNavigationBackend,
    RegulatedTrajectoryFollower,
    SmoothTrajectoryPlanner,
)
from .scene_bundle import SceneBundle
from .sim import create_simulation_environment
from .sim.head_camera import SIM_ROBOT_ID, HeadCameraSpec, MqttFramePublisher


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run System-2 in a selectable MuJoCo or Isaac Sim environment"
    )
    parser.add_argument("--backend", choices=("mujoco", "isaac"), default="mujoco")
    parser.add_argument("--scene", type=Path, default=root / "examples" / "sim_scene.json")
    parser.add_argument("--goal", help="Run deterministic navigation without an LLM")
    parser.add_argument("--mission", help="Run the System-2 model agent")
    parser.add_argument("--model", help="openai/MODEL, google/MODEL, deepseek/MODEL, etc.")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--camera", default=None, help="MuJoCo camera name; omit for free camera")
    parser.add_argument("--with-vision", action="store_true")
    parser.add_argument("--splat", type=Path, help="Override scene's 3DGS PLY and use MuGS")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show the simulator window: MuJoCo's passive viewer, or the Isaac Sim "
        "window. Pair it with --realtime, otherwise the mission finishes faster "
        "than it can be watched.",
    )
    parser.add_argument(
        "--no-local-depth",
        action="store_true",
        help="Disable Isaac head-depth local obstacle observations",
    )
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--max-model-calls", type=int, default=30)
    add_head_camera_arguments(parser)
    args = parser.parse_args()
    if bool(args.goal) == bool(args.mission):
        parser.error("provide exactly one of --goal or --mission")
    if args.mission and not args.model:
        parser.error("--mission requires --model")

    scene = SceneBundle.from_json(args.scene)
    semantic_map = SemanticMapModule.from_json(scene.semantic_map)
    grid = GridMap.from_json(scene.navigation_grid)
    head_camera = head_camera_from_args(args)
    publisher = frame_publisher_from_args(args)
    environment = create_simulation_environment(
        args.backend,
        scene,
        workspace=root.parent,
        with_vision=args.with_vision,
        camera=args.camera,
        splat=args.splat,
        headless=not args.viewer,
        isaac_local_depth=not args.no_local_depth,
        head_camera=head_camera,
        stream=publisher,
        stream_hz=args.stream_hz,
    )
    if publisher is not None:
        print(
            f"[head camera] streaming to {publisher.broker} as {publisher.robot_id}; "
            f"dashboard: ?robot={publisher.robot_id}",
            flush=True,
        )
    base = environment.base
    try:
        if scene.initial_pose is not None:
            initial_x, initial_y, initial_yaw = scene.initial_pose
            if grid.clearance(initial_x, initial_y) < scene.navigation_footprint_radius_m:
                raise SystemExit(
                    "scene initial_pose is not collision-safe for the configured footprint"
                )
            getattr(base, "set_initial_pose")(
                Pose3D(initial_x, initial_y, yaw=initial_yaw)
            )
        planner = SmoothTrajectoryPlanner(
            grid, footprint_radius_m=scene.navigation_footprint_radius_m
        )
        backend = PlannedNavigationBackend(
            planner,
            RegulatedTrajectoryFollower(planner.grid),
            base,
            realtime=args.realtime,
            local_observer=environment.local_observer,
        )
        if args.goal:
            result = backend.navigate(semantic_map.resolve_navigation_goal(args.goal))
            print(json.dumps(result, indent=2))
            return

        modules: list[object] = [
            semantic_map,
            NavigationModule(semantic_map, backend, requires_approval=False),
        ]
        if environment.camera is not None:
            modules.append(CameraModule(environment.camera))
        model = OpenAICompatibleModel.from_env(
            args.model, base_url=args.base_url, api_key_env=args.api_key_env
        )
        outcome = System2Agent(
            model, modules, max_model_calls=args.max_model_calls
        ).run(args.mission)
        print(json.dumps(outcome.__dict__, indent=2, default=list))
    finally:
        environment.close()
        if environment.stream is not None:
            print(json.dumps({"head_camera_stream": environment.stream.summary()}, indent=2))


def add_head_camera_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by the simulation CLIs for the G1's head depth camera."""
    group = parser.add_argument_group("head camera")
    group.add_argument(
        "--no-head-camera",
        action="store_true",
        help="Do not attach the D455-like head depth camera to the robot",
    )
    group.add_argument(
        "--head-pitch-deg",
        type=float,
        default=0.0,
        help="Pitch the head camera down this many degrees (0 = facing forward; "
        "the real G1 bracket is about 48)",
    )
    group.add_argument(
        "--stream-mqtt",
        metavar="HOST[:PORT]",
        help="Publish the head camera's colour and range previews to this MQTT "
        "broker as the sim robot, on the topics the dashboard already watches",
    )
    group.add_argument(
        "--robot-id",
        default=SIM_ROBOT_ID,
        help="Robot id (and MQTT username) to publish as; open the dashboard "
        f"with ?robot=<id>. Default {SIM_ROBOT_ID}",
    )
    group.add_argument("--stream-hz", type=float, default=6.0, help="Preview rate")


def head_camera_from_args(args: argparse.Namespace) -> HeadCameraSpec | None:
    if args.no_head_camera:
        if args.stream_mqtt:
            raise SystemExit("--stream-mqtt needs the head camera; drop --no-head-camera")
        return None
    return HeadCameraSpec(pitch_down_deg=args.head_pitch_deg)


def parse_broker(value: str, default_port: int = 1883) -> tuple[str, int]:
    """``HOST[:PORT]`` -> (host, port)."""
    host, separator, port = value.strip().rpartition(":")
    if not separator:
        host, port = port, ""
    if not host:
        raise ValueError("broker host must not be empty")
    try:
        return host, int(port) if port else default_port
    except ValueError as exc:
        raise ValueError(f"broker port must be an integer: {value!r}") from exc


def frame_publisher_from_args(args: argparse.Namespace) -> MqttFramePublisher | None:
    if not args.stream_mqtt:
        return None
    host, port = parse_broker(args.stream_mqtt)
    return MqttFramePublisher(broker=host, port=port, robot_id=args.robot_id)


if __name__ == "__main__":
    main()
