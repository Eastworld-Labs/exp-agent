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
        "--viewer", action="store_true", help="Show the Isaac Sim window (Isaac backend only)"
    )
    parser.add_argument(
        "--no-local-depth",
        action="store_true",
        help="Disable Isaac head-depth local obstacle observations",
    )
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--max-model-calls", type=int, default=30)
    args = parser.parse_args()
    if bool(args.goal) == bool(args.mission):
        parser.error("provide exactly one of --goal or --mission")
    if args.mission and not args.model:
        parser.error("--mission requires --model")

    scene = SceneBundle.from_json(args.scene)
    semantic_map = SemanticMapModule.from_json(scene.semantic_map)
    grid = GridMap.from_json(scene.navigation_grid)
    environment = create_simulation_environment(
        args.backend,
        scene,
        workspace=root.parent,
        with_vision=args.with_vision,
        camera=args.camera,
        splat=args.splat,
        headless=not args.viewer,
        isaac_local_depth=not args.no_local_depth,
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


if __name__ == "__main__":
    main()
