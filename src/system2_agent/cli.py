from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import System2Agent
from .model import OpenAICompatibleModel
from .modules import (
    DryRunManipulationBackend,
    DryRunNavigationBackend,
    ManipulationModule,
    NavigationModule,
    SemanticMapModule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dry-run robot System-2 agent")
    parser.add_argument("mission")
    parser.add_argument("--model", required=True, help="For example openai/MODEL or deepseek/MODEL")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--map",
        type=Path,
        default=Path("examples/locations.json"),
        help="JSON semantic map",
    )
    parser.add_argument("--max-model-calls", type=int, default=30)
    args = parser.parse_args()

    model = OpenAICompatibleModel.from_env(
        args.model, base_url=args.base_url, api_key_env=args.api_key_env
    )
    semantic_map = SemanticMapModule.from_json(args.map)
    modules = [
        semantic_map,
        NavigationModule(semantic_map, DryRunNavigationBackend(), requires_approval=False),
        ManipulationModule(DryRunManipulationBackend(), requires_approval=False),
    ]
    outcome = System2Agent(
        model, modules, max_model_calls=args.max_model_calls
    ).run(args.mission)
    print(json.dumps(outcome.__dict__, indent=2, default=list))


if __name__ == "__main__":
    main()
