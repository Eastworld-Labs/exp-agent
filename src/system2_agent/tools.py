from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from .types import Json, ToolResult


ToolKind = Literal["observation", "action", "terminal"]


class Module(Protocol):
    """A capability module that contributes tools and optional world context."""

    @property
    def name(self) -> str: ...

    def tools(self) -> Sequence["Tool"]: ...

    def snapshot(self) -> Json | None: ...


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Json
    handler: Callable[[Mapping[str, Any]], Any]
    kind: ToolKind = "observation"
    requires_approval: bool = False
    refresh_world: bool = False

    def schema(self) -> Json:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            _validate(arguments, self.parameters)
            return ToolResult(ok=True, data=self.handler(arguments))
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(ok=False, error=str(exc))


class ToolRegistry:
    def __init__(self, modules: Sequence[Module], core_tools: Sequence[Tool] = ()) -> None:
        self.modules = tuple(modules)
        self._tools: dict[str, Tool] = {}
        for tool in (*core_tools, *(t for module in modules for t in module.tools())):
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[Json]:
        return [tool.schema() for tool in self._tools.values()]

    def snapshot(self) -> Json:
        state: Json = {}
        for module in self.modules:
            value = module.snapshot()
            if value is not None:
                state[module.name] = value
        return state

    def world_content(self, heading: str) -> list[Json]:
        """Build provider-neutral multimodal content for the current world state.

        Model adapters translate this compact text/image representation to a
        provider's native request format when it is not OpenAI-compatible.
        """
        content: list[Json] = [
            {"type": "text", "text": f"{heading}\n{_json(self.snapshot())}"}
        ]
        for module in self.modules:
            provider = getattr(module, "prompt_content", None)
            if provider is not None:
                content.extend(provider())
        return content


def object_schema(
    properties: Json,
    required: Sequence[str] = (),
    *,
    additional_properties: bool = False,
) -> Json:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": additional_properties,
    }


def _validate(arguments: Mapping[str, Any], schema: Json) -> None:
    required = schema.get("required", [])
    missing = [key for key in required if key not in arguments]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")

    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValueError(f"unknown arguments: {', '.join(unknown)}")

    python_types = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for key, value in arguments.items():
        spec = properties.get(key, {})
        expected_name = spec.get("type")
        expected = python_types.get(expected_name)
        if expected and not isinstance(value, expected):
            raise TypeError(f"{key} must be {expected_name}")
        # ⚠️ AN ENUM IS A GUARANTEE, NOT A HINT. Providers that support strict
        # tool schemas make an out-of-set value structurally impossible for the
        # model to emit -- but not every provider does, and this layer is what
        # makes the guarantee hold everywhere. It matters most for the one
        # argument that turns into motion: a hallucinated destination must come
        # back as a refusal carrying the real list, never as a lookup that
        # happens to match something else.
        allowed = spec.get("enum")
        if allowed is not None and value not in allowed:
            listed = ", ".join(repr(option) for option in allowed) or "(nothing)"
            raise ValueError(f"{key} must be one of: {listed}")


def _json(value: object) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), sort_keys=True)
