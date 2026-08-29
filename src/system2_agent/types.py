from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Json = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Json


@dataclass(frozen=True)
class AssistantTurn:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None

    def as_json(self) -> Json:
        result: Json = {"ok": self.ok}
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass(frozen=True)
class AgentOutcome:
    status: str
    summary: str
    model_calls: int
    events: tuple[Json, ...] = field(default_factory=tuple)
