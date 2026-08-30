from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from .model import ChatModel
from .tools import Module, Tool, ToolRegistry, object_schema
from .types import AgentOutcome, AssistantTurn, Json, ToolCall


Approval = Callable[[ToolCall, Tool], bool]


SYSTEM_PROMPT = """You are the slow System-2 mission controller for a physical robot.
Translate the user's goal into cautious calls to the provided semantic capabilities.

Rules:
- Reason at the level of destinations and skills. Never invent joint, torque, or velocity commands.
- Make at most one tool call per turn. Physical actions are sequential.
- Observe or query state when preconditions or outcomes are uncertain.
- After every physical action, inspect the fresh world snapshot and verify progress.
- Treat tool errors as evidence: revise the plan instead of repeating blindly.
- Call finish only after verifying the entire mission. Call request_human when safety or
  missing information prevents a responsible action.
"""


class System2Agent:
    def __init__(
        self,
        model: ChatModel,
        modules: Sequence[Module],
        *,
        approval: Approval | None = None,
        max_model_calls: int = 40,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.approval = approval
        self.max_model_calls = max_model_calls
        self.registry = ToolRegistry(modules, _core_tools())
        self.system_prompt = system_prompt

    def run(self, mission: str) -> AgentOutcome:
        events: list[Json] = []
        messages: list[Json] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Mission: {mission}"},
                    *self.registry.world_content("Initial world snapshot:"),
                ],
            },
        ]

        for model_call in range(1, self.max_model_calls + 1):
            turn = self.model.complete(messages, self.registry.schemas())
            messages.append(_assistant_message(turn))

            if len(turn.tool_calls) != 1:
                error = "make exactly one tool call on each turn"
                events.append({"type": "protocol_error", "error": error})
                messages.append({"role": "user", "content": error})
                continue

            call = turn.tool_calls[0]
            tool = self.registry.get(call.name)
            if tool is None:
                result = {"ok": False, "error": f"unknown tool: {call.name}"}
            elif tool.requires_approval and not self._approved(call, tool):
                result = {"ok": False, "error": "motion was not approved by the safety layer"}
            else:
                result = tool.run(call.arguments).as_json()

            event = {
                "type": "tool",
                "name": call.name,
                "arguments": call.arguments,
                "result": result,
            }
            events.append(event)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": _dump(result),
                }
            )

            if call.name == "finish" and result.get("ok"):
                return AgentOutcome(
                    status="completed",
                    summary=str(call.arguments["summary"]),
                    model_calls=model_call,
                    events=tuple(events),
                )
            if call.name == "request_human" and result.get("ok"):
                return AgentOutcome(
                    status="needs_human",
                    summary=str(call.arguments["reason"]),
                    model_calls=model_call,
                    events=tuple(events),
                )

            if tool is not None and (tool.kind == "action" or tool.refresh_world):
                messages.append(
                    {
                        "role": "user",
                        "content": self.registry.world_content(
                            "Fresh world snapshot after action:"
                        ),
                    }
                )

        return AgentOutcome(
            status="budget_exhausted",
            summary=f"stopped after {self.max_model_calls} model calls",
            model_calls=self.max_model_calls,
            events=tuple(events),
        )

    def _approved(self, call: ToolCall, tool: Tool) -> bool:
        return self.approval(call, tool) if self.approval is not None else False


def _assistant_message(turn: AssistantTurn) -> Json:
    message: Json = {"role": "assistant", "content": turn.content or None}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": _dump(call.arguments)},
            }
            for call in turn.tool_calls
        ]
    return message


def _core_tools() -> tuple[Tool, ...]:
    return (
        Tool(
            name="finish",
            description="Finish only after the full mission has been verified successful.",
            parameters=object_schema(
                {"summary": {"type": "string", "description": "Verified outcome."}},
                ["summary"],
            ),
            handler=lambda args: {"recorded": args["summary"]},
            kind="terminal",
        ),
        Tool(
            name="request_human",
            description="Stop safely and request human help or missing information.",
            parameters=object_schema(
                {"reason": {"type": "string", "description": "Why help is required."}},
                ["reason"],
            ),
            handler=lambda args: {"recorded": args["reason"]},
            kind="terminal",
        ),
    )


def _dump(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)
