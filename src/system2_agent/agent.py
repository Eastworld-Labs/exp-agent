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


MAX_STALLED_TURNS = 3
"""How many consecutive turns may produce nothing runnable before the mission ends."""


def turn_fault(turn: AssistantTurn) -> str | None:
    """Why this turn cannot be executed, as the sentence to feed back, or None.

    Truncation is checked FIRST and is not a protocol error. A reply cut off at the
    output-token limit can carry a tool call whose arguments are incomplete, and the
    dangerous shape is the one that still parses: ``{"distance": 2.5}`` cut to
    ``{"distance": 2}`` is a valid number and a different action, which no schema check
    can catch. ``finish_reason`` is the only thing that distinguishes it. Borrowed from
    pi-agent-core, which fails every call in a length-stopped message for this reason.

    A length stop whose call happened to be complete costs one wasted turn. Running a
    truncated physical action costs an action nobody asked for.
    """
    if turn.finish_reason == "length":
        return (
            "your reply was cut off at the output token limit, so nothing was run and "
            "any tool call it carried may have truncated arguments. Reason in fewer "
            "words and re-issue the single call you wanted, with complete arguments"
        )
    if not turn.tool_calls:
        return (
            "make exactly one tool call on each turn: call finish once the mission is "
            "verified, or request_human if you are stuck"
        )
    if len(turn.tool_calls) > 1:
        return (
            f"you made {len(turn.tool_calls)} tool calls in one turn. Make exactly one: "
            "physical actions are sequential and each outcome must be seen before the "
            "next call. Re-issue only the first thing you wanted to do"
        )
    return None


class System2Agent:
    def __init__(
        self,
        model: ChatModel,
        modules: Sequence[Module],
        *,
        approval: Approval | None = None,
        max_model_calls: int = 40,
        system_prompt: str = SYSTEM_PROMPT,
        on_event: Callable[[Json], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.model = model
        self.approval = approval
        self.max_model_calls = max_model_calls
        self.registry = ToolRegistry(modules, _core_tools())
        self.system_prompt = system_prompt
        # A watcher on the same events `run` returns, delivered AS THEY HAPPEN
        # rather than at the end. A mission takes minutes and moves a robot;
        # a caller that only learns the transcript afterwards cannot show a
        # person what is happening while it happens. Exceptions raised here are
        # swallowed -- see emit() in run().
        self.on_event = on_event
        # Polled between steps. ⚠️ IT ENDS THE MISSION, NOT THE MOTION: a goal
        # the navigation backend has already handed to the robot keeps running,
        # and only that backend's own cancellation (or an E-stop) reaches it.
        self.should_stop = should_stop

    def run(self, mission: str) -> AgentOutcome:
        events: list[Json] = []
        usage_total: Json = {}
        # Consecutive turns that produced NOTHING TO RUN. Bounded separately from any
        # tool-level guard because a model wedged at the output-token limit, or one that
        # keeps emitting two calls, never reaches a tool at all.
        stalled = 0
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

        def emit(event: Json) -> None:
            events.append(event)
            if self.on_event is not None:
                try:
                    self.on_event(event)
                except Exception:  # noqa: BLE001
                    # A watcher that throws must not end a mission a robot is
                    # in the middle of. The transcript is still complete in
                    # `events`; only the live view missed a frame.
                    pass

        for model_call in range(1, self.max_model_calls + 1):
            # ⚠️ CHECKED HERE, BEFORE THE MODEL CALL AND BEFORE ANY TOOL. Stop
            # can only take effect between steps: a `navigate_to` already
            # running blocks inside the tool, and the backend is what has to
            # notice a cancellation there. Stopping the MISSION never stops the
            # ROBOT -- a goal the planner already accepted keeps running.
            if self.should_stop is not None and self.should_stop():
                return AgentOutcome(
                    status="cancelled",
                    summary="the operator stopped the mission",
                    model_calls=model_call - 1,
                    events=tuple(events),
                    usage=usage_total or None,
                )
            turn = self.model.complete(messages, self.registry.schemas())
            messages.append(_assistant_message(turn))
            usage_total = _add_usage(usage_total, turn.usage)
            emit({
                "type": "turn",
                "model_call": model_call,
                "content": turn.content,
                "reasoning": turn.reasoning,
                "usage": dict(usage_total) or None,
            })

            fault = turn_fault(turn)
            if fault is not None:
                emit({"type": "protocol_error", "error": fault})
                messages.append({"role": "user", "content": fault})
                stalled += 1
                if stalled >= MAX_STALLED_TURNS:
                    return AgentOutcome(
                        status="failed",
                        summary=(
                            f"{MAX_STALLED_TURNS} turns in a row produced nothing to "
                            f"run: {fault}"
                        ),
                        model_calls=model_call,
                        events=tuple(events),
                        usage=usage_total or None,
                    )
                continue
            stalled = 0

            call = turn.tool_calls[0]
            tool = self.registry.get(call.name)
            # ⚠️ ANNOUNCED BEFORE IT RUNS, not after. A navigation tool blocks
            # for the whole walk -- tens of seconds -- and a watcher that only
            # learned about the step on completion would show nothing at all
            # while the robot was moving.
            emit({
                "type": "call",
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "moves_robot": bool(tool is not None and tool.requires_approval),
            })
            if tool is None:
                result = {"ok": False, "error": f"unknown tool: {call.name}"}
            elif tool.requires_approval and not self._approved(call, tool):
                result = {"ok": False, "error": "motion was not approved by the safety layer"}
            else:
                result = tool.run(call.arguments).as_json()

            emit({
                "type": "tool",
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "result": result,
            })
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
                    usage=usage_total or None,
                )
            if call.name == "request_human" and result.get("ok"):
                return AgentOutcome(
                    status="needs_human",
                    summary=str(call.arguments["reason"]),
                    model_calls=model_call,
                    events=tuple(events),
                    usage=usage_total or None,
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
            usage=usage_total or None,
        )

    def _approved(self, call: ToolCall, tool: Tool) -> bool:
        return self.approval(call, tool) if self.approval is not None else False


def _assistant_message(turn: AssistantTurn) -> Json:
    message: Json = {"role": "assistant", "content": turn.content or None}
    # ⚠️ VERBATIM, OR THE NEXT REQUEST IS REJECTED. A reasoning model's provider
    # requires the exact sequence of blocks it produced to come back with the
    # tool results. The flattened human-readable `reasoning` is deliberately NOT
    # what goes here.
    if turn.reasoning_details:
        message["reasoning_details"] = list(turn.reasoning_details)
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


#: The same function, for the nested loops that also replay a turn to a provider
#: (`grounding.VisionGrounder`). Shared rather than copied because the
#: `reasoning_details` rule above is the kind of thing a copy silently loses.
assistant_message = _assistant_message


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


def _add_usage(total: Json, turn_usage: "Json | None") -> Json:
    """Accumulate a provider usage block across turns.

    ⚠️ ABSENT AND ZERO ARE DIFFERENT and must stay different all the way to the
    screen: 0 tokens is a claim, `null` is "nobody said". A cost readout that
    renders both as 0 tells an operator the run was free.
    """
    if not turn_usage:
        return total
    merged = dict(total)
    for key, value in turn_usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            merged[key] = merged.get(key, 0) + value
        elif key not in merged:
            merged[key] = value
    return merged
