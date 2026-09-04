"""Agent events -> the frames the dashboard's transcript already renders.

The browser's MissionPanel was written against a `MissionEvent` union
(`reasoning`, `say`, `call`, `gate`, `result`, `images`, `note`, `end`). Mapping
onto it here rather than teaching the panel a second shape means the host-side
loop and the in-browser one show identically, and the panel needs no change at
all to display a mission it is no longer running.

PURE: no server, no state, no clock. `at_ms` is passed in.
"""
from __future__ import annotations

import json
from typing import Any

from ..types import Json


def summarize(result: Json) -> str:
    """One line for a tool result, for the collapsed transcript row.

    Prefers the fields that carry the outcome over the ones that carry the most
    text, so a navigation row reads "arrived · 0.31 m · planner" rather than the
    first 80 characters of a disclaimer.
    """
    if not result.get("ok", True):
        return str(result.get("error") or "refused")[:200]
    data = result.get("data")
    if not isinstance(data, dict):
        return "" if data is None else str(data)[:200]
    parts: list[str] = []
    if data.get("state"):
        parts.append(str(data["state"]))
    if data.get("remaining_m") is not None:
        parts.append(f"{data['remaining_m']} m")
    if data.get("verdict_source"):
        parts.append(str(data["verdict_source"]))
    if data.get("locations"):
        parts.append(f"{len(data['locations'])} places")
    if data.get("recorded"):
        parts.append(str(data["recorded"])[:160])
    return " · ".join(parts) or json.dumps(data, default=str)[:200]


def to_frames(event: Json, at_ms: int) -> list[Json]:
    """One agent event -> zero or more transcript frames."""
    kind = event.get("type")
    frames: list[Json] = []

    if kind == "turn":
        if event.get("reasoning"):
            frames.append({"t": "reasoning", "text": event["reasoning"], "at": at_ms})
        if event.get("content"):
            frames.append({"t": "say", "text": event["content"], "at": at_ms})
        # ⚠️ SENT ON EVERY TURN, not only at the end, and `null` is preserved.
        # A cost readout that renders "nobody reported this" and "zero" alike
        # tells an operator the run was free.
        frames.append({
            "t": "usage",
            "usage": event.get("usage"),
            "modelCalls": event.get("model_call"),
            "at": at_ms,
        })
    elif kind == "call":
        frames.append({
            "t": "call",
            "id": event.get("id"),
            "name": event.get("name"),
            "args": event.get("arguments") or {},
            "movesRobot": bool(event.get("moves_robot")),
            "at": at_ms,
        })
    elif kind == "tool":
        result = event.get("result") or {}
        frames.append({
            "t": "result",
            "id": event.get("id"),
            "name": event.get("name"),
            "ok": bool(result.get("ok")),
            "summary": summarize(result),
            "data": result.get("data") if result.get("ok") else result.get("error"),
            "at": at_ms,
        })
    elif kind == "protocol_error":
        frames.append({
            "t": "note", "text": str(event.get("error") or ""), "tone": "warn", "at": at_ms,
        })
    return frames


def gate_frame(call: Any, approved: bool, mode: str, at_ms: int) -> Json:
    return {
        "t": "gate",
        "id": getattr(call, "id", None),
        "name": getattr(call, "name", None),
        "verdict": "approved" if approved else "declined",
        "mode": mode,
        "at": at_ms,
    }


def end_frame(outcome: Any, at_ms: int) -> Json:
    return {
        "t": "end",
        "status": getattr(outcome, "status", "failed"),
        "summary": getattr(outcome, "summary", ""),
        "modelCalls": getattr(outcome, "model_calls", 0),
        "usage": getattr(outcome, "usage", None),
        "at": at_ms,
    }
