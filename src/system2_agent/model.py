from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .types import AssistantTurn, Json, ToolCall


class ChatModel(Protocol):
    def complete(self, messages: Sequence[Json], tools: Sequence[Json]) -> AssistantTurn: ...


@dataclass
class OpenAICompatibleModel:
    """Tiny Chat Completions client for OpenAI-compatible model servers."""

    model: str
    base_url: str
    api_key: str
    timeout_s: float = 120.0
    temperature: float | None = None
    # ⚠️ NO max_tokens IS SENT UNLESS THIS IS SET, and that is deliberate: a
    # limit low enough to truncate a tool call produces the failure mode
    # agent.turn_fault exists for. Set it only when a provider requires it.
    max_tokens: int | None = None
    # Provider-specific request fields (OpenRouter's `reasoning`/`usage`, say).
    # Merged UNDER the standard keys -- see complete().
    extra_body: Json | None = None
    extra_headers: Mapping[str, str] | None = None

    @classmethod
    def from_env(
        cls,
        model: str,
        *,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> "OpenAICompatibleModel":
        provider, separator, bare_model = model.partition("/")
        routes = {
            "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
            "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
            "google": (
                "https://generativelanguage.googleapis.com/v1beta/openai",
                "GEMINI_API_KEY",
            ),
            "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        }
        inferred_url, inferred_key = routes.get(provider, (None, None))
        resolved_url = base_url or inferred_url or os.getenv("SYSTEM2_BASE_URL")
        key_name = api_key_env or inferred_key or "SYSTEM2_API_KEY"
        key = os.getenv(key_name, "")
        if not resolved_url:
            raise ValueError("set --base-url or SYSTEM2_BASE_URL for this model")
        if not key:
            raise ValueError(f"missing API key in {key_name}")

        # Provider prefixes are routing hints, except OpenRouter where the nested
        # provider remains part of the model id.
        resolved_model = model
        if separator and provider != "openrouter":
            resolved_model = bare_model
        elif provider == "openrouter":
            resolved_model = bare_model
        return cls(model=resolved_model, base_url=resolved_url, api_key=key)

    def complete(self, messages: Sequence[Json], tools: Sequence[Json]) -> AssistantTurn:
        payload: Json = {}
        # Provider-specific extras first, so the four keys below can never be
        # overridden by them: `messages` and `tools` are the conversation this
        # agent built, and a caller that could replace them could replace the
        # tool set a physical action is gated on.
        if self.extra_body:
            payload.update(self.extra_body)
        payload.update({
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
        })
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **dict(self.extra_headers or {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"model HTTP {exc.code}: {detail}") from exc

        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected model response: {body}") from exc

        calls = []
        for raw_call in message.get("tool_calls") or []:
            function = raw_call["function"]
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid tool arguments from model: {function}") from exc
            calls.append(
                ToolCall(
                    id=raw_call.get("id", f"call_{len(calls)}"),
                    name=function["name"],
                    arguments=arguments,
                )
            )
        finish_reason = choice.get("finish_reason")
        details = tuple(message.get("reasoning_details") or ())
        return AssistantTurn(
            content=message.get("content") or "",
            tool_calls=tuple(calls),
            finish_reason=str(finish_reason) if finish_reason else None,
            reasoning=str(message.get("reasoning") or "") or _reasoning_text(details),
            reasoning_details=details,
            usage=body.get("usage") or None,
        )


def _reasoning_text(details: Sequence[Json]) -> str:
    """Flatten provider reasoning blocks into something readable.

    ⚠️ NEVER the `encrypted` variant. Some providers return reasoning as an
    opaque blob that exists to be echoed back, not read; rendering it would put
    a wall of base64 in front of an operator who is deciding whether to let a
    robot walk.
    """
    parts = []
    for block in details:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "reasoning.encrypted":
            continue
        text = block.get("text") or block.get("summary") or ""
        if text:
            parts.append(str(text))
    return "\n".join(parts)
