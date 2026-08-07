"""LLM streaming protocol and the built-in OpenAI-compatible transport."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .state import ToolCall, new_id


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ModelCompleted:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    provider_request_id: str | None = None


@dataclass(frozen=True, slots=True)
class ModelRetrying:
    retry: int
    max_retries: int
    delay_seconds: float
    reason: str
    tool_name: str | None = None
    arguments_chars: int | None = None
    error_kind: str | None = None
    error_position: int | None = None


ModelEvent = TextDelta | ModelRetrying | ModelCompleted


class ModelOutputError(ValueError):
    """The provider completed a response that cannot be consumed safely."""


class Model(Protocol):
    def stream(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        model: str | None = None,
    ) -> AsyncIterator[ModelEvent]: ...


class OpenAICompatibleModel:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 120,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
        extra_body: dict[str, Any] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.extra_body = dict(extra_body or {})
        self.tool_choice = tool_choice
        reserved = {"model", "messages", "tools", "tool_choice", "stream", "stream_options"}
        overlap = reserved.intersection(self.extra_body)
        if overlap:
            raise ValueError(f"extra_body cannot override {sorted(overlap)[0]}")
        self._client = client

    @classmethod
    def from_env(cls) -> OpenAICompatibleModel:
        raw_extra_body = os.environ.get("LLM_EXTRA_BODY", "").strip()
        extra_body: dict[str, Any] = {}
        if raw_extra_body:
            parsed = json.loads(raw_extra_body)
            if not isinstance(parsed, dict):
                raise ValueError("LLM_EXTRA_BODY must be a JSON object")
            extra_body = parsed
        return cls(
            api_key=os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("LLM_MODEL_ID", "gpt-5-mini"),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
            retry_backoff_seconds=float(os.environ.get("LLM_RETRY_BACKOFF_SECONDS", "0.5")),
            extra_body=extra_body,
        )

    @staticmethod
    def _retry_reason(exc: Exception) -> str | None:
        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.TransportError):
            return "network"
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 408:
                return "timeout"
            if status == 429:
                return "rate_limit"
            if 500 <= status <= 599:
                return "server_error"
        return None

    async def stream(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        model: str | None = None,
    ) -> AsyncIterator[ModelEvent]:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is not configured")
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.timeout_seconds)
        payload = {
            "model": model or self.model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": self.tool_choice,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.extra_body,
        }
        request_messages = list(messages)
        try:
            for attempt in range(self.max_retries + 1):
                accumulated_text = ""
                calls: dict[int, dict[str, str]] = {}
                usage: dict[str, Any] | None = None
                finish_reason: str | None = None
                provider_request_id: str | None = None
                try:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={**payload, "messages": request_messages},
                    ) as response:
                        response.raise_for_status()
                        provider_request_id = response.headers.get(
                            "x-request-id"
                        ) or response.headers.get("request-id")
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            raw = line[6:]
                            if raw == "[DONE]":
                                break
                            chunk = json.loads(raw)
                            if isinstance(chunk.get("usage"), dict):
                                usage = chunk["usage"]
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0]
                            if choice.get("finish_reason") is not None:
                                finish_reason = str(choice["finish_reason"])
                            delta = choice.get("delta") or {}
                            text = delta.get("content")
                            if isinstance(text, str) and text:
                                accumulated_text += text
                                yield TextDelta(text)
                            for item in delta.get("tool_calls") or []:
                                index = int(item.get("index", 0))
                                target = calls.setdefault(
                                    index, {"id": "", "name": "", "arguments": ""}
                                )
                                if item.get("id"):
                                    target["id"] = item["id"]
                                function = item.get("function") or {}
                                target["name"] += function.get("name") or ""
                                target["arguments"] += function.get("arguments") or ""
                except Exception as exc:
                    reason = self._retry_reason(exc)
                    if reason is None or attempt >= self.max_retries:
                        raise
                    retry = attempt + 1
                    delay = min(self.retry_backoff_seconds * (2**attempt), 8.0)
                    yield ModelRetrying(
                        retry=retry,
                        max_retries=self.max_retries,
                        delay_seconds=delay,
                        reason=reason,
                    )
                    if delay:
                        await asyncio.sleep(delay)
                    continue

                parsed_calls: list[ToolCall] = []
                invalid_target = "unknown"
                try:
                    for index in sorted(calls):
                        item = calls[index]
                        invalid_target = item["name"] or str(index)
                        arguments = json.loads(item["arguments"] or "{}")
                        if not isinstance(arguments, dict):
                            raise ModelOutputError(
                                "Model returned non-object arguments for tool "
                                f"{item['name'] or index}"
                            )
                        parsed_calls.append(
                            ToolCall(
                                id=item["id"] or new_id("call"),
                                name=item["name"],
                                arguments=arguments,
                            )
                        )
                except (json.JSONDecodeError, ModelOutputError) as exc:
                    if attempt >= self.max_retries:
                        raise ModelOutputError(
                            f"Model returned invalid arguments for tool {invalid_target}"
                        ) from exc
                    retry = attempt + 1
                    delay = min(self.retry_backoff_seconds * (2**attempt), 8.0)
                    yield ModelRetrying(
                        retry=retry,
                        max_retries=self.max_retries,
                        delay_seconds=delay,
                        reason="invalid_tool_arguments",
                        tool_name=invalid_target,
                        arguments_chars=len(
                            next(
                                (
                                    item["arguments"]
                                    for item in calls.values()
                                    if (item["name"] or "unknown") == invalid_target
                                ),
                                "",
                            )
                        ),
                        error_kind=type(exc).__name__,
                        error_position=exc.pos if isinstance(exc, json.JSONDecodeError) else None,
                    )
                    request_messages = [
                        *messages,
                        {
                            "role": "system",
                            "content": (
                                "Your previous tool arguments were incomplete or invalid JSON. "
                                "Retry the same operation with compact, valid JSON. Keep every "
                                "required item, but omit verbose repetition in notes."
                            ),
                        },
                    ]
                    if delay:
                        await asyncio.sleep(delay)
                    continue
                yield ModelCompleted(
                    text=accumulated_text,
                    tool_calls=parsed_calls,
                    usage=usage,
                    finish_reason=finish_reason,
                    provider_request_id=provider_request_id,
                )
                return
        finally:
            if owns_client:
                await client.aclose()
