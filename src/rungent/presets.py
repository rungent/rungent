"""Plug-and-play helpers: OpenAPI/allowlist HTTP agent + System One shortlist."""

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .agent import Agent, RunActivityProvider
from .http_tools import (
    HttpOperation,
    HttpToolSettings,
    load_operations,
    load_operations_from_config,
    operations_from_openapi,
    tools_from_operations,
)
from .llm import Model
from .runtime import Runtime
from .store import Store
from .systemone import SystemOneSettings
from .tools import Tool, ToolContext

ContextProvider = Callable[[ToolContext], str | Awaitable[str]]


def load_allowlist_file(path: str | Path) -> list[HttpOperation]:
    """Load operations from a legacy allowlist or a thin OpenAPI tools config."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "sources" in payload:
        return load_operations_from_config(path)
    return load_operations(payload)


def create_openapi_agent(
    *,
    name: str,
    instructions: str | Path,
    operations: Sequence[HttpOperation | Mapping[str, Any]],
    http: HttpToolSettings,
    context: ContextProvider | None = None,
    run_activity: RunActivityProvider | None = None,
    model: str | None = None,
    extra_tools: Sequence[Tool] = (),
) -> Agent:
    tools = [*tools_from_operations(operations, http), *extra_tools]
    path = Path(instructions)
    text = path.read_text(encoding="utf-8") if path.exists() and path.is_file() else str(instructions)
    return Agent(
        name=name,
        instructions=text,
        tools=tools,
        context=context,
        run_activity=run_activity,
        model=model,
    )


def create_openapi_runtime(
    *,
    name: str,
    instructions: str | Path,
    model: Model,
    store: Store,
    operations: Sequence[HttpOperation | Mapping[str, Any]],
    http: HttpToolSettings,
    systemone: SystemOneSettings | None = None,
    context: ContextProvider | None = None,
    run_activity: RunActivityProvider | None = None,
    extra_tools: Sequence[Tool] = (),
    max_model_steps: int = 32,
    context_budget_tokens: int = 80_000,
    dependency_provider=None,
    external_task_resolver=None,
    event_listener=None,
    model_step_timeout_seconds: float | None = 60,
    model_step_total_timeout_seconds: float | None = 150,
    approval_revision: str = "1",
) -> Runtime:
    agent = create_openapi_agent(
        name=name,
        instructions=instructions,
        operations=operations,
        http=http,
        context=context,
        run_activity=run_activity,
        extra_tools=extra_tools,
    )
    return Runtime(
        agents=[agent],
        model=model,
        store=store,
        systemone=systemone,
        max_model_steps=max_model_steps,
        context_budget_tokens=context_budget_tokens,
        dependency_provider=dependency_provider,
        external_task_resolver=external_task_resolver,
        event_listener=event_listener,
        model_step_timeout_seconds=model_step_timeout_seconds,
        model_step_total_timeout_seconds=model_step_total_timeout_seconds,
        approval_revision=approval_revision,
    )


__all__ = [
    "HttpOperation",
    "HttpToolSettings",
    "create_openapi_agent",
    "create_openapi_runtime",
    "load_allowlist_file",
    "load_operations",
    "load_operations_from_config",
    "operations_from_openapi",
    "tools_from_operations",
]
