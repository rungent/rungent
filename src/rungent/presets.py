"""Plug-and-play helpers: OpenAPI HTTP agent + System One shortlist."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .agent import Agent, RunActivityProvider
from .http_tools import (
    HttpOperation,
    HttpToolSettings,
    OpenApiHost,
    load_operations_from_hosts,
    load_operations_from_sources,
    tools_from_operations,
)
from .llm import Model
from .runtime import Runtime
from .store import Store
from .systemone import SystemOneSettings
from .tools import Tool, ToolContext

ContextProvider = Callable[[ToolContext], str | Awaitable[str]]


def create_openapi_agent(
    *,
    name: str,
    instructions: str | Path,
    http: HttpToolSettings,
    operations: Sequence[HttpOperation | Mapping[str, Any]] | None = None,
    hosts: Sequence[OpenApiHost | Mapping[str, Any]] | None = None,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
    context: ContextProvider | None = None,
    run_activity: RunActivityProvider | None = None,
    model: str | None = None,
    extra_tools: Sequence[Tool] = (),
) -> Agent:
    resolved = _resolve_operations(operations=operations, hosts=hosts, specs=specs)
    tools = [*tools_from_operations(resolved, http), *extra_tools]
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
    http: HttpToolSettings,
    operations: Sequence[HttpOperation | Mapping[str, Any]] | None = None,
    hosts: Sequence[OpenApiHost | Mapping[str, Any]] | None = None,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
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
        http=http,
        operations=operations,
        hosts=hosts,
        specs=specs,
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


def replace_openapi_tools(
    runtime: Runtime,
    *,
    hosts: Sequence[OpenApiHost | Mapping[str, Any]],
    http: HttpToolSettings,
    name: str | None = None,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
    extra_tools: Sequence[Tool] = (),
    env: Mapping[str, str] | None = None,
) -> list[HttpOperation]:
    """Reload an agent's HTTP tools from OpenAPI hosts (keeps instructions/context)."""
    agent_name = name or runtime.default_agent
    old = runtime.agents[agent_name]
    operations = load_operations_from_hosts(hosts, specs=specs, env=env)
    runtime.agents[agent_name] = Agent(
        name=old.name,
        instructions=old.instructions,
        tools=[*tools_from_operations(operations, http), *extra_tools],
        context=old.context,
        run_activity=old.run_activity,
        run_initializer=old.run_initializer,
        model=old.model,
    )
    return operations


def _resolve_operations(
    *,
    operations: Sequence[HttpOperation | Mapping[str, Any]] | None,
    hosts: Sequence[OpenApiHost | Mapping[str, Any]] | None,
    specs: Mapping[str, Mapping[str, Any]] | None,
) -> list[HttpOperation | Mapping[str, Any]]:
    if hosts is not None:
        if operations is not None:
            raise ValueError("Pass hosts= or operations=, not both")
        return load_operations_from_hosts(hosts, specs=specs)
    if operations is None:
        return []
    return list(operations)


__all__ = [
    "HttpOperation",
    "HttpToolSettings",
    "OpenApiHost",
    "create_openapi_agent",
    "create_openapi_runtime",
    "load_operations_from_hosts",
    "load_operations_from_sources",
    "replace_openapi_tools",
    "tools_from_operations",
]
