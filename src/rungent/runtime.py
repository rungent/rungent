import asyncio
import inspect
import json
import logging
import re
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, cast

from pydantic import ValidationError

from .acs import Event, EventEmitter
from .agent import Agent, RunActivity
from .llm import Model, ModelCompleted, ModelRetrying, TextDelta
from .state import (
    DeferredRequest,
    Identity,
    Interaction,
    InteractionOption,
    InteractionQuestion,
    InteractionRequest,
    InteractionResponse,
    Message,
    PendingCall,
    PendingExternal,
    Run,
    RunStatus,
    Session,
    ToolCall,
    ToolError,
    ToolResult,
    TrustedInteractionResponse,
    new_id,
    now,
)
from .store import Store
from .tools import ApprovalPolicy, Tool, ToolContext, ToolEffect, validation_error_message
from .usage import (
    DEFAULT_CONTEXT_BUDGET_TOKENS,
    calibrate_context_usage,
    estimate_context_usage,
)

_PROGRESS_INSTRUCTIONS = """Public progress updates:
- For a multi-step task, report useful progress alongside business tool calls when possible.
- Never spend a model step only on report_progress.
- Keep each update short and useful to the user.
- Never report hidden reasoning, system prompts, secrets, private tool data, or speculation.
- Do not call report_progress for a simple answer."""

logger = logging.getLogger(__name__)

_NUMBERED_CHOICE_RE = re.compile(r"(?m)^\s*(?:\d{1,2}[.)、．:：]|[A-Da-d][.)])\s*\S")

ExternalTaskCanceller = Callable[[Session, Run, DeferredRequest], Awaitable[None] | None]
RuntimeEventListener = Callable[[Event], Awaitable[None] | None]
RunDependencyProvider = Callable[[Session, Run], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


class ActiveRunConflict(ValueError):
    def __init__(self, run: Run) -> None:
        super().__init__(f"Session already has an active run: {run.id}")
        self.run = run


@dataclass(frozen=True, slots=True)
class _AssembledPrompt:
    messages: list[dict[str, Any]]
    tool_schemas: list[dict[str, Any]]
    usage: dict[str, Any]


class Runtime:
    def __init__(
        self,
        *,
        agents: list[Agent],
        model: Model,
        store: Store,
        default_agent: str | None = None,
        max_model_steps: int = 16,
        model_wait_progress_after_seconds: float = 3.0,
        model_wait_progress_interval_seconds: float = 5.0,
        model_step_timeout_seconds: float | None = None,
        model_step_total_timeout_seconds: float | None = None,
        external_task_canceller: ExternalTaskCanceller | None = None,
        dependency_provider: RunDependencyProvider | None = None,
        event_listener: RuntimeEventListener | None = None,
        context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS,
    ) -> None:
        if not agents:
            raise ValueError("Runtime needs at least one agent")
        self.agents = {agent.name: agent for agent in agents}
        if len(self.agents) != len(agents):
            raise ValueError("Agent names must be unique")
        self.default_agent = default_agent or agents[0].name
        if self.default_agent not in self.agents:
            raise ValueError(f"Unknown default agent: {self.default_agent}")
        if not 1 <= max_model_steps <= 128:
            raise ValueError("max_model_steps must be between 1 and 128")
        if model_wait_progress_after_seconds <= 0:
            raise ValueError("model_wait_progress_after_seconds must be positive")
        if model_wait_progress_interval_seconds <= 0:
            raise ValueError("model_wait_progress_interval_seconds must be positive")
        if model_step_timeout_seconds is not None and model_step_timeout_seconds <= 0:
            raise ValueError("model_step_timeout_seconds must be positive when provided")
        if model_step_total_timeout_seconds is not None and model_step_total_timeout_seconds <= 0:
            raise ValueError("model_step_total_timeout_seconds must be positive when provided")
        if context_budget_tokens <= 0:
            raise ValueError("context_budget_tokens must be positive")
        self.model = model
        self.store = store
        self.max_model_steps = max_model_steps
        self.model_wait_progress_after_seconds = model_wait_progress_after_seconds
        self.model_wait_progress_interval_seconds = model_wait_progress_interval_seconds
        self.model_step_timeout_seconds = model_step_timeout_seconds
        self.model_step_total_timeout_seconds = model_step_total_timeout_seconds
        self.external_task_canceller = external_task_canceller
        self.dependency_provider = dependency_provider
        self.event_listener = event_listener
        self.context_budget_tokens = context_budget_tokens
        self._session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._run_tasks: dict[str, asyncio.Task[Any]] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._worker_id = new_id("worker")
        self._lease_seconds = 30.0

    async def create_session(
        self,
        *,
        identity: Identity,
        agent_name: str | None = None,
        title: str | None = None,
        resource: Mapping[str, Any] | None = None,
    ) -> Session:
        selected = agent_name or self.default_agent
        if selected not in self.agents:
            raise KeyError(f"Unknown agent: {selected}")
        session = Session(
            agent_name=selected,
            subject_id=identity.subject_id,
            tenant_id=identity.tenant_id,
            title=title,
            resource=dict(resource or {}),
        )
        await self.store.create_session(session)
        return session

    @staticmethod
    def _authorize(session: Session, identity: Identity) -> None:
        if session.subject_id != identity.subject_id or session.tenant_id != identity.tenant_id:
            raise PermissionError("Session does not belong to this identity")

    async def get_session(
        self, session_id: str, *, identity: Identity
    ) -> tuple[Session, list[Message]]:
        session = await self.store.get_session(session_id)
        self._authorize(session, identity)
        return session, list(await self.store.list_messages(session_id))

    async def get_context_usage(
        self,
        session_id: str,
        *,
        identity: Identity,
        deps: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        session, _ = await self.get_session(session_id, identity=identity)
        agent = self.agents[session.agent_name]
        ctx = ToolContext(
            identity=identity,
            session_id=session.id,
            run_id="",
            current_input="",
            resource=session.resource,
            deps=deps or {},
        )
        assembled = await self._assemble_prompt(session, agent, ctx)
        return assembled.usage

    async def _assemble_prompt(
        self,
        session: Session,
        agent: Agent,
        ctx: ToolContext,
    ) -> _AssembledPrompt:
        conversation = [item.model_message() for item in await self.store.list_messages(session.id)]
        context_text = ""
        if agent.context:
            context = agent.context(ctx)
            if inspect.isawaitable(context):
                context = await context
            if context:
                context_text = f"Current context:\n{context}"
        messages = [
            {"role": "system", "content": agent.instructions},
            {"role": "system", "content": _PROGRESS_INSTRUCTIONS},
        ]
        if context_text:
            messages.append({"role": "system", "content": context_text})
        messages.extend(conversation)
        tool_schemas = agent.tool_schemas(
            interaction_response_available=ctx.interaction_response is not None
        )
        return _AssembledPrompt(
            messages=messages,
            tool_schemas=tool_schemas,
            usage=estimate_context_usage(
                instructions=agent.instructions,
                runtime=_PROGRESS_INSTRUCTIONS,
                context=context_text,
                conversation=conversation,
                tool_schemas=tool_schemas,
                budget=self.context_budget_tokens,
            ),
        )

    async def list_sessions(self, *, identity: Identity) -> list[Session]:
        return list(await self.store.list_sessions(identity))

    async def set_session_title(
        self, session_id: str, title: str | None, *, identity: Identity
    ) -> Session:
        session = await self.store.get_session(session_id)
        self._authorize(session, identity)
        updated = session.model_copy(update={"title": None if title is None else title})
        await self.store.save_session(updated)
        return updated

    async def get_run_events(
        self,
        run_id: str,
        *,
        identity: Identity,
        after_seq: int = 0,
    ) -> list[Event]:
        run = await self.store.get_run(run_id)
        session = await self.store.get_session(run.session_id)
        self._authorize(session, identity)
        return list(await self.store.list_events(run_id, after_seq=after_seq))

    async def list_session_runs(self, session_id: str, *, identity: Identity) -> list[Run]:
        session = await self.store.get_session(session_id)
        self._authorize(session, identity)
        return list(await self.store.list_runs(session_id))

    async def cancel_run(self, run_id: str, *, identity: Identity) -> Run:
        """Cancel an active drive and persist a terminal event, including after recovery."""
        run = await self.store.get_run(run_id)
        session = await self.store.get_session(run.session_id)
        self._authorize(session, identity)
        if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return run
        task = self._run_tasks.get(run_id)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            return await self.store.get_run(run_id)

        pending_external = run.pending_external
        if pending_external is not None and self.external_task_canceller is not None:
            cancelled = self.external_task_canceller(session, run, pending_external.task)
            if inspect.isawaitable(cancelled):
                await cancelled
        run.status = RunStatus.CANCELLED
        run.error = "Run cancelled"
        run.pending_external = None
        persisted_events = await self.store.list_events(run.id, after_seq=0)
        sequence = max(
            [run.event_seq, *(event.seq for event in persisted_events)],
            default=run.event_seq,
        )
        emitter = EventEmitter(session.id, run.id, sequence=sequence)
        if pending_external is not None:
            await self._emit(
                emitter,
                "external_task.cancelled",
                task_id=pending_external.task.task_id,
                call_id=pending_external.call.id,
                message="External task cancelled",
            )
        await self._emit(emitter, "run.cancelled", status=run.status)
        run.event_seq = emitter.sequence
        await self.store.save_run(run)
        return run

    async def report_external_progress(
        self,
        run_id: str,
        task_id: str,
        *,
        identity: Identity,
        message: str,
        public: Any = None,
    ) -> Event:
        """Persist progress for the one host task currently blocking a Run."""
        message = message.strip()
        if not message or len(message) > 280:
            raise ValueError("Progress must be a non-empty string of at most 280 characters")
        initial = await self.store.get_run(run_id)
        session = await self.store.get_session(initial.session_id)
        self._authorize(session, identity)
        async with self._session_locks[session.id]:
            run = await self.store.get_run(run_id)
            pending = run.pending_external
            if run.status is not RunStatus.WAITING_EXTERNAL or pending is None:
                raise ValueError("Run is not waiting for external work")
            if pending.task.task_id != task_id:
                raise ValueError("External task does not belong to this run")
            emitter = await self._recovered_emitter(session, run)
            event = await self._emit(
                emitter,
                "external_task.progress",
                task_id=task_id,
                call_id=pending.call.id,
                message=message,
                public=public,
            )
            run.event_seq = emitter.sequence
            await self.store.save_run(run)
            return event

    async def resume_deferred(
        self,
        run_id: str,
        task_id: str,
        result: ToolResult,
        *,
        identity: Identity,
        deps: Mapping[str, Any] | None = None,
    ) -> list[Event]:
        """Finish a trusted host task exactly once and resume the model."""
        if result.deferred is not None:
            raise ValueError("A deferred completion cannot start another deferred task")
        initial = await self.store.get_run(run_id)
        session = await self.store.get_session(initial.session_id)
        self._authorize(session, identity)
        events: list[Event] = []
        async with self._session_locks[session.id]:
            run = await self.store.get_run(run_id)
            pending = run.pending_external
            if run.status is not RunStatus.WAITING_EXTERNAL or pending is None:
                raise ValueError("Run is not waiting for external work")
            if pending.task.task_id != task_id:
                raise ValueError("External task does not belong to this run")
            claimed = await self.store.claim_run(
                run_id,
                self._worker_id,
                self._lease_seconds,
                expected_status=RunStatus.WAITING_EXTERNAL,
            )
            if claimed is None:
                raise ValueError("Run is not waiting for external work")
            run = claimed
            pending = run.pending_external
            assert pending is not None
            emitter = await self._recovered_emitter(session, run)
            task = asyncio.current_task()
            if task is not None:
                self._run_tasks[run.id] = task
            heartbeat: asyncio.Task[None] | None = None
            try:

                async def renew_lease() -> None:
                    while True:
                        await asyncio.sleep(self._lease_seconds / 3)
                        renewed = await self.store.renew_run_lease(
                            run.id, self._worker_id, self._lease_seconds
                        )
                        if not renewed:
                            raise RuntimeError("Run execution lease was lost")
                        run.lease_expires_at = now() + timedelta(seconds=self._lease_seconds)

                heartbeat = asyncio.create_task(renew_lease())
                run.pending_external = None
                await self.store.save_run(run)
                event_type = (
                    "external_task.completed" if result.succeeded else "external_task.failed"
                )
                events.append(
                    await self._emit(
                        emitter,
                        event_type,
                        task_id=task_id,
                        call_id=pending.call.id,
                        message=result.message,
                        public=result.public,
                    )
                )
                tool = self.agents[session.agent_name].tool_map()[pending.call.name]
                await self.store.append_message(
                    session.id,
                    Message(
                        role="tool",
                        content=result.for_model(),
                        tool_call_id=pending.call.id,
                        name=tool.name,
                    ),
                )
                events.append(
                    await self._emit(
                        emitter,
                        "tool.completed" if result.succeeded else "tool.failed",
                        call_id=pending.call.id,
                        name=tool.name,
                        title=tool.title,
                        code=None if result.succeeded else "external_task_failed",
                        message=result.message,
                        public=result.public,
                    )
                )
                if result.succeeded and result.interaction is not None:
                    continuation, interaction = self._prepare_tool_interaction(
                        session, result.interaction
                    )
                    await self.store.append_message(
                        session.id,
                        Message(role="assistant", tool_calls=[continuation]),
                    )
                    run.status = RunStatus.WAITING_INPUT
                    run.pending_call = PendingCall(
                        kind="continuation",
                        call=continuation,
                        interaction=interaction,
                    )
                    await self.store.save_run(run)
                    events.append(await self._emit(emitter, "run.waiting_input", status=run.status))
                    events.append(
                        await self._emit(
                            emitter,
                            "interaction.requested",
                            **interaction.model_dump(mode="json"),
                        )
                    )
                    return events
                if not result.succeeded:
                    run.status = RunStatus.FAILED
                    run.error = result.message or "External task failed"
                    await self.store.save_run(run)
                    events.append(
                        await self._emit(
                            emitter,
                            "run.failed",
                            status=run.status,
                            code="external_task_failed",
                            error="The external task could not complete.",
                            retryable=True,
                        )
                    )
                    return events
                async for event in self._safe_drive(
                    session,
                    run,
                    identity,
                    dict(deps or {}),
                    emitter,
                ):
                    events.append(event)
            finally:
                if heartbeat is not None and not heartbeat.done():
                    heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat
                if self._run_tasks.get(run.id) is task:
                    self._run_tasks.pop(run.id, None)
                run.event_seq = emitter.sequence
                await self.store.save_run(run)
                await self.store.release_run_lease(run.id, self._worker_id)
        return events

    async def _recovered_emitter(self, session: Session, run: Run) -> EventEmitter:
        persisted = await self.store.list_events(run.id, after_seq=0)
        sequence = max([run.event_seq, *(event.seq for event in persisted)], default=run.event_seq)
        return EventEmitter(session.id, run.id, sequence=sequence)

    async def _emit(self, emitter: EventEmitter, event_type: str, **data: Any) -> Event:
        event = emitter.emit(event_type, **data)
        await self.store.append_event(event)
        if self.event_listener is not None:
            try:
                observed = self.event_listener(event)
                if inspect.isawaitable(observed):
                    await observed
            except Exception:
                logger.exception(
                    "runtime event listener failed event_type=%s run_id=%s",
                    event.type,
                    event.run_id,
                )
        return event

    async def create_run(
        self,
        *,
        session_id: str,
        content: str,
        identity: Identity,
        idempotency_key: str | None = None,
        deps: Mapping[str, Any] | None = None,
    ) -> Run:
        """Persist and detach a Run from the request that created it."""
        session = await self.store.get_session(session_id)
        self._authorize(session, identity)
        content = content.strip()
        if not content:
            raise ValueError("Run input cannot be empty")
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not idempotency_key or len(idempotency_key) > 200:
                raise ValueError("Idempotency-Key must contain 1 to 200 characters")
        async with self._session_locks[session.id]:
            if idempotency_key is not None:
                existing = await self.store.find_run_by_idempotency_key(session.id, idempotency_key)
                if existing is not None:
                    return existing
            active = await self.store.find_active_run(session.id)
            if active is not None:
                if active.status is RunStatus.WAITING_INPUT:
                    await self.cancel_run(active.id, identity=identity)
                else:
                    raise ActiveRunConflict(active)
            await self.store.append_message(session.id, Message(role="user", content=content))
            run = Run(
                session_id=session.id,
                status=RunStatus.QUEUED,
                input=content,
                idempotency_key=idempotency_key,
            )
            await self.store.create_run(run)
        self._schedule_run(run.id, deps=deps)
        return run

    def _schedule_run(
        self, run_id: str, *, deps: Mapping[str, Any] | None = None
    ) -> asyncio.Task[Any]:
        current = self._run_tasks.get(run_id)
        if current is not None and not current.done():
            return current
        task = asyncio.create_task(self._execute_created_run(run_id, deps=deps))
        self._run_tasks[run_id] = task
        return task

    async def recover_runs(self) -> None:
        """Resume queued work and safely retry interrupted model-only work."""
        for run in await self.store.list_recoverable_runs():
            if run.status is RunStatus.RUNNING:
                session = await self.store.get_session(run.session_id)
                emitter = await self._recovered_emitter(session, run)
                run.status = RunStatus.FAILED
                run.error = "Run was interrupted before reaching a durable boundary"
                run.lease_owner = None
                run.lease_expires_at = None
                await self.store.save_run(run)
                await self._emit(
                    emitter,
                    "run.failed",
                    status=run.status,
                    code="run_interrupted",
                    error="The request was interrupted and can be retried.",
                    retryable=True,
                )
                run.event_seq = emitter.sequence
                await self.store.save_run(run)
                continue
            self._schedule_run(run.id)

    async def start_worker(self) -> None:
        """Start the lightweight SQL-backed recovery monitor."""
        if self._worker_task is not None and not self._worker_task.done():
            return

        async def monitor() -> None:
            while True:
                await self.recover_runs()
                await asyncio.sleep(5)

        await self.recover_runs()
        self._worker_task = asyncio.create_task(monitor())

    async def stop_worker(self) -> None:
        task = self._worker_task
        self._worker_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _resolve_run_deps(
        self, session: Session, run: Run, supplied: Mapping[str, Any] | None
    ) -> Mapping[str, Any]:
        if supplied is not None:
            return supplied
        if self.dependency_provider is None:
            return {}
        resolved = self.dependency_provider(session, run)
        if inspect.isawaitable(resolved):
            return await resolved
        return resolved

    async def _execute_created_run(
        self, run_id: str, *, deps: Mapping[str, Any] | None = None
    ) -> None:
        run = await self.store.get_run(run_id)
        session = await self.store.get_session(run.session_id)
        identity = Identity(subject_id=session.subject_id, tenant_id=session.tenant_id)
        async with self._session_locks[session.id]:
            claimed = await self.store.claim_run(run_id, self._worker_id, self._lease_seconds)
            if claimed is None:
                return
            run = claimed
            emitter = await self._recovered_emitter(session, run)
            heartbeat: asyncio.Task[None] | None = None
            try:

                async def renew_lease() -> None:
                    while True:
                        await asyncio.sleep(self._lease_seconds / 3)
                        renewed = await self.store.renew_run_lease(
                            run.id, self._worker_id, self._lease_seconds
                        )
                        if not renewed:
                            raise RuntimeError("Run execution lease was lost")
                        run.lease_expires_at = now() + timedelta(seconds=self._lease_seconds)

                heartbeat = asyncio.create_task(renew_lease())
                resolved_deps = await self._resolve_run_deps(session, run, deps)
                await self._emit(emitter, "run.started", status=run.status)
                agent = self.agents[session.agent_name]
                ctx = ToolContext(
                    identity=identity,
                    session_id=session.id,
                    run_id=run.id,
                    current_input=run.input,
                    resource=session.resource,
                    deps=resolved_deps,
                )
                run_activity: RunActivity | None = None
                if agent.run_activity is not None:
                    activity_value = agent.run_activity(ctx, run.input)
                    run_activity = (
                        await activity_value
                        if inspect.isawaitable(activity_value)
                        else activity_value
                    )
                    await self._emit(
                        emitter,
                        "activity.updated",
                        id=f"run:{run.id}:model-wait",
                        step=0,
                        kind="progress",
                        message=run_activity.message,
                        status="running",
                        public=dict(run_activity.public),
                    )
                if agent.run_initializer is not None:
                    initial = agent.run_initializer(ctx, run.input)
                    call = await initial if inspect.isawaitable(initial) else initial
                    if call is not None:
                        tool = agent.tool_map().get(call.name)
                        if tool is None or tool.requires_interaction_response:
                            raise ValueError("Run initializer returned an unavailable tool")
                        call = call.model_copy(update={"arguments": tool.normalize(call.arguments)})
                        await self.store.append_message(
                            session.id,
                            Message(role="assistant", tool_calls=[call]),
                        )
                        async for _event in self._execute_tool(
                            session, run, ctx, tool, call, emitter
                        ):
                            pass
                        if run.status in {
                            RunStatus.WAITING_INPUT,
                            RunStatus.WAITING_EXTERNAL,
                        }:
                            return
                async for _event in self._safe_drive(
                    session,
                    run,
                    identity,
                    resolved_deps,
                    emitter,
                    run_activity=run_activity,
                ):
                    pass
            except asyncio.CancelledError:
                if run.status is not RunStatus.CANCELLED:
                    run.status = RunStatus.CANCELLED
                    run.error = "Run cancelled"
                    await self._emit(emitter, "run.cancelled", status=run.status)
                raise
            except Exception as exc:
                logger.exception(
                    "detached run failed session_id=%s run_id=%s error_type=%s",
                    session.id,
                    run.id,
                    type(exc).__name__,
                )
                run.status = RunStatus.FAILED
                run.error = str(exc)
                await self._emit(
                    emitter,
                    "run.failed",
                    status=run.status,
                    code="run_initialization_failed",
                    error="The request could not be started.",
                    retryable=True,
                )
            finally:
                if heartbeat is not None and not heartbeat.done():
                    heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat
                run.event_seq = emitter.sequence
                await self.store.save_run(run)
                await self.store.release_run_lease(run.id, self._worker_id)
                if self._run_tasks.get(run.id) is asyncio.current_task():
                    self._run_tasks.pop(run.id, None)

    async def stream_events(
        self,
        run_id: str,
        *,
        identity: Identity,
        after_seq: int = 0,
        poll_interval_seconds: float = 0.2,
    ) -> AsyncIterator[Event]:
        """Replay persisted events and follow a detached Run until a client-visible boundary."""
        run = await self.store.get_run(run_id)
        session = await self.store.get_session(run.session_id)
        self._authorize(session, identity)
        cursor = after_seq
        while True:
            events = await self.store.list_events(run_id, after_seq=cursor)
            for event in events:
                cursor = event.seq
                yield event
            run = await self.store.get_run(run_id)
            if run.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.WAITING_INPUT,
            } and not await self.store.list_events(run_id, after_seq=cursor):
                return
            await asyncio.sleep(poll_interval_seconds)

    async def stream_run(
        self,
        *,
        session_id: str,
        content: str,
        identity: Identity,
        deps: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[Event]:
        session = await self.store.get_session(session_id)
        self._authorize(session, identity)
        if not content.strip():
            raise ValueError("Run input cannot be empty")
        async with self._session_locks[session.id]:
            active = await self.store.find_active_run(session.id)
            if active is not None:
                if active.status is RunStatus.WAITING_INPUT:
                    await self.cancel_run(active.id, identity=identity)
                else:
                    raise ActiveRunConflict(active)
            await self.store.append_message(session.id, Message(role="user", content=content))
            run = Run(session_id=session.id, status=RunStatus.RUNNING, input=content)
            await self.store.create_run(run)
            emitter = EventEmitter(session.id, run.id)
            task = asyncio.current_task()
            if task is not None:
                self._run_tasks[run.id] = task
            try:
                yield await self._emit(emitter, "run.started", status=run.status)
                run_activity: RunActivity | None = None
                agent = self.agents[session.agent_name]
                if agent.run_activity is not None:
                    activity_ctx = ToolContext(
                        identity=identity,
                        session_id=session.id,
                        run_id=run.id,
                        current_input=content,
                        resource=session.resource,
                        deps=dict(deps or {}),
                    )
                    activity_value = agent.run_activity(activity_ctx, content)
                    if inspect.isawaitable(activity_value):
                        run_activity = await cast(Awaitable[RunActivity], activity_value)
                    else:
                        run_activity = activity_value
                    yield await self._emit(
                        emitter,
                        "activity.updated",
                        id=f"run:{run.id}:model-wait",
                        step=0,
                        kind="progress",
                        message=run_activity.message,
                        status="running",
                        public=dict(run_activity.public),
                    )
                async for event in self._safe_drive(
                    session,
                    run,
                    identity,
                    dict(deps or {}),
                    emitter,
                    run_activity=run_activity,
                ):
                    yield event
            finally:
                if self._run_tasks.get(run.id) is task:
                    self._run_tasks.pop(run.id, None)
                run.event_seq = emitter.sequence
                await self.store.save_run(run)

    async def stream_response(
        self,
        *,
        run_id: str,
        response: InteractionResponse,
        identity: Identity,
        deps: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[Event]:
        initial_run = await self.store.get_run(run_id)
        session = await self.store.get_session(initial_run.session_id)
        self._authorize(session, identity)
        async with self._session_locks[session.id]:
            run = await self.store.get_run(run_id)
            pending = run.pending_call
            if run.status != RunStatus.WAITING_INPUT or pending is None:
                raise ValueError("Run is not waiting for input")
            if pending.interaction.id != response.interaction_id:
                raise ValueError("Interaction does not belong to this run")
            response_value = self._normalize_interaction_response(
                pending.interaction, response.value
            )

            emitter = EventEmitter(session.id, run.id, sequence=run.event_seq)
            task = asyncio.current_task()
            if task is not None:
                self._run_tasks[run.id] = task
            try:
                pending.interaction.resolved = True
                yield await self._emit(
                    emitter,
                    "interaction.resolved",
                    interaction_id=pending.interaction.id,
                    kind=pending.interaction.kind,
                )
                run.status = RunStatus.RUNNING
                run.pending_call = None
                await self.store.save_run(run)

                ctx = ToolContext(
                    identity=identity,
                    session_id=session.id,
                    run_id=run.id,
                    current_input=run.input,
                    resource=session.resource,
                    deps=dict(deps or {}),
                )
                call = pending.call
                trusted_response = (
                    TrustedInteractionResponse(
                        interaction_id=pending.interaction.id,
                        kind=pending.interaction.kind,
                        prompt=pending.interaction.prompt,
                        value=response_value,
                    )
                    if pending.interaction.kind != "approval"
                    else None
                )
                drive_response: TrustedInteractionResponse | None = trusted_response
                if pending.kind == "approval":
                    approved = response_value is True or str(response_value).lower() in {
                        "approve",
                        "yes",
                        "true",
                    }
                    if approved:
                        tool = self.agents[session.agent_name].tool_map()[call.name]
                        async for event in self._execute_tool(
                            session, run, ctx, tool, call, emitter
                        ):
                            yield event
                    else:
                        await self.store.append_message(
                            session.id,
                            Message(
                                role="tool",
                                content=ToolError(
                                    code="rejected",
                                    message="The user rejected this operation.",
                                ).for_model(),
                                tool_call_id=call.id,
                                name=call.name,
                            ),
                        )
                    drive_response = None
                    if run.status == RunStatus.WAITING_INPUT:
                        return
                elif pending.kind == "continuation":
                    assert trusted_response is not None
                    tool = self.agents[session.agent_name].tool_map()[call.name]
                    continuation_ctx = replace(ctx, interaction_response=trusted_response)
                    async for event in self._execute_tool(
                        session, run, continuation_ctx, tool, call, emitter
                    ):
                        yield event
                    drive_response = None
                    if run.status == RunStatus.WAITING_INPUT:
                        return
                else:
                    await self.store.append_message(
                        session.id,
                        Message(
                            role="tool",
                            content=__import__("json").dumps(
                                {"user_response": response_value}, ensure_ascii=False
                            ),
                            tool_call_id=call.id,
                            name="request_input",
                        ),
                    )
                async for event in self._safe_drive(
                    session,
                    run,
                    identity,
                    dict(deps or {}),
                    emitter,
                    interaction_response=drive_response,
                ):
                    yield event
            finally:
                if self._run_tasks.get(run.id) is task:
                    self._run_tasks.pop(run.id, None)
                run.event_seq = emitter.sequence
                await self.store.save_run(run)

    async def submit_response(
        self,
        *,
        run_id: str,
        response: InteractionResponse,
        identity: Identity,
        deps: Mapping[str, Any] | None = None,
    ) -> Run:
        """Validate an answer and detach its continuation from the HTTP request."""
        run = await self.store.get_run(run_id)
        session = await self.store.get_session(run.session_id)
        self._authorize(session, identity)
        pending = run.pending_call
        if run.status is not RunStatus.WAITING_INPUT or pending is None:
            existing_task = self._run_tasks.get(run_id)
            if existing_task is not None and not existing_task.done():
                return run
            raise ValueError("Run is not waiting for input")
        if pending.interaction.id != response.interaction_id:
            raise ValueError("Interaction does not belong to this run")
        self._normalize_interaction_response(pending.interaction, response.value)
        existing_task = self._run_tasks.get(run_id)
        if existing_task is None or existing_task.done():

            async def consume() -> None:
                async for _event in self.stream_response(
                    run_id=run_id,
                    response=response,
                    identity=identity,
                    deps=deps,
                ):
                    pass

            self._run_tasks[run_id] = asyncio.create_task(consume())
        return run

    async def _safe_drive(
        self,
        session: Session,
        run: Run,
        identity: Identity,
        deps: Mapping[str, Any],
        emitter: EventEmitter,
        interaction_response: TrustedInteractionResponse | None = None,
        run_activity: RunActivity | None = None,
    ) -> AsyncIterator[Event]:
        try:
            async for event in self._drive(
                session,
                run,
                identity,
                deps,
                emitter,
                interaction_response,
                run_activity,
            ):
                yield event
        except asyncio.CancelledError:
            run.status = RunStatus.CANCELLED
            run.error = "Run cancelled"
            await self._emit(emitter, "run.cancelled", status=run.status)
            run.event_seq = emitter.sequence
            await self.store.save_run(run)
            raise
        except Exception as exc:
            logger.exception(
                "agent run failed session_id=%s run_id=%s error_type=%s",
                session.id,
                run.id,
                type(exc).__name__,
            )
            run.status = RunStatus.FAILED
            run.error = str(exc)
            await self.store.save_run(run)
            error_code = "model_step_timeout" if isinstance(exc, TimeoutError) else "agent_failed"
            yield await self._emit(
                emitter,
                "run.failed",
                status=run.status,
                error=(
                    "Model response timed out"
                    if error_code == "model_step_timeout"
                    else "Agent execution failed"
                ),
                code=error_code,
                retryable=error_code == "model_step_timeout",
            )

    async def _drive(
        self,
        session: Session,
        run: Run,
        identity: Identity,
        deps: Mapping[str, Any],
        emitter: EventEmitter,
        interaction_response: TrustedInteractionResponse | None = None,
        run_activity: RunActivity | None = None,
    ) -> AsyncIterator[Event]:
        agent = self.agents[session.agent_name]
        ctx = ToolContext(
            identity=identity,
            session_id=session.id,
            run_id=run.id,
            current_input=run.input,
            resource=session.resource,
            deps=deps,
            interaction_response=interaction_response,
        )
        while run.model_steps < self.max_model_steps:
            run.model_steps += 1
            step = run.model_steps
            await self.store.save_run(run)
            assembled = await self._assemble_prompt(session, agent, ctx)
            yield await self._emit(emitter, "context.usage", **assembled.usage)
            yield await self._emit(emitter, "model.started", step=step)
            completed: ModelCompleted | None = None
            stream = self.model.stream(
                messages=assembled.messages,
                tools=assembled.tool_schemas,
                model=agent.model,
            ).__aiter__()
            pending = asyncio.ensure_future(anext(stream))
            step_started_at = asyncio.get_running_loop().time()
            attempt_started_at = step_started_at
            wait_delay = self.model_wait_progress_after_seconds
            progress_emitted = run_activity is not None
            wait_updates = 0
            try:
                while True:
                    now_float = asyncio.get_running_loop().time()
                    elapsed_float = now_float - attempt_started_at
                    if (
                        self.model_step_timeout_seconds is not None
                        and elapsed_float >= self.model_step_timeout_seconds
                    ):
                        raise TimeoutError("Model attempt exceeded its runtime deadline")
                    total_elapsed = now_float - step_started_at
                    if (
                        self.model_step_total_timeout_seconds is not None
                        and total_elapsed >= self.model_step_total_timeout_seconds
                    ):
                        raise TimeoutError("Model step exceeded its total runtime deadline")
                    timeout = wait_delay
                    if self.model_step_timeout_seconds is not None:
                        timeout = min(
                            timeout,
                            self.model_step_timeout_seconds - elapsed_float,
                        )
                    if self.model_step_total_timeout_seconds is not None:
                        timeout = min(
                            timeout,
                            self.model_step_total_timeout_seconds - total_elapsed,
                        )
                    done, _ = await asyncio.wait({pending}, timeout=timeout)
                    if not done:
                        wait_updates += 1
                        elapsed_float = asyncio.get_running_loop().time() - step_started_at
                        elapsed = max(1, int(elapsed_float))
                        message = (
                            run_activity.waiting_message
                            if run_activity is not None and step == 1 and wait_updates == 1
                            else run_activity.long_wait_message
                            if run_activity is not None and step == 1
                            else run_activity.continuation_message
                            if run_activity is not None
                            else "Still working on the request"
                        )
                        public = dict(run_activity.public) if run_activity is not None else {}
                        public["elapsed_seconds"] = elapsed
                        yield await self._emit(
                            emitter,
                            "activity.updated",
                            id=f"run:{run.id}:model-wait",
                            step=step,
                            kind="progress",
                            message=message,
                            status="running",
                            public=public,
                        )
                        progress_emitted = True
                        wait_delay = self.model_wait_progress_interval_seconds
                        continue
                    try:
                        model_event = pending.result()
                    except StopAsyncIteration:
                        break
                    pending = asyncio.ensure_future(anext(stream))
                    wait_delay = self.model_wait_progress_interval_seconds
                    if isinstance(model_event, TextDelta):
                        yield await self._emit(emitter, "message.delta", delta=model_event.text)
                    elif isinstance(model_event, ModelRetrying):
                        attempt_started_at = asyncio.get_running_loop().time()
                        yield await self._emit(emitter, "message.reset")
                        retry_details = {
                            key: value
                            for key, value in {
                                "tool_name": model_event.tool_name,
                                "arguments_chars": model_event.arguments_chars,
                                "error_kind": model_event.error_kind,
                                "error_position": model_event.error_position,
                            }.items()
                            if value is not None
                        }
                        yield await self._emit(
                            emitter,
                            "model.retrying",
                            step=step,
                            retry=model_event.retry,
                            max_retries=model_event.max_retries,
                            delay_seconds=model_event.delay_seconds,
                            reason=model_event.reason,
                            **retry_details,
                        )
                    else:
                        completed = model_event
            finally:
                if not pending.done():
                    pending.cancel()
                    with suppress(asyncio.CancelledError):
                        await pending
                close = getattr(stream, "aclose", None)
                if close is not None:
                    with suppress(Exception):
                        await close()
            if progress_emitted:
                public = dict(run_activity.public) if run_activity is not None else {}
                yield await self._emit(
                    emitter,
                    "activity.updated",
                    id=f"run:{run.id}:model-wait",
                    step=step,
                    kind="progress",
                    message=(
                        run_activity.message
                        if run_activity is not None and step == 1
                        else run_activity.continuation_message
                        if run_activity is not None
                        else "Request processing continued"
                    ),
                    status="completed",
                    public=public,
                )
            if completed is None:
                raise RuntimeError("Model stream ended without a completion")

            if completed.tool_calls:
                outcome = "tools"
            elif completed.text.strip():
                outcome = "final"
            else:
                outcome = "empty"
            yield await self._emit(
                emitter,
                "model.completed",
                step=step,
                outcome=outcome,
                finish_reason=completed.finish_reason,
                provider_request_id=completed.provider_request_id,
            )
            if completed.usage:
                calibrated = calibrate_context_usage(
                    assembled.usage,
                    provider_usage=completed.usage,
                    budget=self.context_budget_tokens,
                )
                if calibrated.get("source") == "provider":
                    yield await self._emit(emitter, "context.usage", **calibrated)

            # A few compatible providers occasionally finish a stream without content or calls.
            # Retry it as a bounded model step instead of returning a successful empty answer.
            if outcome == "empty":
                continue

            if not completed.tool_calls:
                await self.store.append_message(
                    session.id, Message(role="assistant", content=completed.text)
                )
                run.status = RunStatus.COMPLETED
                await self.store.save_run(run)
                yield await self._emit(emitter, "message.completed", content=completed.text)
                yield await self._emit(
                    emitter, "run.completed", status=run.status, usage=completed.usage
                )
                return

            await self.store.append_message(
                session.id,
                Message(
                    role="assistant",
                    content=completed.text,
                    tool_calls=completed.tool_calls,
                ),
            )

            for call in completed.tool_calls:
                if call.name == "report_progress":
                    message = call.arguments.get("message")
                    if not isinstance(message, str) or not message.strip() or len(message) > 280:
                        await self.store.append_message(
                            session.id,
                            Message(
                                role="tool",
                                content=ToolError(
                                    code="invalid_progress",
                                    message=(
                                        "Progress must be a non-empty string of at most "
                                        "280 characters."
                                    ),
                                ).for_model(),
                                tool_call_id=call.id,
                                name=call.name,
                            ),
                        )
                    else:
                        public_message = message.strip()
                        await self.store.append_message(
                            session.id,
                            Message(
                                role="tool",
                                content='{"reported":true}',
                                tool_call_id=call.id,
                                name=call.name,
                            ),
                        )
                        yield await self._emit(
                            emitter,
                            "activity.updated",
                            id=call.id,
                            step=step,
                            kind="progress",
                            message=public_message,
                            status="completed",
                        )
                    continue

                if call.name == "request_input":
                    try:
                        interaction = self._input_interaction(call)
                    except (ValueError, ValidationError) as exc:
                        await self.store.append_message(
                            session.id,
                            Message(
                                role="tool",
                                content=ToolError(
                                    code="invalid_interaction", message=str(exc)
                                ).for_model(),
                                tool_call_id=call.id,
                                name=call.name,
                            ),
                        )
                        yield await self._emit(
                            emitter,
                            "tool.failed",
                            call_id=call.id,
                            name=call.name,
                            title="Request input",
                            code="invalid_interaction",
                            message="The question could not be displayed",
                        )
                        continue
                    run.status = RunStatus.WAITING_INPUT
                    run.pending_call = PendingCall(kind="model", call=call, interaction=interaction)
                    await self.store.save_run(run)
                    yield await self._emit(emitter, "run.waiting_input", status=run.status)
                    yield await self._emit(
                        emitter, "interaction.requested", **interaction.model_dump(mode="json")
                    )
                    return

                tool = agent.tool_map().get(call.name)
                if tool is None:
                    await self.store.append_message(
                        session.id,
                        Message(
                            role="tool",
                            content=ToolError(
                                code="unknown_tool", message=f"Unknown tool: {call.name}"
                            ).for_model(),
                            tool_call_id=call.id,
                            name=call.name,
                        ),
                    )
                    continue

                if tool.requires_interaction_response and ctx.interaction_response is None:
                    await self.store.append_message(
                        session.id,
                        Message(
                            role="tool",
                            content=ToolError(
                                code="interaction_required",
                                message=(
                                    "This tool requires the user's latest interaction response."
                                ),
                            ).for_model(),
                            tool_call_id=call.id,
                            name=tool.name,
                        ),
                    )
                    yield await self._emit(
                        emitter,
                        "tool.failed",
                        call_id=call.id,
                        name=tool.name,
                        title=tool.title,
                        code="interaction_required",
                        message="Waiting for a user answer",
                    )
                    continue

                try:
                    normalized_call = call.model_copy(
                        update={"arguments": tool.normalize(call.arguments)}
                    )
                except ValidationError:
                    normalized_call = call
                if (
                    tool.effect is not ToolEffect.READ
                    and tool.deduplicate
                    and await self._has_successful_duplicate(session.id, tool, normalized_call)
                ):
                    message = "Equivalent tool call already completed; duplicate was skipped"
                    await self.store.append_message(
                        session.id,
                        Message(
                            role="tool",
                            content=json.dumps(
                                {"deduplicated": True, "message": message},
                                ensure_ascii=False,
                            ),
                            tool_call_id=call.id,
                            name=tool.name,
                        ),
                    )
                    yield await self._emit(
                        emitter,
                        "tool.completed",
                        call_id=call.id,
                        name=tool.name,
                        title=tool.title,
                        message=message,
                        deduplicated=True,
                    )
                    continue
                call = normalized_call

                if tool.approval is ApprovalPolicy.ALWAYS:
                    try:
                        validated = tool.normalize(call.arguments)
                        confirmation = await tool.confirmation_prompt(ctx, validated)
                        call = call.model_copy(update={"arguments": validated})
                    except ValidationError as exc:
                        await self.store.append_message(
                            session.id,
                            Message(
                                role="tool",
                                content=validation_error_message(exc),
                                tool_call_id=call.id,
                                name=tool.name,
                            ),
                        )
                        yield await self._emit(
                            emitter,
                            "tool.failed",
                            call_id=call.id,
                            name=tool.name,
                            code="invalid_arguments",
                            message="Tool arguments failed validation",
                        )
                        continue
                    except Exception as exc:
                        await self.store.append_message(
                            session.id,
                            Message(
                                role="tool",
                                content=ToolError(
                                    code="approval_unavailable",
                                    message=str(exc),
                                    retryable=True,
                                ).for_model(),
                                tool_call_id=call.id,
                                name=tool.name,
                            ),
                        )
                        yield await self._emit(
                            emitter,
                            "tool.failed",
                            call_id=call.id,
                            name=tool.name,
                            title=tool.title,
                            code="approval_unavailable",
                            message="Operation is not ready for approval",
                        )
                        continue
                    interaction = Interaction(
                        kind="approval",
                        prompt=confirmation,
                        options=[
                            InteractionOption(id="approve", label="Approve"),
                            InteractionOption(id="reject", label="Reject"),
                        ],
                        tool_call_id=call.id,
                    )
                    run.status = RunStatus.WAITING_INPUT
                    run.pending_call = PendingCall(
                        kind="approval", call=call, interaction=interaction
                    )
                    await self.store.save_run(run)
                    yield await self._emit(emitter, "run.waiting_input", status=run.status)
                    yield await self._emit(
                        emitter,
                        "interaction.requested",
                        **interaction.model_dump(mode="json"),
                        tool={"name": tool.name, "title": tool.title, "arguments": call.arguments},
                    )
                    return

                async for event in self._execute_tool(session, run, ctx, tool, call, emitter):
                    yield event
                if run.status in {RunStatus.WAITING_INPUT, RunStatus.WAITING_EXTERNAL}:
                    return
                if tool.requires_interaction_response:
                    ctx.interaction_response = None

        run.status = RunStatus.FAILED
        run.error = f"Agent exceeded the maximum of {self.max_model_steps} model steps"
        await self.store.save_run(run)
        yield await self._emit(
            emitter,
            "run.failed",
            status=run.status,
            code="model_step_limit_exceeded",
            error="The assistant could not finish this request.",
            retryable=True,
        )

    async def _has_successful_duplicate(
        self,
        session_id: str,
        tool: Tool,
        call: ToolCall,
    ) -> bool:
        messages = list(await self.store.list_messages(session_id))
        matching_ids: set[str] = set()
        for message in messages:
            if message.role != "assistant":
                continue
            for previous in message.tool_calls:
                if previous.id == call.id or previous.name != call.name:
                    continue
                try:
                    arguments = tool.normalize(previous.arguments)
                except ValidationError:
                    continue
                if arguments == call.arguments:
                    matching_ids.add(previous.id)
        for message in messages:
            if message.role != "tool" or message.tool_call_id not in matching_ids:
                continue
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("tool_status") == "success":
                return True
        return False

    @staticmethod
    def _input_interaction(call: ToolCall) -> Interaction:
        kind = call.arguments.get("kind", "text")
        if kind == "form":
            questions = []
            for item in call.arguments.get("questions", []):
                if not isinstance(item, dict):
                    continue
                normalized = dict(item)
                normalized["options"] = [
                    option for option in normalized.get("options", []) if isinstance(option, dict)
                ]
                questions.append(InteractionQuestion.model_validate(normalized))
            if not 1 <= len(questions) <= 8:
                raise ValueError("Form interactions require between 1 and 8 questions")
            question_ids = [question.id for question in questions]
            if len(question_ids) != len(set(question_ids)):
                raise ValueError("Form question ids must be unique")
            for question in questions:
                if not question.id.strip() or not question.prompt.strip():
                    raise ValueError("Form question ids and prompts cannot be empty")
                if question.kind == "choice" and not question.options and not question.allow_custom:
                    raise ValueError("Choice questions require options or allow_custom=true")
                if question.kind == "text" and (
                    question.options or question.multiple or question.allow_custom
                ):
                    raise ValueError("Text questions cannot declare choice properties")
                if (
                    question.kind == "text"
                    and len(_NUMBERED_CHOICE_RE.findall(question.prompt)) >= 2
                ):
                    raise ValueError(
                        "Text questions cannot embed numbered options; use kind=choice"
                    )
            return Interaction(
                kind="form",
                prompt=str(call.arguments.get("prompt", "Please answer these questions.")),
                questions=questions,
                tool_call_id=call.id,
            )

        options = [
            InteractionOption.model_validate(item)
            for item in call.arguments.get("options", [])
            if isinstance(item, dict)
        ]
        multiple = bool(call.arguments.get("multiple", False))
        allow_custom = bool(call.arguments.get("allow_custom", False))
        allow_skip = bool(call.arguments.get("allow_skip", False))
        skip_label = call.arguments.get("skip_label")
        if kind == "choice" and not options and not allow_custom and not allow_skip:
            kind = "text"
        if kind != "choice":
            multiple = False
            allow_custom = False
            allow_skip = False
            skip_label = None
        prompt = str(call.arguments.get("prompt", "Please provide more information."))
        if kind == "text" and len(_NUMBERED_CHOICE_RE.findall(prompt)) >= 2:
            raise ValueError("Text interactions cannot embed numbered options; use kind=choice")
        return Interaction(
            kind=kind,
            prompt=prompt,
            options=options,
            multiple=multiple,
            allow_custom=allow_custom,
            allow_skip=allow_skip,
            skip_label=str(skip_label).strip() if skip_label else None,
            tool_call_id=call.id,
        )

    @staticmethod
    def _normalize_interaction_response(interaction: Interaction, value: Any) -> Any:
        if interaction.kind == "approval":
            return value
        if interaction.kind == "form":
            if not isinstance(value, dict) or not isinstance(value.get("answers"), dict):
                raise ValueError("Form interaction response must contain an answers object")
            answers = value["answers"]
            question_ids = {question.id for question in interaction.questions}
            if any(key not in question_ids for key in answers):
                raise ValueError("Form response contains an unknown question")
            normalized: dict[str, Any] = {}
            for question in interaction.questions:
                if question.id not in answers or answers[question.id] is None:
                    if question.required:
                        raise ValueError(
                            f"Form response is missing required question {question.id}"
                        )
                    continue
                normalized[question.id] = Runtime._normalize_input_value(
                    kind=question.kind,
                    options=question.options,
                    multiple=question.multiple,
                    allow_custom=question.allow_custom,
                    allow_skip=False,
                    value=answers[question.id],
                )
            return {"answers": normalized}

        return Runtime._normalize_input_value(
            kind=interaction.kind,
            options=interaction.options,
            multiple=interaction.multiple,
            allow_custom=interaction.allow_custom,
            allow_skip=interaction.allow_skip,
            value=value,
        )

    @staticmethod
    def _normalize_input_value(
        *,
        kind: str,
        options: list[InteractionOption],
        multiple: bool,
        allow_custom: bool,
        allow_skip: bool,
        value: Any,
    ) -> Any:
        if kind == "text":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Text interaction response must be a non-empty string")
            if len(value.strip()) > 4000:
                raise ValueError("Text interaction response exceeds 4000 characters")
            return value.strip()

        if not isinstance(value, dict):
            raise ValueError("Choice interaction response must be an object")
        selected = value.get("selected", [])
        custom = value.get("custom")
        skipped = value.get("skipped", False)
        if not isinstance(skipped, bool):
            raise ValueError("Choice response skipped must be a boolean")
        if skipped and not allow_skip:
            raise ValueError("This choice interaction cannot be skipped")
        if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
            raise ValueError("Choice response selected must be a string array")
        if len(selected) != len(set(selected)):
            raise ValueError("Choice response contains duplicate selections")
        valid_ids = {option.id for option in options}
        if any(item not in valid_ids for item in selected):
            raise ValueError("Choice response contains an unknown option")
        if custom is not None:
            if not isinstance(custom, str):
                raise ValueError("Choice response custom must be a string")
            custom = custom.strip()
            if not custom:
                custom = None
            elif len(custom) > 1000:
                raise ValueError("Choice response custom exceeds 1000 characters")
        if custom is not None and not allow_custom:
            raise ValueError("This choice interaction does not allow a custom answer")
        if skipped and (selected or custom is not None):
            raise ValueError("A skipped choice cannot contain another answer")
        answer_count = len(selected) + (1 if custom is not None else 0)
        if answer_count == 0 and not skipped:
            raise ValueError("Choice interaction response must contain an answer")
        if not multiple and answer_count > 1:
            raise ValueError("This choice interaction accepts only one answer")
        normalized: dict[str, Any] = {"selected": selected}
        if skipped:
            normalized["skipped"] = True
        if custom is not None:
            normalized["custom"] = custom
        return normalized

    def _prepare_tool_interaction(
        self,
        session: Session,
        request: InteractionRequest,
    ) -> tuple[ToolCall, Interaction]:
        agent = self.agents[session.agent_name]
        target = agent.tool_map().get(request.continuation.tool)
        if target is None:
            raise ValueError(f"Unknown interaction continuation: {request.continuation.tool}")
        if not target.requires_interaction_response:
            raise ValueError("Interaction continuation must require an interaction response")
        if target.approval is not ApprovalPolicy.NEVER:
            raise ValueError("Interaction continuation cannot require approval")
        call = ToolCall(
            id=new_id("call"),
            name=target.name,
            arguments=target.normalize(request.continuation.arguments),
        )
        arguments = request.model_dump(mode="json", exclude={"continuation"})
        interaction = self._input_interaction(
            ToolCall(id=call.id, name="request_input", arguments=arguments)
        )
        if interaction.kind != request.kind:
            raise ValueError("Interaction request is incomplete")
        return call, interaction

    async def _execute_tool(
        self,
        session: Session,
        run: Run,
        ctx: ToolContext,
        tool: Tool,
        call: ToolCall,
        emitter: EventEmitter,
    ) -> AsyncIterator[Event]:
        yield await self._emit(
            emitter, "tool.started", call_id=call.id, name=tool.name, title=tool.title
        )
        progress_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def report_progress(message: str, public: Any) -> None:
            await progress_queue.put((message, public))

        execution_ctx = replace(ctx, _progress_reporter=report_progress)
        execution: asyncio.Task[ToolResult] | None = None
        progress_id = f"{call.id}:progress"
        progress_message: str | None = None
        progress_public: Any = None
        try:
            arguments = tool.validate(call.arguments)
            execution = asyncio.create_task(
                asyncio.wait_for(tool(execution_ctx, **arguments), timeout=tool.timeout_seconds)
            )
            while not execution.done() or not progress_queue.empty():
                if not progress_queue.empty():
                    message, public = progress_queue.get_nowait()
                else:
                    next_progress = asyncio.create_task(progress_queue.get())
                    done, _ = await asyncio.wait(
                        {execution, next_progress}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if next_progress not in done:
                        next_progress.cancel()
                        with suppress(asyncio.CancelledError):
                            await next_progress
                        continue
                    message, public = next_progress.result()
                progress_message = message
                progress_public = public
                yield await self._emit(
                    emitter,
                    "activity.updated",
                    id=progress_id,
                    step=run.model_steps,
                    kind="progress",
                    message=message,
                    status="running",
                    public=public,
                )
            result = await execution
            if progress_message is not None:
                yield await self._emit(
                    emitter,
                    "activity.updated",
                    id=progress_id,
                    step=run.model_steps,
                    kind="progress",
                    message=progress_message,
                    status="completed" if result.succeeded else "failed",
                    public=progress_public,
                )
            if result.succeeded and result.deferred is not None:
                run.status = RunStatus.WAITING_EXTERNAL
                run.pending_external = PendingExternal(call=call, task=result.deferred)
                await self.store.save_run(run)
                yield await self._emit(
                    emitter,
                    "external_task.started",
                    task_id=result.deferred.task_id,
                    call_id=call.id,
                    message=result.deferred.message,
                    public=result.deferred.public,
                )
                yield await self._emit(emitter, "run.waiting_external", status=run.status)
                return
            prepared_interaction = (
                self._prepare_tool_interaction(session, result.interaction)
                if result.succeeded and result.interaction is not None
                else None
            )
            await self.store.append_message(
                session.id,
                Message(
                    role="tool",
                    content=result.for_model(),
                    tool_call_id=call.id,
                    name=tool.name,
                ),
            )
            if result.succeeded:
                yield await self._emit(
                    emitter,
                    "tool.completed",
                    call_id=call.id,
                    name=tool.name,
                    title=tool.title,
                    message=result.message,
                    public=result.public,
                )
                if prepared_interaction is not None:
                    continuation, interaction = prepared_interaction
                    await self.store.append_message(
                        session.id,
                        Message(role="assistant", tool_calls=[continuation]),
                    )
                    run.status = RunStatus.WAITING_INPUT
                    run.pending_call = PendingCall(
                        kind="continuation",
                        call=continuation,
                        interaction=interaction,
                    )
                    await self.store.save_run(run)
                    yield await self._emit(emitter, "run.waiting_input", status=run.status)
                    yield await self._emit(
                        emitter,
                        "interaction.requested",
                        **interaction.model_dump(mode="json"),
                    )
            else:
                yield await self._emit(
                    emitter,
                    "tool.failed",
                    call_id=call.id,
                    name=tool.name,
                    title=tool.title,
                    code="tool_rejected",
                    message=result.message or "Tool could not complete the operation",
                )
        except ValidationError as exc:
            if progress_message is not None:
                yield await self._emit(
                    emitter,
                    "activity.updated",
                    id=progress_id,
                    step=run.model_steps,
                    kind="progress",
                    message=progress_message,
                    status="failed",
                    public=progress_public,
                )
            content = validation_error_message(exc)
            await self.store.append_message(
                session.id,
                Message(role="tool", content=content, tool_call_id=call.id, name=tool.name),
            )
            yield await self._emit(
                emitter,
                "tool.failed",
                call_id=call.id,
                name=tool.name,
                title=tool.title,
                code="invalid_arguments",
                message="Tool arguments failed validation",
            )
        except TimeoutError:
            if progress_message is not None:
                yield await self._emit(
                    emitter,
                    "activity.updated",
                    id=progress_id,
                    step=run.model_steps,
                    kind="progress",
                    message=progress_message,
                    status="failed",
                    public=progress_public,
                )
            content = ToolError(
                code="timeout", message=f"Tool timed out after {tool.timeout_seconds} seconds"
            ).for_model()
            await self.store.append_message(
                session.id,
                Message(role="tool", content=content, tool_call_id=call.id, name=tool.name),
            )
            yield await self._emit(
                emitter,
                "tool.failed",
                call_id=call.id,
                name=tool.name,
                title=tool.title,
                code="timeout",
                message="Tool execution timed out",
            )
        except Exception as exc:
            if progress_message is not None:
                yield await self._emit(
                    emitter,
                    "activity.updated",
                    id=progress_id,
                    step=run.model_steps,
                    kind="progress",
                    message=progress_message,
                    status="failed",
                    public=progress_public,
                )
            content = ToolError(code="tool_error", message=str(exc)).for_model()
            await self.store.append_message(
                session.id,
                Message(role="tool", content=content, tool_call_id=call.id, name=tool.name),
            )
            yield await self._emit(
                emitter,
                "tool.failed",
                call_id=call.id,
                name=tool.name,
                title=tool.title,
                code="tool_error",
                message="Tool execution failed",
            )
        finally:
            if execution is not None and not execution.done():
                execution.cancel()
                with suppress(asyncio.CancelledError):
                    await execution
