import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Any

import pytest

from rungent import (
    Agent,
    DeferredRequest,
    Identity,
    InteractionRequest,
    InteractionResponse,
    RunActivity,
    Runtime,
    ToolContext,
    ToolContinuation,
    ToolResult,
    tool,
)
from rungent.llm import ModelCompleted, ModelEvent, ModelRetrying, TextDelta
from rungent.state import InteractionOption, InteractionQuestion, RunStatus, ToolCall
from rungent.store import MemoryStore
from rungent.testing import ScriptedModel


async def collect(stream):
    return [event async for event in stream]


class PausedModel:
    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def stream(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        model: str | None = None,
    ) -> AsyncIterator[ModelEvent]:
        await self.release.wait()
        yield ModelCompleted(text="Done")


class DeferredResumeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.resumed = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        model: str | None = None,
    ) -> AsyncIterator[ModelEvent]:
        self.calls += 1
        if self.calls == 1:
            yield ModelCompleted(tool_calls=[ToolCall(id="begin", name="begin_job", arguments={})])
            return
        self.resumed.set()
        await self.release.wait()
        yield ModelCompleted(text="Done")


def runtime_with(responses, tools=()):
    return Runtime(
        agents=[Agent(name="assistant", instructions="Use tools.", tools=list(tools))],
        model=ScriptedModel(responses),
        store=MemoryStore(),
    )


async def test_model_wait_progress_is_immediate_persisted_and_reuses_one_activity():
    model = PausedModel()
    agent = Agent(
        name="assistant",
        instructions="Answer.",
        run_activity=lambda _ctx, _content: RunActivity(
            message="Received the request",
            waiting_message="Still preparing the plan",
            long_wait_message="The request is detailed; still preparing the plan",
            continuation_message="Preparing the result",
            public={"stage": "planning"},
        ),
    )
    runtime = Runtime(
        agents=[agent],
        model=model,
        store=MemoryStore(),
        model_wait_progress_after_seconds=0.01,
        model_wait_progress_interval_seconds=0.01,
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    stream = runtime.stream_run(session_id=session.id, content="Plan a trip", identity=identity)

    started = await anext(stream)
    initial = await anext(stream)
    usage = await anext(stream)
    model_started = await anext(stream)
    waiting = await asyncio.wait_for(anext(stream), timeout=0.2)
    long_waiting = await asyncio.wait_for(anext(stream), timeout=0.2)

    assert started.type == "run.started"
    assert usage.type == "context.usage"
    assert usage.data["source"] == "estimated"
    assert usage.data["used_tokens"] > 0
    assert initial.type == waiting.type == long_waiting.type == "activity.updated"
    assert initial.data["id"] == waiting.data["id"] == long_waiting.data["id"]
    assert initial.data["status"] == "running"
    assert waiting.data["message"] == "Still preparing the plan"
    assert long_waiting.data["message"].startswith("The request is detailed")
    assert model_started.type == "model.started"

    model.release.set()
    remaining = await collect(stream)
    completed_activity = next(
        event
        for event in remaining
        if event.type == "activity.updated" and event.data["status"] == "completed"
    )
    persisted = await runtime.get_run_events(started.run_id, identity=identity)
    assert completed_activity.data["id"] == initial.data["id"]
    assert [event.seq for event in persisted] == list(range(1, len(persisted) + 1))


async def test_model_step_deadline_fails_instead_of_leaving_a_run_active():
    model = PausedModel()
    runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Answer.")],
        model=model,
        store=MemoryStore(),
        model_wait_progress_after_seconds=0.005,
        model_wait_progress_interval_seconds=0.005,
        model_step_timeout_seconds=0.02,
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)

    events = await collect(
        runtime.stream_run(session_id=session.id, content="Plan a trip", identity=identity)
    )

    assert events[-1].type == "run.failed"
    assert events[-1].data["code"] == "model_step_timeout"
    assert events[-1].data["retryable"] is True
    run = await runtime.store.get_run(events[-1].run_id)
    assert run.status == RunStatus.FAILED
    assert "deadline" in (run.error or "")


async def test_plain_response_completes_run_and_persists_messages():
    runtime = runtime_with([ModelCompleted(text="Hello")])
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Hi", identity=identity)
    )

    assert [event.type for event in events] == [
        "run.started",
        "context.usage",
        "model.started",
        "message.delta",
        "model.completed",
        "message.completed",
        "run.completed",
    ]
    _, messages = await runtime.get_session(session.id, identity=identity)
    assert [(message.role, message.content) for message in messages] == [
        ("user", "Hi"),
        ("assistant", "Hello"),
    ]


async def test_deferred_tool_waits_without_model_step_and_resumes_once():
    @tool(effect="write", approval="never")
    async def start_job(ctx: ToolContext) -> ToolResult:
        """Start a durable host job."""
        return ToolResult(
            deferred=DeferredRequest(
                task_id="job_1",
                message="Waiting for route observations",
                public={"stage": "observing"},
            )
        )

    runtime = runtime_with(
        [
            ModelCompleted(tool_calls=[ToolCall(id="call_job", name="start_job", arguments={})]),
            ModelCompleted(text="The route policy is complete."),
        ],
        tools=[start_job],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)

    started = await collect(
        runtime.stream_run(session_id=session.id, content="Apply the rule", identity=identity)
    )

    assert [event.type for event in started[-2:]] == [
        "external_task.started",
        "run.waiting_external",
    ]
    waiting = await runtime.store.get_run(started[0].run_id)
    assert waiting.status == RunStatus.WAITING_EXTERNAL
    assert waiting.model_steps == 1
    assert waiting.pending_external is not None
    assert waiting.pending_external.task.task_id == "job_1"

    progress = await runtime.report_external_progress(
        waiting.id,
        "job_1",
        identity=identity,
        message="Checked 1 of 2 routes",
        public={"completed": 1, "total": 2},
    )
    assert progress.type == "external_task.progress"

    resumed = await runtime.resume_deferred(
        waiting.id,
        "job_1",
        ToolResult(
            data={"ok": True, "changed": 2},
            message="Updated 2 routes",
            public={"trip_changed": True},
        ),
        identity=identity,
    )
    assert resumed[0].type == "external_task.completed"
    assert resumed[1].type == "tool.completed"
    assert resumed[-1].type == "run.completed"
    completed = await runtime.store.get_run(waiting.id)
    assert completed.status == RunStatus.COMPLETED
    assert completed.model_steps == 2

    with pytest.raises(ValueError, match="not waiting"):
        await runtime.resume_deferred(
            waiting.id,
            "job_1",
            ToolResult(data={"ok": True}),
            identity=identity,
        )


async def test_deferred_tool_rejects_wrong_task_and_cancels_host_job():
    cancelled: list[str] = []

    async def cancel_external(_session, _run, request):
        cancelled.append(request.task_id)

    @tool(effect="write", approval="never")
    async def start_job(ctx: ToolContext) -> ToolResult:
        """Start a durable host job."""
        return ToolResult(deferred=DeferredRequest(task_id="job_cancel", message="Working"))

    runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Use tools.", tools=[start_job])],
        model=ScriptedModel(
            [ModelCompleted(tool_calls=[ToolCall(id="call_job", name="start_job", arguments={})])]
        ),
        store=MemoryStore(),
        external_task_canceller=cancel_external,
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Start", identity=identity)
    )
    run_id = events[0].run_id

    with pytest.raises(ValueError, match="does not belong"):
        await runtime.report_external_progress(
            run_id,
            "wrong",
            identity=identity,
            message="Nope",
        )

    run = await runtime.cancel_run(run_id, identity=identity)
    assert run.status == RunStatus.CANCELLED
    assert cancelled == ["job_cancel"]
    persisted = await runtime.get_run_events(run_id, identity=identity)
    assert [event.type for event in persisted[-2:]] == [
        "external_task.cancelled",
        "run.cancelled",
    ]


async def test_deferred_resume_lease_prevents_recovery_worker_from_interrupting_run():
    @tool(effect="external", approval="never")
    async def begin_job(_ctx: ToolContext) -> ToolResult:
        """Begin durable work."""
        return ToolResult(deferred=DeferredRequest(task_id="job-1", message="Working"))

    model = DeferredResumeModel()
    runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Use tools.", tools=[begin_job])],
        model=model,
        store=MemoryStore(),
    )
    runtime._lease_seconds = 0.3
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    first = await collect(
        runtime.stream_run(session_id=session.id, content="Start", identity=identity)
    )
    run_id = first[0].run_id

    resume = asyncio.create_task(
        runtime.resume_deferred(
            run_id,
            "job-1",
            ToolResult(message="External work finished"),
            identity=identity,
        )
    )
    await asyncio.wait_for(model.resumed.wait(), timeout=0.2)

    await runtime.recover_runs()
    running = await runtime.store.get_run(run_id)
    assert running.status is RunStatus.RUNNING
    assert running.lease_owner is not None
    assert not resume.done()

    model.release.set()
    resumed = await resume
    completed = await runtime.store.get_run(run_id)
    persisted = await runtime.get_run_events(run_id, identity=identity)
    assert resumed[-1].type == "run.completed"
    assert completed.status is RunStatus.COMPLETED
    assert completed.lease_owner is None
    assert [event.seq for event in persisted] == list(range(1, len(persisted) + 1))
    assert all(event.data.get("code") != "run_interrupted" for event in persisted)


async def test_request_input_pauses_and_resumes_same_run():
    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="call_input",
                        name="request_input",
                        arguments={
                            "kind": "choice",
                            "prompt": "Which day?",
                            "options": [
                                {"id": "one", "label": "Day 1"},
                                {"id": "two", "label": "Day 2"},
                            ],
                        },
                    )
                ]
            ),
            ModelCompleted(text="Moved to day 2"),
        ]
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    first = await collect(
        runtime.stream_run(session_id=session.id, content="Move it", identity=identity)
    )
    interaction = first[-1]
    assert interaction.type == "interaction.requested"
    run = await runtime.store.get_run(interaction.run_id)
    assert run.status == RunStatus.WAITING_INPUT

    resumed = await collect(
        runtime.stream_response(
            run_id=run.id,
            response=InteractionResponse(
                interaction_id=interaction.data["id"], value={"selected": ["two"]}
            ),
            identity=identity,
        )
    )
    assert resumed[-1].type == "run.completed"
    assert resumed[0].seq == interaction.seq + 1
    completed_run = await runtime.store.get_run(run.id)
    assert completed_run.status == RunStatus.COMPLETED
    assert completed_run.model_steps == 2
    assert next(event for event in resumed if event.type == "model.started").data["step"] == 2


async def test_request_input_ignores_non_object_choice_options():
    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="ask",
                        name="request_input",
                        arguments={
                            "kind": "choice",
                            "prompt": "Pick one",
                            "options": ["\n", {"id": "one", "label": "One"}],
                        },
                    )
                ]
            )
        ]
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)

    events = await collect(
        runtime.stream_run(session_id=session.id, content="Pick", identity=identity)
    )

    interaction = events[-1]
    assert interaction.type == "interaction.requested"
    assert interaction.data["options"] == [
        {"id": "one", "label": "One", "description": None, "recommended": False}
    ]


async def test_request_input_rejects_numbered_options_inside_text_prompt():
    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="bad-text-choice",
                        name="request_input",
                        arguments={
                            "kind": "text",
                            "prompt": "Choose one:\n1. Walking\n2. Driving\n3. Something else",
                        },
                    )
                ]
            ),
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="valid-choice",
                        name="request_input",
                        arguments={
                            "kind": "choice",
                            "prompt": "Choose one",
                            "options": [
                                {"id": "walking", "label": "Walking"},
                                {"id": "driving", "label": "Driving"},
                            ],
                            "allow_custom": True,
                        },
                    )
                ]
            ),
        ]
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)

    events = await collect(
        runtime.stream_run(session_id=session.id, content="Choose", identity=identity)
    )

    failed = next(event for event in events if event.type == "tool.failed")
    assert failed.data["code"] == "invalid_interaction"
    interaction = events[-1]
    assert interaction.type == "interaction.requested"
    assert interaction.data["kind"] == "choice"
    assert interaction.data["allow_custom"] is True


async def test_tool_directed_interactions_run_frozen_continuations_without_model_steps():
    answers: list[dict] = []

    @tool(
        effect="external",
        approval="never",
        requires_interaction_response=True,
    )
    async def resolve(ctx: ToolContext, draft_id: str) -> ToolResult:
        """Resolve the current issue."""
        assert ctx.interaction_response is not None
        answers.append(ctx.interaction_response.value)
        if len(answers) == 1:
            return ToolResult(
                message="one remains",
                interaction=InteractionRequest(
                    kind="choice",
                    prompt="Second place?",
                    options=[InteractionOption(id="b", label="B")],
                    continuation=ToolContinuation(tool="resolve", arguments={"draft_id": draft_id}),
                ),
            )
        return ToolResult(message="ready")

    @tool(effect="external", approval="never")
    async def prepare(ctx: ToolContext) -> ToolResult:
        """Prepare a draft."""
        return ToolResult(
            message="needs input",
            interaction=InteractionRequest(
                kind="choice",
                prompt="First place?",
                options=[InteractionOption(id="a", label="A", recommended=True)],
                allow_custom=True,
                allow_skip=True,
                skip_label="Skip place",
                continuation=ToolContinuation(tool="resolve", arguments={"draft_id": "draft-1"}),
            ),
        )

    runtime = runtime_with(
        [
            ModelCompleted(tool_calls=[ToolCall(id="prepare", name="prepare", arguments={})]),
            ModelCompleted(text="Ready"),
        ],
        tools=[prepare, resolve],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)

    first = await collect(
        runtime.stream_run(session_id=session.id, content="Import", identity=identity)
    )
    first_interaction = first[-1]
    run = await runtime.store.get_run(first_interaction.run_id)
    assert run.model_steps == 1
    assert first_interaction.data["allow_skip"] is True
    assert first_interaction.data["options"][0]["recommended"] is True

    second = await collect(
        runtime.stream_response(
            run_id=run.id,
            response=InteractionResponse(
                interaction_id=first_interaction.data["id"],
                value={"selected": [], "skipped": True},
            ),
            identity=identity,
        )
    )
    second_interaction = second[-1]
    assert second_interaction.type == "interaction.requested"
    assert not any(event.type == "model.started" for event in second)
    assert (await runtime.store.get_run(run.id)).model_steps == 1

    final = await collect(
        runtime.stream_response(
            run_id=run.id,
            response=InteractionResponse(
                interaction_id=second_interaction.data["id"],
                value={"selected": ["b"]},
            ),
            identity=identity,
        )
    )
    assert final[-1].type == "run.completed"
    assert next(event for event in final if event.type == "model.started").data["step"] == 2
    assert answers == [{"selected": [], "skipped": True}, {"selected": ["b"]}]


async def test_approved_write_can_pause_for_a_continuation_form():
    seen: list[str] = []

    @tool(
        effect="write",
        approval="never",
        requires_interaction_response=True,
    )
    async def complete_create(ctx: ToolContext, name: str) -> ToolResult:
        """Finish the approved create after the user supplies an owner."""
        assert ctx.interaction_response is not None
        seen.append(f"{name}:{ctx.interaction_response.value['answers']['owner']}")
        return ToolResult(message="created")

    @tool(
        effect="write",
        approval="always",
        confirmation="Create {name}?",
    )
    async def create_item(ctx: ToolContext, name: str) -> ToolResult:
        """Create one item after approval."""
        return ToolResult(
            message="need owner",
            interaction=InteractionRequest(
                kind="form",
                prompt="Who owns it?",
                questions=[
                    InteractionQuestion(id="owner", prompt="Owner", kind="text"),
                ],
                continuation=ToolContinuation(tool="complete_create", arguments={"name": name}),
            ),
        )

    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[ToolCall(id="c1", name="create_item", arguments={"name": "box"})]
            ),
            ModelCompleted(text="Created"),
        ],
        tools=[create_item, complete_create],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    first = await collect(
        runtime.stream_run(session_id=session.id, content="Create box", identity=identity)
    )
    approval = first[-1]
    after_approve = await collect(
        runtime.stream_response(
            run_id=approval.run_id,
            response=InteractionResponse(interaction_id=approval.data["id"], value="approve"),
            identity=identity,
        )
    )
    assert after_approve[-1].type == "interaction.requested"
    assert after_approve[-1].data["kind"] == "form"
    assert not any(event.type == "model.started" for event in after_approve)
    final = await collect(
        runtime.stream_response(
            run_id=approval.run_id,
            response=InteractionResponse(
                interaction_id=after_approve[-1].data["id"],
                value={"answers": {"owner": "ada"}},
            ),
            identity=identity,
        )
    )
    assert final[-1].type == "run.completed"
    assert seen == ["box:ada"]


async def test_choice_skip_is_rejected_unless_the_interaction_allows_it():
    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="ask",
                        name="request_input",
                        arguments={
                            "kind": "choice",
                            "prompt": "Choose",
                            "options": [{"id": "one", "label": "One"}],
                        },
                    )
                ]
            )
        ]
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Choose", identity=identity)
    )

    with pytest.raises(ValueError, match="cannot be skipped"):
        await collect(
            runtime.stream_response(
                run_id=events[-1].run_id,
                response=InteractionResponse(
                    interaction_id=events[-1].data["id"],
                    value={"selected": [], "skipped": True},
                ),
                identity=identity,
            )
        )


async def test_tool_directed_interaction_recovers_from_persisted_run_state():
    observed: list[str] = []

    @tool(
        effect="external",
        approval="never",
        requires_interaction_response=True,
    )
    async def resolve(ctx: ToolContext, item_id: str) -> ToolResult:
        """Resolve a persisted issue."""
        assert ctx.interaction_response is not None
        observed.append(f"{item_id}:{ctx.interaction_response.value['custom']}")
        return ToolResult(message="resolved")

    @tool(effect="external", approval="never")
    async def prepare(ctx: ToolContext) -> ToolResult:
        """Prepare one issue."""
        return ToolResult(
            interaction=InteractionRequest(
                kind="choice",
                prompt="Name?",
                allow_custom=True,
                continuation=ToolContinuation(tool="resolve", arguments={"item_id": "item-1"}),
            )
        )

    store = MemoryStore()
    identity = Identity(subject_id="u1")
    first_runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Use tools", tools=[prepare, resolve])],
        model=ScriptedModel(
            [ModelCompleted(tool_calls=[ToolCall(id="prepare", name="prepare", arguments={})])]
        ),
        store=store,
    )
    session = await first_runtime.create_session(identity=identity)
    first = await collect(
        first_runtime.stream_run(session_id=session.id, content="Start", identity=identity)
    )

    recovered_runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Use tools", tools=[prepare, resolve])],
        model=ScriptedModel([ModelCompleted(text="Done")]),
        store=store,
    )
    resumed = await collect(
        recovered_runtime.stream_response(
            run_id=first[-1].run_id,
            response=InteractionResponse(
                interaction_id=first[-1].data["id"],
                value={"selected": [], "custom": "Kyoto"},
            ),
            identity=identity,
        )
    )

    assert resumed[-1].type == "run.completed"
    assert observed == ["item-1:Kyoto"]
    with pytest.raises(ValueError, match="not waiting"):
        await collect(
            recovered_runtime.stream_response(
                run_id=first[-1].run_id,
                response=InteractionResponse(
                    interaction_id=first[-1].data["id"],
                    value={"selected": [], "custom": "Osaka"},
                ),
                identity=identity,
            )
        )


async def test_interaction_bound_tool_is_hidden_then_consumes_response_once():
    observed: list[str] = []
    advertised: list[list[str]] = []

    @tool(
        effect="write",
        approval="never",
        requires_interaction_response=True,
    )
    async def resolve(ctx: ToolContext) -> ToolResult:
        """Consume one trusted answer."""
        assert ctx.interaction_response is not None
        observed.append(ctx.interaction_response.interaction_id)
        return ToolResult(message="resolved")

    class InspectingModel:
        step = 0

        async def stream(self, *, messages, tools, model=None):
            advertised.append([item["function"]["name"] for item in tools])
            self.step += 1
            if self.step == 1:
                yield ModelCompleted(
                    tool_calls=[
                        ToolCall(
                            id="ask",
                            name="request_input",
                            arguments={"kind": "text", "prompt": "Which place?"},
                        )
                    ]
                )
            elif self.step == 2:
                yield ModelCompleted(
                    tool_calls=[
                        ToolCall(id="resolve-once", name="resolve", arguments={}),
                        ToolCall(id="resolve-twice", name="resolve", arguments={}),
                    ]
                )
            else:
                yield ModelCompleted(text="Done")

    runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Use tools", tools=[resolve])],
        model=InspectingModel(),
        store=MemoryStore(),
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    first = await collect(
        runtime.stream_run(session_id=session.id, content="Plan", identity=identity)
    )
    interaction = first[-1]
    resumed = await collect(
        runtime.stream_response(
            run_id=interaction.run_id,
            response=InteractionResponse(
                interaction_id=interaction.data["id"], value="Grand Front Osaka"
            ),
            identity=identity,
        )
    )

    assert "resolve" not in advertised[0]
    assert "resolve" in advertised[1]
    assert "resolve" not in advertised[2]
    assert observed == [interaction.data["id"]]
    failed = next(event for event in resumed if event.type == "tool.failed")
    assert failed.data["code"] == "interaction_required"


async def test_multi_choice_accepts_options_and_custom_text_as_one_explicit_response():
    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="ask",
                        name="request_input",
                        arguments={
                            "kind": "choice",
                            "prompt": "Pick neighborhoods",
                            "options": [
                                {"id": "shibuya", "label": "Shibuya"},
                                {"id": "asakusa", "label": "Asakusa"},
                            ],
                            "multiple": True,
                            "allow_custom": True,
                        },
                    )
                ]
            ),
            ModelCompleted(text="Selected"),
        ]
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Plan", identity=identity)
    )
    interaction = events[-1]
    assert interaction.data["multiple"] is True
    assert interaction.data["allow_custom"] is True

    resumed = await collect(
        runtime.stream_response(
            run_id=interaction.run_id,
            response=InteractionResponse(
                interaction_id=interaction.data["id"],
                value={"selected": ["shibuya"], "custom": "Kichijoji"},
            ),
            identity=identity,
        )
    )
    assert resumed[-1].type == "run.completed"
    _, messages = await runtime.get_session(session.id, identity=identity)
    response_message = next(message for message in messages if message.name == "request_input")
    assert response_message.content == (
        '{"user_response": {"selected": ["shibuya"], "custom": "Kichijoji"}}'
    )


async def test_form_groups_independent_questions_into_one_durable_interaction():
    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="ask-form",
                        name="request_input",
                        arguments={
                            "kind": "form",
                            "prompt": "Trip preferences",
                            "questions": [
                                {
                                    "id": "pace",
                                    "kind": "choice",
                                    "prompt": "Preferred pace?",
                                    "options": [
                                        {"id": "slow", "label": "Slow"},
                                        {"id": "fast", "label": "Fast"},
                                    ],
                                },
                                {
                                    "id": "notes",
                                    "kind": "text",
                                    "prompt": "Anything else?",
                                    "required": False,
                                },
                            ],
                        },
                    )
                ]
            ),
            ModelCompleted(text="Ready"),
        ]
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    interaction = (
        await collect(runtime.stream_run(session_id=session.id, content="Plan", identity=identity))
    )[-1]

    assert interaction.data["kind"] == "form"
    assert [question["id"] for question in interaction.data["questions"]] == ["pace", "notes"]
    resumed = await collect(
        runtime.stream_response(
            run_id=interaction.run_id,
            response=InteractionResponse(
                interaction_id=interaction.data["id"],
                value={"answers": {"pace": {"selected": ["slow"]}}},
            ),
            identity=identity,
        )
    )

    assert resumed[-1].type == "run.completed"
    _, messages = await runtime.get_session(session.id, identity=identity)
    response_message = next(message for message in messages if message.name == "request_input")
    assert response_message.content == (
        '{"user_response": {"answers": {"pace": {"selected": ["slow"]}}}}'
    )


async def test_form_requires_all_required_answers_and_rejects_unknown_questions():
    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="ask-form",
                        name="request_input",
                        arguments={
                            "kind": "form",
                            "prompt": "Preferences",
                            "questions": [{"id": "city", "kind": "text", "prompt": "Which city?"}],
                        },
                    )
                ]
            )
        ]
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    interaction = (
        await collect(runtime.stream_run(session_id=session.id, content="Plan", identity=identity))
    )[-1]

    for value in ({"answers": {}}, {"answers": {"city": "Tokyo", "other": "x"}}):
        try:
            await collect(
                runtime.stream_response(
                    run_id=interaction.run_id,
                    response=InteractionResponse(
                        interaction_id=interaction.data["id"], value=value
                    ),
                    identity=identity,
                )
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid form response was accepted: {value!r}")


async def test_choice_response_rejects_unknown_or_implicit_values():
    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="ask",
                        name="request_input",
                        arguments={
                            "kind": "choice",
                            "prompt": "Pick one",
                            "options": [{"id": "one", "label": "One"}],
                        },
                    )
                ]
            )
        ]
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    interaction = (
        await collect(runtime.stream_run(session_id=session.id, content="Pick", identity=identity))
    )[-1]

    for value in ("one", {"selected": ["unknown"]}, {"selected": []}):
        try:
            await collect(
                runtime.stream_response(
                    run_id=interaction.run_id,
                    response=InteractionResponse(
                        interaction_id=interaction.data["id"], value=value
                    ),
                    identity=identity,
                )
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid choice response was accepted: {value!r}")


async def test_session_cancels_waiting_input_when_a_new_run_starts():
    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="call_input",
                        name="request_input",
                        arguments={"kind": "text", "prompt": "Which city?"},
                    )
                ]
            ),
            ModelCompleted(text="New request"),
        ]
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    first = await collect(
        runtime.stream_run(session_id=session.id, content="Plan", identity=identity)
    )
    assert first[-1].type == "interaction.requested"
    second = await collect(
        runtime.stream_run(session_id=session.id, content="Again", identity=identity)
    )
    cancelled = await runtime.store.get_run(first[-1].run_id)
    assert cancelled.status == RunStatus.CANCELLED
    assert second[-1].type == "run.completed"
    assert second[-1].run_id != first[-1].run_id


async def test_empty_model_completion_is_retried_instead_of_returned_to_user():
    runtime = runtime_with([ModelCompleted(), ModelCompleted(text="Recovered answer")])
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)

    events = await collect(
        runtime.stream_run(session_id=session.id, content="Answer me", identity=identity)
    )

    outcomes = [event.data["outcome"] for event in events if event.type == "model.completed"]
    assert outcomes == ["empty", "final"]
    assert [event.data["content"] for event in events if event.type == "message.completed"] == [
        "Recovered answer"
    ]
    assert events[-1].type == "run.completed"


async def test_approval_executes_frozen_arguments_only_after_acceptance():
    calls: list[tuple[str, int]] = []

    @tool(
        effect="destructive",
        approval="always",
        title="Delete day",
        confirmation="Delete day {day} from trip {trip_id}?",
    )
    async def delete_day(
        ctx: ToolContext,
        trip_id: Annotated[str, "Trip id"],
        day: int,
    ) -> ToolResult:
        """Delete one trip day."""
        calls.append((trip_id, day))
        return ToolResult(message="Deleted")

    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="call_delete",
                        name="delete_day",
                        arguments={"trip_id": "trip-1", "day": "3"},
                    )
                ]
            ),
            ModelCompleted(text="Day deleted"),
        ],
        [delete_day],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    first = await collect(
        runtime.stream_run(session_id=session.id, content="Delete day 3", identity=identity)
    )
    assert calls == []
    interaction = first[-1]
    assert interaction.data["prompt"] == "Delete day 3 from trip trip-1?"
    assert interaction.data["tool"]["arguments"] == {"trip_id": "trip-1", "day": 3}

    resumed = await collect(
        runtime.stream_response(
            run_id=interaction.run_id,
            response=InteractionResponse(interaction_id=interaction.data["id"], value="approve"),
            identity=identity,
        )
    )
    assert calls == [("trip-1", 3)]
    assert [event.type for event in resumed] == [
        "interaction.resolved",
        "tool.started",
        "tool.completed",
        "context.usage",
        "model.started",
        "message.delta",
        "model.completed",
        "message.completed",
        "run.completed",
    ]


async def test_new_message_cancels_waiting_approval_without_executing():
    calls: list[tuple[str, int]] = []

    @tool(
        effect="destructive",
        approval="always",
        confirmation="Delete day {day} from trip {trip_id}?",
    )
    async def delete_day(
        ctx: ToolContext,
        trip_id: Annotated[str, "Trip id"],
        day: Annotated[int, "Day number"],
    ) -> ToolResult:
        """Delete one day from a trip."""
        calls.append((trip_id, day))
        return ToolResult(data={"deleted": day})

    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="del",
                        name="delete_day",
                        arguments={"trip_id": "trip-1", "day": 3},
                    )
                ]
            ),
            ModelCompleted(text="Acknowledged"),
        ],
        [delete_day],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    first = await collect(
        runtime.stream_run(session_id=session.id, content="Delete day 3", identity=identity)
    )
    assert first[-1].type == "interaction.requested"
    second = await collect(
        runtime.stream_run(session_id=session.id, content="确认", identity=identity)
    )
    assert calls == []
    cancelled = await runtime.store.get_run(first[-1].run_id)
    assert cancelled.status == RunStatus.CANCELLED
    assert second[-1].type == "run.completed"
    assert second[-1].run_id != first[-1].run_id


async def test_invalid_destructive_arguments_do_not_request_approval():
    @tool(effect="destructive", approval="always", confirmation="Remove day {day}?")
    async def remove_day(ctx: ToolContext, day: int) -> ToolResult:
        """Remove one day."""
        raise AssertionError("invalid call executed")

    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[ToolCall(id="bad", name="remove_day", arguments={"day": "later"})]
            ),
            ModelCompleted(text="Which day should I remove?"),
        ],
        [remove_day],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Remove a day", identity=identity)
    )
    assert "interaction.requested" not in [event.type for event in events]
    assert "tool.failed" in [event.type for event in events]
    assert events[-1].type == "run.completed"


async def test_failed_confirmation_does_not_create_approval():
    async def not_ready(ctx: ToolContext, draft_id: str) -> str:
        raise RuntimeError("draft is not ready")

    @tool(effect="destructive", approval="always", confirmation=not_ready)
    async def commit(ctx: ToolContext, draft_id: str) -> ToolResult:
        """Commit a ready draft."""
        raise AssertionError("unready draft executed")

    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[ToolCall(id="early", name="commit", arguments={"draft_id": "draft-1"})]
            ),
            ModelCompleted(text="The draft is not ready"),
        ],
        [commit],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Commit", identity=identity)
    )

    assert not any(event.type == "interaction.requested" for event in events)
    failed = next(event for event in events if event.type == "tool.failed")
    assert failed.data["code"] == "approval_unavailable"


async def test_rejected_approval_never_executes_tool():
    called = False

    @tool(effect="destructive", approval="always", confirmation="Erase all data?")
    async def erase(ctx: ToolContext) -> ToolResult:
        """Erase data."""
        nonlocal called
        called = True
        return ToolResult(message="erased")

    runtime = runtime_with(
        [
            ModelCompleted(tool_calls=[ToolCall(id="c", name="erase", arguments={})]),
            ModelCompleted(text="Cancelled"),
        ],
        [erase],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Erase", identity=identity)
    )
    interaction = events[-1]
    resumed = await collect(
        runtime.stream_response(
            run_id=interaction.run_id,
            response=InteractionResponse(interaction_id=interaction.data["id"], value="reject"),
            identity=identity,
        )
    )
    assert called is False
    assert resumed[-1].type == "run.completed"


async def test_session_ownership_is_enforced():
    runtime = runtime_with([ModelCompleted(text="unused")])
    owner = Identity(subject_id="owner", tenant_id="tenant")
    session = await runtime.create_session(identity=owner)
    try:
        await runtime.get_session(
            session.id, identity=Identity(subject_id="other", tenant_id="tenant")
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("foreign identity accessed the session")


async def test_invalid_arguments_are_returned_to_model_without_execution():
    called = False

    @tool(effect="write", approval="never")
    async def set_day(ctx: ToolContext, day: int) -> ToolResult:
        """Set a day."""
        nonlocal called
        called = True
        return ToolResult(message="set")

    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[ToolCall(id="bad", name="set_day", arguments={"day": "nope"})]
            ),
            ModelCompleted(text="Please provide a valid day"),
        ],
        [set_day],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Set day", identity=identity)
    )
    assert called is False
    assert "tool.failed" in [event.type for event in events]
    assert events[-1].type == "run.completed"


async def test_tool_exception_is_isolated_and_model_can_recover():
    @tool(effect="read", approval="never")
    async def lookup(ctx: ToolContext) -> ToolResult:
        """Look up data."""
        raise RuntimeError("private backend detail")

    runtime = runtime_with(
        [
            ModelCompleted(tool_calls=[ToolCall(id="lookup", name="lookup", arguments={})]),
            ModelCompleted(text="The lookup is temporarily unavailable"),
        ],
        [lookup],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Look up", identity=identity)
    )
    failed = next(event for event in events if event.type == "tool.failed")
    assert failed.data["message"] == "Tool execution failed"
    assert events[-1].type == "run.completed"


async def test_runtime_deduplicates_equivalent_successful_write_calls_in_one_run():
    calls: list[int] = []

    @tool(effect="write", approval="never")
    async def add_day(ctx: ToolContext, count: int) -> ToolResult:
        """Add days."""
        calls.append(count)
        return ToolResult(data={"ok": True}, message="added")

    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(id="first", name="add_day", arguments={"count": "1"}),
                    ToolCall(id="duplicate", name="add_day", arguments={"count": 1}),
                ]
            ),
            ModelCompleted(
                tool_calls=[ToolCall(id="again", name="add_day", arguments={"count": 1})]
            ),
            ModelCompleted(text="Done"),
        ],
        [add_day],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Add one day", identity=identity)
    )

    assert calls == [1]
    assert [event.data["call_id"] for event in events if event.type == "tool.started"] == ["first"]
    deduplicated = [
        event
        for event in events
        if event.type == "tool.completed" and event.data.get("deduplicated")
    ]
    assert [event.data["call_id"] for event in deduplicated] == ["duplicate", "again"]


async def test_failed_tool_result_can_be_corrected_and_retried():
    calls = 0

    @tool(effect="write", approval="never")
    async def update(ctx: ToolContext, value: int) -> ToolResult:
        """Update a value."""
        nonlocal calls
        calls += 1
        if calls == 1:
            return ToolResult(data={"ok": False, "error": "temporary"}, message="temporary")
        return ToolResult(data={"ok": True}, message="updated")

    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[ToolCall(id="first", name="update", arguments={"value": 1})]
            ),
            ModelCompleted(
                tool_calls=[ToolCall(id="retry", name="update", arguments={"value": 1})]
            ),
            ModelCompleted(text="Done"),
        ],
        [update],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Update", identity=identity)
    )

    assert calls == 2
    failed = [event for event in events if event.type == "tool.failed"]
    assert len(failed) == 1
    assert failed[0].data["message"] == "temporary"
    assert not any(event.data.get("deduplicated") for event in events)


async def test_tool_receives_only_runtime_validated_latest_interaction_response():
    captured = None

    @tool(effect="write", approval="never")
    async def consume(ctx: ToolContext) -> ToolResult:
        """Consume the current answer."""
        nonlocal captured
        captured = ctx.interaction_response
        return ToolResult(data={"ok": True})

    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="ask",
                        name="request_input",
                        arguments={"kind": "text", "prompt": "Which city?"},
                    )
                ]
            ),
            ModelCompleted(tool_calls=[ToolCall(id="consume", name="consume", arguments={})]),
            ModelCompleted(text="Done"),
        ],
        [consume],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    first = await collect(
        runtime.stream_run(session_id=session.id, content="Plan", identity=identity)
    )
    interaction = first[-1]
    await collect(
        runtime.stream_response(
            run_id=interaction.run_id,
            response=InteractionResponse(interaction_id=interaction.data["id"], value="Quanzhou"),
            identity=identity,
        )
    )

    assert captured is not None
    assert captured.interaction_id == interaction.data["id"]
    assert captured.prompt == "Which city?"
    assert captured.value == "Quanzhou"


async def test_tool_can_explicitly_allow_repeated_equivalent_writes():
    calls = 0

    @tool(effect="write", approval="never", deduplicate=False)
    async def tick(ctx: ToolContext) -> ToolResult:
        """Record one tick."""
        nonlocal calls
        calls += 1
        return ToolResult(message="ticked")

    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(id="one", name="tick", arguments={}),
                    ToolCall(id="two", name="tick", arguments={}),
                ]
            ),
            ModelCompleted(text="Done"),
        ],
        [tick],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    await collect(
        runtime.stream_run(session_id=session.id, content="Tick twice", identity=identity)
    )

    assert calls == 2


async def test_tool_timeout_is_enforced():
    import asyncio

    @tool(effect="read", approval="never", timeout_seconds=0.001)
    async def slow(ctx: ToolContext) -> ToolResult:
        """Wait too long."""
        await asyncio.sleep(0.1)
        return ToolResult(message="late")

    runtime = runtime_with(
        [
            ModelCompleted(tool_calls=[ToolCall(id="slow", name="slow", arguments={})]),
            ModelCompleted(text="Timed out"),
        ],
        [slow],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Wait", identity=identity)
    )
    failed = next(event for event in events if event.type == "tool.failed")
    assert failed.data["code"] == "timeout"


async def test_model_failure_marks_run_failed_with_safe_event(caplog):
    runtime = runtime_with([RuntimeError("provider secret")])
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Hi", identity=identity)
    )
    assert events[-1].type == "run.failed"
    assert events[-1].data["error"] == "Agent execution failed"
    run = await runtime.store.get_run(events[-1].run_id)
    assert run.status == RunStatus.FAILED
    assert "error_type=RuntimeError" in caplog.text
    assert f"run_id={run.id}" in caplog.text


async def test_cancelling_stream_persists_terminal_event():
    import asyncio

    started = asyncio.Event()

    class SlowModel:
        async def stream(self, *, messages, tools, model=None):
            started.set()
            await asyncio.sleep(30)
            yield ModelCompleted(text="late")

    runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Answer")],
        model=SlowModel(),
        store=MemoryStore(),
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)

    task = asyncio.create_task(
        collect(runtime.stream_run(session_id=session.id, content="Plan", identity=identity))
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    runs = await runtime.list_session_runs(session.id, identity=identity)
    events = await runtime.get_run_events(runs[0].id, identity=identity)
    assert runs[0].status == RunStatus.CANCELLED
    assert events[-1].type == "run.cancelled"


async def test_model_retry_resets_partial_output_and_keeps_same_model_step():
    class RetryingModel:
        async def stream(self, *, messages, tools, model=None):
            yield TextDelta("partial")
            yield ModelRetrying(
                retry=1,
                max_retries=3,
                delay_seconds=0,
                reason="network",
            )
            yield TextDelta("recovered")
            yield ModelCompleted(text="recovered")

    runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Answer")],
        model=RetryingModel(),
        store=MemoryStore(),
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Hi", identity=identity)
    )

    assert [event.type for event in events] == [
        "run.started",
        "context.usage",
        "model.started",
        "message.delta",
        "message.reset",
        "model.retrying",
        "message.delta",
        "model.completed",
        "message.completed",
        "run.completed",
    ]
    retry = next(event for event in events if event.type == "model.retrying")
    assert retry.data == {
        "step": 1,
        "retry": 1,
        "max_retries": 3,
        "delay_seconds": 0,
        "reason": "network",
    }


async def test_max_model_steps_terminates_run():
    model = ScriptedModel(
        [
            ModelCompleted(tool_calls=[ToolCall(id="a", name="missing", arguments={})]),
            ModelCompleted(tool_calls=[ToolCall(id="b", name="missing", arguments={})]),
        ]
    )
    runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Use tools")],
        model=model,
        store=MemoryStore(),
        max_model_steps=2,
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Loop", identity=identity)
    )
    assert events[-1].type == "run.failed"
    assert events[-1].data == {
        "status": "failed",
        "code": "model_step_limit_exceeded",
        "error": "The assistant could not finish this request.",
        "retryable": True,
    }
    run = await runtime.store.get_run(events[-1].run_id)
    assert run.error == "Agent exceeded the maximum of 2 model steps"


async def test_report_progress_emits_public_activity_and_is_persisted():
    runtime = runtime_with(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="progress",
                        name="report_progress",
                        arguments={"message": "Checking the current itinerary"},
                    )
                ]
            ),
            ModelCompleted(text="Done"),
        ]
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Check it", identity=identity)
    )

    activity = next(event for event in events if event.type == "activity.updated")
    assert activity.data == {
        "id": "progress",
        "step": 1,
        "kind": "progress",
        "message": "Checking the current itinerary",
        "status": "completed",
    }
    persisted = await runtime.get_run_events(activity.run_id, identity=identity)
    assert [event.id for event in persisted] == [event.id for event in events]
    assert [event.seq for event in persisted] == list(range(1, len(events) + 1))
    assert [
        event.seq
        for event in await runtime.get_run_events(
            activity.run_id, identity=identity, after_seq=activity.seq
        )
    ] == list(range(activity.seq + 1, len(events) + 1))


async def test_long_running_tool_streams_progress_before_completion():
    @tool(effect="write", approval="never")
    async def import_places(ctx: ToolContext) -> ToolResult:
        """Import places progressively."""
        await ctx.report_progress("Added Tokyo", public={"trip_changed": True})
        await ctx.report_progress("Added Kyoto", public={"trip_changed": True})
        return ToolResult(message="Imported")

    runtime = runtime_with(
        [
            ModelCompleted(tool_calls=[ToolCall(id="import", name="import_places", arguments={})]),
            ModelCompleted(text="Done"),
        ],
        [import_places],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Import", identity=identity)
    )

    types = [event.type for event in events]
    started = types.index("tool.started")
    completed = types.index("tool.completed")
    progress = [event for event in events if event.type == "activity.updated"]
    assert started < types.index("activity.updated") < completed
    assert [event.data["message"] for event in progress] == [
        "Added Tokyo",
        "Added Kyoto",
        "Added Kyoto",
    ]
    assert {event.data["id"] for event in progress} == {"import:progress"}
    assert [event.data["status"] for event in progress] == [
        "running",
        "running",
        "completed",
    ]
    assert all(event.data["public"] == {"trip_changed": True} for event in progress)


async def test_provider_reasoning_is_not_part_of_public_events():
    runtime = runtime_with([ModelCompleted(text="Safe answer")])
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Question", identity=identity)
    )
    serialized = " ".join(event.model_dump_json() for event in events)
    assert "reasoning_content" not in serialized


async def test_run_initializer_starts_deferred_tool_without_a_model_step():
    @tool(effect="external", approval="never")
    async def ingest(_ctx: ToolContext) -> ToolResult:
        """Start ingestion."""
        return ToolResult(deferred=DeferredRequest(task_id="job-1", message="Extracting"))

    async def initialize(_ctx: ToolContext, _content: str) -> ToolCall:
        return ToolCall(id="initial", name="ingest", arguments={})

    runtime = Runtime(
        agents=[
            Agent(
                name="assistant",
                instructions="unused",
                tools=[ingest],
                run_initializer=initialize,
            )
        ],
        model=ScriptedModel([]),
        store=MemoryStore(),
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    run = await runtime.create_run(
        session_id=session.id,
        content="pasted itinerary",
        identity=identity,
    )
    events = []
    async for event in runtime.stream_events(run.id, identity=identity):
        events.append(event)
        if event.type == "run.waiting_external":
            break

    persisted = await runtime.store.get_run(run.id)
    assert persisted.status is RunStatus.WAITING_EXTERNAL
    assert persisted.model_steps == 0
    assert [event.type for event in events][-2:] == [
        "external_task.started",
        "run.waiting_external",
    ]


async def test_deferred_completion_can_request_a_runtime_owned_interaction():
    @tool(effect="external", approval="never")
    async def begin(_ctx: ToolContext) -> ToolResult:
        """Begin work."""
        return ToolResult(deferred=DeferredRequest(task_id="job-1", message="Working"))

    @tool(
        effect="write",
        approval="never",
        requires_interaction_response=True,
    )
    async def continue_work(ctx: ToolContext) -> ToolResult:
        """Continue after a trusted answer."""
        assert ctx.interaction_response is not None
        return ToolResult(message="Done")

    runtime = runtime_with(
        [ModelCompleted(tool_calls=[ToolCall(id="begin", name="begin", arguments={})])],
        [begin, continue_work],
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    first = await collect(
        runtime.stream_run(session_id=session.id, content="Import", identity=identity)
    )
    run_id = first[0].run_id
    resumed = await runtime.resume_deferred(
        run_id,
        "job-1",
        ToolResult(
            message="Needs confirmation",
            interaction=InteractionRequest(
                kind="choice",
                prompt="Which place?",
                options=[InteractionOption(id="a", label="A")],
                continuation=ToolContinuation(tool="continue_work"),
            ),
        ),
        identity=identity,
    )

    run = await runtime.store.get_run(run_id)
    assert run.status is RunStatus.WAITING_INPUT
    assert run.model_steps == 1
    assert [event.type for event in resumed][-2:] == [
        "run.waiting_input",
        "interaction.requested",
    ]


async def test_get_context_usage_estimates_assembled_prompt():
    async def current_user(ctx: ToolContext) -> str:
        return f"user={ctx.identity.subject_id}"

    runtime = Runtime(
        agents=[
            Agent(
                name="assistant",
                instructions="You are a console assistant.",
                context=current_user,
            )
        ],
        model=ScriptedModel([]),
        store=MemoryStore(),
        context_budget_tokens=12_000,
    )
    identity = Identity(subject_id="alice")
    session = await runtime.create_session(identity=identity)
    usage = await runtime.get_context_usage(session.id, identity=identity)
    assert usage["budget_tokens"] == 12_000
    assert usage["used_tokens"] > 0
    assert usage["source"] == "estimated"
    assert {item["id"] for item in usage["categories"]} >= {
        "instructions",
        "runtime",
        "context",
        "tool_definitions",
    }


async def test_drive_emits_estimated_then_provider_context_usage():
    runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Reply")],
        model=ScriptedModel(
            [ModelCompleted(text="Hello", usage={"prompt_tokens": 80, "completion_tokens": 4})]
        ),
        store=MemoryStore(),
        context_budget_tokens=200,
    )
    identity = Identity(subject_id="u1")
    session = await runtime.create_session(identity=identity)
    events = await collect(
        runtime.stream_run(session_id=session.id, content="Hi", identity=identity)
    )
    usages = [event for event in events if event.type == "context.usage"]
    assert [event.data["source"] for event in usages] == ["estimated", "provider"]
    assert usages[1].data["used_tokens"] == 80
    assert usages[1].data["prompt_tokens"] == 80
    assert usages[1].data["budget_tokens"] == 200
