from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

from .acs import Event
from .llm import Model, ModelCompleted, ModelEvent, TextDelta
from .runtime import Runtime
from .state import Identity, InteractionResponse, RunStatus


class ScriptedModel(Model):
    """Deterministic model for harness tests and application baselines."""

    def __init__(self, responses: Sequence[ModelCompleted | Exception]) -> None:
        self.responses = deque(responses)
        self.requests: list[dict[str, Any]] = []

    async def stream(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        model: str | None = None,
    ) -> AsyncIterator[ModelEvent]:
        self.requests.append({"messages": list(messages), "tools": list(tools), "model": model})
        if not self.responses:
            raise AssertionError("ScriptedModel has no response left")
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        if response.text:
            yield TextDelta(response.text)
        yield response


@dataclass(frozen=True, slots=True)
class BaselineEnvironment:
    resource: Mapping[str, Any] = field(default_factory=dict)
    deps: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BaselineCase:
    name: str
    input: str
    followups: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] = ()
    tool_match: Literal["exact", "contains"] = "exact"
    expected_interactions: tuple[str, ...] = ()
    interaction_match: Literal["exact", "contains"] = "exact"
    responses: tuple[Any, ...] = ()
    expected_status: RunStatus = RunStatus.COMPLETED
    assert_state: "BaselineAssertion | None" = None


@dataclass(frozen=True, slots=True)
class BaselineContext:
    case: BaselineCase
    session_id: str
    resource: Mapping[str, Any]
    deps: Mapping[str, Any]
    events: Sequence[Event]


BaselineAssertion = Callable[[BaselineContext], None | Awaitable[None]]
BaselineSetup = Callable[[BaselineCase], BaselineEnvironment | Awaitable[BaselineEnvironment]]
BaselineHook = Callable[[], None | Awaitable[None]]


def _contains_in_order(actual: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    remaining = iter(actual)
    return all(any(item == wanted for item in remaining) for wanted in expected)


@dataclass(slots=True)
class BaselineResult:
    name: str
    passed: bool
    tools: list[str] = field(default_factory=list)
    interactions: list[str] = field(default_factory=list)
    final_text: str = ""
    error: str | None = None


@dataclass(slots=True)
class BaselineReport:
    results: list[BaselineResult]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.results)

    @property
    def pass_rate(self) -> float:
        return (
            sum(item.passed for item in self.results) / len(self.results) if self.results else 1.0
        )

    def assert_passed(self) -> None:
        failures = [f"{item.name}: {item.error}" for item in self.results if not item.passed]
        if failures:
            raise AssertionError("Baseline failures:\n" + "\n".join(failures))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "results": [asdict(item) for item in self.results],
        }


@dataclass(frozen=True, slots=True)
class BaselineSuite:
    runtime: Runtime
    cases: Sequence[BaselineCase]
    identity: Identity = field(default_factory=lambda: Identity(subject_id="baseline"))
    resource: Mapping[str, Any] = field(default_factory=dict)
    deps: Mapping[str, Any] = field(default_factory=dict)
    setup_case: BaselineSetup | None = None
    before_all: BaselineHook | None = None
    after_all: BaselineHook | None = None

    async def run(self) -> BaselineReport:
        if self.before_all:
            value = self.before_all()
            if __import__("inspect").isawaitable(value):
                await cast(Awaitable[None], value)
        try:
            return await run_baseline(
                self.runtime,
                self.cases,
                identity=self.identity,
                resource=self.resource,
                deps=self.deps,
                setup_case=self.setup_case,
            )
        finally:
            if self.after_all:
                value = self.after_all()
                if __import__("inspect").isawaitable(value):
                    await cast(Awaitable[None], value)


async def _collect(stream: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in stream]


async def run_baseline(
    runtime: Runtime,
    cases: Sequence[BaselineCase],
    *,
    identity: Identity,
    resource: Mapping[str, Any] | None = None,
    deps: Mapping[str, Any] | None = None,
    setup_case: BaselineSetup | None = None,
) -> BaselineReport:
    results: list[BaselineResult] = []
    for case in cases:
        result = BaselineResult(name=case.name, passed=False)
        try:
            environment = BaselineEnvironment(resource=resource or {}, deps=deps or {})
            if setup_case is not None:
                configured = setup_case(case)
                if __import__("inspect").isawaitable(configured):
                    environment = await cast(Awaitable[BaselineEnvironment], configured)
                else:
                    environment = cast(BaselineEnvironment, configured)
            session = await runtime.create_session(identity=identity, resource=environment.resource)
            events: list[Event] = []
            response_index = 0
            for user_input in (case.input, *case.followups):
                turn_events = await _collect(
                    runtime.stream_run(
                        session_id=session.id,
                        content=user_input,
                        identity=identity,
                        deps=environment.deps,
                    )
                )
                events.extend(turn_events)
                while turn_events and turn_events[-1].type not in {
                    "run.completed",
                    "run.failed",
                }:
                    interaction_event = next(
                        (
                            item
                            for item in reversed(turn_events)
                            if item.type == "interaction.requested"
                        ),
                        None,
                    )
                    if interaction_event is None or response_index >= len(case.responses):
                        break
                    turn_events = await _collect(
                        runtime.stream_response(
                            run_id=interaction_event.run_id,
                            response=InteractionResponse(
                                interaction_id=interaction_event.data["id"],
                                value=case.responses[response_index],
                            ),
                            identity=identity,
                            deps=environment.deps,
                        )
                    )
                    events.extend(turn_events)
                    response_index += 1
                if not turn_events or turn_events[-1].type != "run.completed":
                    break

            result.tools = [
                str(item.data["name"]) for item in events if item.type == "tool.started"
            ]
            result.interactions = [
                str(item.data["kind"]) for item in events if item.type == "interaction.requested"
            ]
            completed = [item for item in events if item.type == "message.completed"]
            result.final_text = str(completed[-1].data.get("content", "")) if completed else ""
            if not events:
                raise AssertionError("Baseline case produced no events")
            run = await runtime.store.get_run(events[-1].run_id)

            actual_tools = tuple(result.tools)
            actual_interactions = tuple(result.interactions)
            tools_match = (
                actual_tools == case.expected_tools
                if case.tool_match == "exact"
                else _contains_in_order(actual_tools, case.expected_tools)
            )
            if not tools_match:
                raise AssertionError(
                    f"expected tools {case.tool_match} {case.expected_tools}, got {actual_tools}"
                )
            interactions_match = (
                actual_interactions == case.expected_interactions
                if case.interaction_match == "exact"
                else _contains_in_order(actual_interactions, case.expected_interactions)
            )
            if not interactions_match:
                raise AssertionError(
                    "expected interactions "
                    f"{case.interaction_match} {case.expected_interactions}, "
                    f"got {actual_interactions}"
                )
            if run.status != case.expected_status:
                raise AssertionError(f"expected status {case.expected_status}, got {run.status}")
            if case.assert_state:
                value = case.assert_state(
                    BaselineContext(
                        case=case,
                        session_id=session.id,
                        resource=environment.resource,
                        deps=environment.deps,
                        events=events,
                    )
                )
                if __import__("inspect").isawaitable(value):
                    await cast(Awaitable[None], value)
            result.passed = True
        except Exception as exc:
            result.error = str(exc)
        results.append(result)
    return BaselineReport(results)
