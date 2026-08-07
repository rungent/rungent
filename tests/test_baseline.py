from rungent import Agent, Identity, Runtime, ToolContext, ToolResult, tool
from rungent.llm import ModelCompleted
from rungent.state import ToolCall
from rungent.store import MemoryStore
from rungent.testing import (
    BaselineCase,
    BaselineContext,
    BaselineSuite,
    ScriptedModel,
    run_baseline,
)


async def test_baseline_runner_covers_tools_interactions_and_state():
    moved: list[int] = []

    @tool(effect="write", approval="never")
    async def move(ctx: ToolContext, day: int) -> ToolResult:
        """Move to a day."""
        moved.append(day)
        return ToolResult(message="moved")

    model = ScriptedModel(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="ask",
                        name="request_input",
                        arguments={
                            "kind": "choice",
                            "prompt": "Which day?",
                            "options": [{"id": "2", "label": "Day 2"}],
                        },
                    )
                ]
            ),
            ModelCompleted(tool_calls=[ToolCall(id="move", name="move", arguments={"day": 2})]),
            ModelCompleted(text="Done"),
        ]
    )
    runtime = Runtime(
        agents=[Agent(name="trip", instructions="Plan", tools=[move])],
        model=model,
        store=MemoryStore(),
    )

    def assert_moved(_context: BaselineContext) -> None:
        assert moved == [2]

    report = await run_baseline(
        runtime,
        [
            BaselineCase(
                name="move after choosing day",
                input="Move the temple",
                expected_tools=("move",),
                expected_interactions=("choice",),
                responses=({"selected": ["2"]},),
                assert_state=assert_moved,
            )
        ],
        identity=Identity(subject_id="baseline"),
    )
    report.assert_passed()
    assert report.pass_rate == 1.0


async def test_baseline_runner_keeps_followups_in_the_same_session():
    model = ScriptedModel([ModelCompleted(text="First"), ModelCompleted(text="Second")])
    runtime = Runtime(
        agents=[Agent(name="trip", instructions="Plan")],
        model=model,
        store=MemoryStore(),
    )
    report = await run_baseline(
        runtime,
        [BaselineCase(name="follow-up", input="Plan Tokyo", followups=("Add one day",))],
        identity=Identity(subject_id="baseline"),
    )
    report.assert_passed()
    second_request = model.requests[1]["messages"]
    assert any(item.get("content") == "Plan Tokyo" for item in second_request)
    assert any(item.get("content") == "First" for item in second_request)


async def test_baseline_runner_covers_grouped_form_interactions():
    model = ScriptedModel(
        [
            ModelCompleted(
                tool_calls=[
                    ToolCall(
                        id="preferences",
                        name="request_input",
                        arguments={
                            "kind": "form",
                            "prompt": "Trip preferences",
                            "questions": [
                                {"id": "city", "kind": "text", "prompt": "Which city?"},
                                {
                                    "id": "pace",
                                    "kind": "choice",
                                    "prompt": "What pace?",
                                    "options": [{"id": "slow", "label": "Slow"}],
                                },
                            ],
                        },
                    )
                ]
            ),
            ModelCompleted(text="Planned"),
        ]
    )
    runtime = Runtime(
        agents=[Agent(name="trip", instructions="Plan")], model=model, store=MemoryStore()
    )
    report = await run_baseline(
        runtime,
        [
            BaselineCase(
                name="group preferences",
                input="Plan a trip",
                expected_interactions=("form",),
                responses=(
                    {
                        "answers": {
                            "city": "Tokyo",
                            "pace": {"selected": ["slow"]},
                        }
                    },
                ),
            )
        ],
        identity=Identity(subject_id="baseline"),
    )
    report.assert_passed()


async def test_baseline_suite_is_an_embeddable_application_tool():
    lifecycle: list[str] = []
    runtime = Runtime(
        agents=[Agent(name="trip", instructions="Plan")],
        model=ScriptedModel([ModelCompleted(text="Done")]),
        store=MemoryStore(),
    )
    suite = BaselineSuite(
        runtime=runtime,
        cases=[BaselineCase(name="plain", input="Plan Tokyo")],
        before_all=lambda: lifecycle.append("before"),
        after_all=lambda: lifecycle.append("after"),
    )
    report = await suite.run()
    assert report.to_dict()["passed"] is True
    assert lifecycle == ["before", "after"]


async def test_baseline_contains_tool_match_allows_extra_calls_in_order():
    calls: list[int] = []

    @tool(effect="write", approval="never")
    async def move(ctx: ToolContext, day: int) -> ToolResult:
        """Move to a day."""
        calls.append(day)
        return ToolResult(message="moved")

    runtime = Runtime(
        agents=[Agent(name="trip", instructions="Plan", tools=[move])],
        model=ScriptedModel(
            [
                ModelCompleted(
                    tool_calls=[ToolCall(id="move-1", name="move", arguments={"day": 2})]
                ),
                ModelCompleted(
                    tool_calls=[ToolCall(id="move-2", name="move", arguments={"day": 3})]
                ),
                ModelCompleted(text="Done"),
            ]
        ),
        store=MemoryStore(),
    )
    report = await run_baseline(
        runtime,
        [
            BaselineCase(
                name="idempotent retry",
                input="Move the temple",
                expected_tools=("move",),
                tool_match="contains",
            )
        ],
        identity=Identity(subject_id="baseline"),
    )
    report.assert_passed()
    assert calls == [2, 3]


async def test_baseline_contains_interaction_match_allows_repeated_questions():
    choice = ModelCompleted(
        tool_calls=[
            ToolCall(
                id="ask",
                name="request_input",
                arguments={
                    "kind": "choice",
                    "prompt": "Which day?",
                    "options": [{"id": "2", "label": "Day 2"}],
                },
            )
        ]
    )
    runtime = Runtime(
        agents=[Agent(name="trip", instructions="Plan")],
        model=ScriptedModel([choice, choice, ModelCompleted(text="Done")]),
        store=MemoryStore(),
    )
    report = await run_baseline(
        runtime,
        [
            BaselineCase(
                name="repeated clarification",
                input="Move the temple",
                expected_interactions=("choice",),
                interaction_match="contains",
                responses=({"selected": ["2"]}, {"selected": ["2"]}),
            )
        ],
        identity=Identity(subject_id="baseline"),
    )
    report.assert_passed()
