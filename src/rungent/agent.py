from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from .state import ToolCall
from .tools import Tool, ToolContext

ContextProvider = Callable[[ToolContext], str | Awaitable[str]]


@dataclass(frozen=True, slots=True)
class RunActivity:
    """Public, host-authored status shown while the model has not produced an event."""

    message: str
    waiting_message: str
    long_wait_message: str
    continuation_message: str
    public: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value in (
            self.message,
            self.waiting_message,
            self.long_wait_message,
            self.continuation_message,
        ):
            if not value.strip() or len(value) > 280:
                raise ValueError("Run activity messages must contain 1 to 280 characters")


RunActivityProvider = Callable[[ToolContext, str], RunActivity | Awaitable[RunActivity]]
RunInitializer = Callable[[ToolContext, str], ToolCall | None | Awaitable[ToolCall | None]]


@dataclass(frozen=True, slots=True)
class Agent:
    name: str
    instructions: str
    tools: Sequence[Tool] = field(default_factory=tuple)
    context: ContextProvider | None = None
    run_activity: RunActivityProvider | None = None
    run_initializer: RunInitializer | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("Agent name must contain only letters, digits, and underscores")
        names = [item.name for item in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"Agent {self.name} has duplicate tool names")
        reserved = {"request_input", "report_progress"}.intersection(names)
        if reserved:
            raise ValueError(f"{sorted(reserved)[0]} is reserved by Rungent")

    @classmethod
    def from_markdown(
        cls,
        *,
        name: str,
        instructions: str | Path,
        tools: Sequence[Tool] = (),
        context: ContextProvider | None = None,
        run_activity: RunActivityProvider | None = None,
        run_initializer: RunInitializer | None = None,
        model: str | None = None,
    ) -> Self:
        path = Path(instructions)
        return cls(
            name=name,
            instructions=path.read_text(encoding="utf-8"),
            tools=tools,
            context=context,
            run_activity=run_activity,
            run_initializer=run_initializer,
            model=model,
        )

    def tool_map(self) -> dict[str, Tool]:
        return {item.name: item for item in self.tools}

    def tool_schemas(self, *, interaction_response_available: bool = False) -> list[dict[str, Any]]:
        from .tools import REPORT_PROGRESS_SCHEMA, REQUEST_INPUT_SCHEMA

        return [
            REPORT_PROGRESS_SCHEMA,
            REQUEST_INPUT_SCHEMA,
            *(
                item.schema()
                for item in self.tools
                if interaction_response_available or not item.requires_interaction_response
            ),
        ]
