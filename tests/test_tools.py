import json
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ValidationError

from rungent import Identity, ToolContext, ToolResult, tool


@tool(effect="write", approval="never")
async def move_place(
    ctx: ToolContext,
    name: Annotated[str, "Exact place name"],
    day: int | None,
    mode: Literal["walking", "transit"] = "walking",
) -> ToolResult:
    """Move a place to a day."""
    return ToolResult(data={"name": name, "day": day, "mode": mode})


def test_tool_schema_is_inferred_from_signature():
    schema = move_place.schema()["function"]
    assert schema["name"] == "move_place"
    assert schema["parameters"]["additionalProperties"] is False
    assert schema["parameters"]["properties"]["mode"]["enum"] == ["walking", "transit"]
    assert schema["parameters"]["properties"]["name"]["description"] == "Exact place name"
    assert set(schema["parameters"]["required"]) == {"name", "day"}


def test_tool_rejects_extra_and_invalid_arguments():
    with pytest.raises(ValidationError):
        move_place.validate({"name": "Temple", "day": 1, "extra": True})
    with pytest.raises(ValidationError):
        move_place.validate({"name": "Temple", "day": 1, "mode": "flying"})


async def test_tool_returns_structured_result():
    ctx = ToolContext(identity=Identity(subject_id="user"), session_id="s", run_id="r")
    result = await move_place(ctx, name="Temple", day=None)
    assert result.data == {"name": "Temple", "day": None, "mode": "walking"}
    assert json.loads(result.for_model())["tool_status"] == "success"


def test_ok_false_result_uses_error_envelope():
    result = ToolResult(data={"ok": False, "error": "try again"})
    assert json.loads(result.for_model())["tool_status"] == "error"


def test_tool_requires_async_function_and_typed_context():
    with pytest.raises(TypeError, match="async function"):

        @tool(effect="read", approval="never")  # pyright: ignore[reportArgumentType]
        def sync_tool(ctx: ToolContext) -> None:
            """Invalid synchronous tool."""

    with pytest.raises(TypeError, match="first parameter"):

        @tool(effect="read", approval="never")
        async def wrong_context(value: str) -> None:
            """Invalid context parameter."""


def test_approval_tool_requires_concrete_confirmation():
    with pytest.raises(TypeError, match="needs a confirmation"):

        @tool(effect="destructive", approval="always")
        async def erase(ctx: ToolContext) -> None:
            """Erase data."""


def test_tool_can_require_a_trusted_interaction_response():
    @tool(
        effect="write",
        approval="never",
        requires_interaction_response=True,
    )
    async def resolve(ctx: ToolContext) -> None:
        """Resolve one pending question."""

    assert resolve.requires_interaction_response is True


async def test_tool_preserves_nested_pydantic_values_for_execution():
    class Filters(BaseModel):
        city: str

    @tool(effect="read", approval="never")
    async def search(ctx: ToolContext, filters: Filters) -> str:
        """Search with structured filters."""
        assert isinstance(filters, Filters)
        return filters.city

    result = await search(
        ToolContext(identity=Identity(subject_id="user"), session_id="s", run_id="r"),
        filters={"city": "Tokyo"},
    )
    assert result.data == "Tokyo"
