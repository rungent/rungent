"""Reference shape for Roasea; adapt imports to the application modules."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from rungent import Agent, ToolContext, ToolResult, tool


class ImportStop(BaseModel):
    query: str
    day_number: int | None = None
    start_time: str | None = None
    note: str = ""


class ImportCandidate(BaseModel):
    title: str | None = None
    stops: list[ImportStop] = Field(default_factory=list)


@tool(effect="external", approval="never", title="Add place")
async def add_place(
    ctx: ToolContext,
    query: Annotated[str, "Place name or search query"],
) -> ToolResult:
    """Find and add one place to the trip's unplanned pool."""
    trips: Any = ctx.deps["trips"]
    stop = await trips.add_place(ctx.resource["trip_id"], query)
    return ToolResult(data={"stop_id": stop.id}, message=f"Added {stop.place_name}")


@tool(effect="write", approval="never", title="Move place")
async def move_place(
    ctx: ToolContext,
    name: Annotated[str, "Exact place name from the current trip"],
    day_number: Annotated[int | None, "One-based day number; null means unplanned"],
) -> ToolResult:
    """Move a place to another day or back to the unplanned pool."""
    trips: Any = ctx.deps["trips"]
    stop = await trips.move_place(ctx.resource["trip_id"], name, day_number)
    return ToolResult(data={"stop_id": stop.id}, message=f"Moved {stop.place_name}")


@tool(
    effect="destructive",
    approval="always",
    title="Remove place",
    confirmation="Remove {name} from this trip?",
)
async def remove_place(
    ctx: ToolContext,
    name: Annotated[str, "Exact place name from the current trip"],
) -> ToolResult:
    """Remove one place from the trip."""
    trips: Any = ctx.deps["trips"]
    await trips.remove_place(ctx.resource["trip_id"], name)
    return ToolResult(message=f"Removed {name}")


@tool(effect="write", approval="never", title="Swap places")
async def swap_places(
    ctx: ToolContext,
    first: Annotated[str, "First exact place name"],
    second: Annotated[str, "Second exact place name"],
) -> ToolResult:
    """Swap the itinerary positions of two places."""
    trips: Any = ctx.deps["trips"]
    await trips.swap_places(ctx.resource["trip_id"], first, second)
    return ToolResult(message=f"Swapped {first} and {second}")


@tool(effect="write", approval="never", title="Add trip day")
async def add_day(
    ctx: ToolContext,
    after_day: Annotated[int | None, "Insert after this one-based day; null appends"],
) -> ToolResult:
    """Add one day to the trip itinerary."""
    trips: Any = ctx.deps["trips"]
    day = await trips.add_day(ctx.resource["trip_id"], after_day)
    return ToolResult(data={"day_number": day.number}, message=f"Added day {day.number}")


@tool(
    effect="destructive",
    approval="always",
    title="Delete trip day",
    confirmation="Delete day {day_number} and move its places to the unplanned pool?",
)
async def remove_day(ctx: ToolContext, day_number: int) -> ToolResult:
    """Delete a trip day and move its places to the unplanned pool."""
    trips: Any = ctx.deps["trips"]
    await trips.remove_day(ctx.resource["trip_id"], day_number)
    return ToolResult(message=f"Deleted day {day_number}")


@tool(effect="write", approval="never", title="Set transport")
async def set_transport(
    ctx: ToolContext,
    from_place: Annotated[str, "Exact origin place name"],
    to_place: Annotated[str, "Exact destination place name"],
    mode: Literal["walking", "driving", "transit", "bicycling"],
) -> ToolResult:
    """Set the transport mode between two consecutive places."""
    trips: Any = ctx.deps["trips"]
    await trips.set_transport(ctx.resource["trip_id"], from_place, to_place, mode)
    return ToolResult(message=f"Set {mode} from {from_place} to {to_place}")


@tool(effect="external", approval="never", title="Prepare itinerary import")
async def prepare_itinerary_import(
    ctx: ToolContext,
    candidate: Annotated[ImportCandidate, "Denoised final itinerary candidate"],
) -> ToolResult:
    """Resolve places and persist a staged import without changing the trip."""
    draft = await ctx.deps["trips"].prepare_import(
        ctx.resource["trip_id"], ctx.run_id, candidate.model_dump(mode="json")
    )
    return ToolResult(data=draft.for_model(), message="Import preview ready", public=draft.preview)


@tool(effect="external", approval="never", title="Resolve itinerary import")
async def resolve_itinerary_import(
    ctx: ToolContext,
    draft_id: str,
    resolutions: dict[str, str],
) -> ToolResult:
    """Apply clarification answers and recompute the staged import preview."""
    draft = await ctx.deps["trips"].resolve_import(draft_id, resolutions)
    return ToolResult(
        data=draft.for_model(), message="Import preview updated", public=draft.preview
    )


async def import_confirmation(ctx: ToolContext, draft_id: str, expected_revision: int) -> str:
    preview = await ctx.deps["trips"].import_summary(draft_id, expected_revision)
    return f"Merge {preview.change_count} reviewed changes into {preview.trip_title}?"


@tool(
    effect="destructive",
    approval="always",
    title="Commit itinerary import",
    confirmation=import_confirmation,
)
async def commit_itinerary_import(
    ctx: ToolContext,
    draft_id: str,
    expected_revision: int,
) -> ToolResult:
    """Atomically commit one reviewed import if the trip revision is unchanged."""
    result = await ctx.deps["trips"].commit_import(draft_id, expected_revision)
    return ToolResult(
        data={"revision": result.revision},
        message="Itinerary import committed",
        public={"trip_changed": True},
    )


async def trip_context(ctx: ToolContext) -> str:
    trips: Any = ctx.deps["trips"]
    return await trips.compact_snapshot(ctx.resource["trip_id"])


trip_agent = Agent(
    name="trip_planner",
    instructions="""
    You help users edit a trip itinerary. Use tools for every state change. Use request_input when
    a place or day is ambiguous. Never claim success until a tool confirms it.
    """,
    tools=[
        add_place,
        move_place,
        remove_place,
        swap_places,
        add_day,
        remove_day,
        set_transport,
        prepare_itinerary_import,
        resolve_itinerary_import,
        commit_itinerary_import,
    ],
    context=trip_context,
)
