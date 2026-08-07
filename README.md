# Rungent

Rungent is a small, typed agent runtime that embeds into an existing Python application. It
standardizes tool calling, the harness loop, durable sessions and runs, user interactions,
deterministic approval gates, and an SSE event protocol.

Its headless activity protocol exposes model steps, safe public progress, tools, and interactions as
persisted events. It never exposes raw provider reasoning or hidden chain of thought.

```python
from typing import Annotated

from rungent import Agent, RunActivity, Runtime, ToolContext, ToolResult, tool
from rungent.llm import OpenAICompatibleModel
from rungent.store import MemoryStore


@tool(effect="write", approval="never")
async def move_place(
    ctx: ToolContext,
    name: Annotated[str, "Exact place name"],
    day_number: Annotated[int | None, "Target day; null means unplanned"],
) -> ToolResult:
    await ctx.deps["trips"].move(ctx.resource["trip_id"], name, day_number)
    return ToolResult(message=f"Moved {name}")


agent = Agent(
    name="trip_planner",
    instructions="Help the user update their trip. Use tools for every change.",
    tools=[move_place],
    run_activity=lambda _ctx, _content: RunActivity(
        message="Received the request",
        waiting_message="Still preparing; no changes have been made",
        long_wait_message="The request is detailed; still preparing",
        continuation_message="Preparing the result",
    ),
)

runtime = Runtime(
    agents=[agent],
    model=OpenAICompatibleModel.from_env(),
    store=MemoryStore(),
)
```

The canonical documentation lives in `docs/` and is rendered by the Fumadocs app in `apps/docs`.
AI coding agents should start with [`RUNGENT.md`](RUNGENT.md); the running site also exposes
`/llms.txt`, `/llms-full.txt`, and per-page raw Markdown under `/markdown/*`.

## Development

```bash
uv sync --all-extras
uv run pytest -W error
uv run ruff check src tests

pnpm install
pnpm dev:docs
pnpm test
pnpm build
```

## Release

`rungent` (PyPI) and `@rungent/sdk` (npm) share one semver. Publishing is done by GitHub Actions
on `v*` tags — do not `uv publish` / `npm publish` from your laptop.

```bash
./scripts/release.sh 0.2.0
git push origin main --tags
```

One-time Trusted Publishing setup: see [`RELEASING.md`](RELEASING.md).
