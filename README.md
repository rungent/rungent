# Rungent

Rungent is a small, typed agent runtime that embeds into an existing Python application. It
standardizes tool calling, the harness loop, durable sessions and runs, user interactions,
deterministic approval gates, and an SSE event protocol. The same tools can be mounted as an
optional MCP adapter for external agents; OAuth stays in the host application.

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
    model=OpenAICompatibleModel(
        api_key="...",
        base_url="https://api.openai.com/v1",
        model="gpt-5-mini",
    ),
    store=MemoryStore(),
)
```

Documentation: [rungent.github.io](https://rungent.github.io). Source lives in `docs/` and is
rendered by the Fumadocs app in `apps/docs`.

AI coding agents should start with [`RUNGENT.md`](RUNGENT.md), then load
[llms.txt](https://rungent.github.io/llms.txt) or the full dump
[llms-full.txt](https://rungent.github.io/llms-full.txt). Per-page raw Markdown is at
`https://rungent.github.io/markdown/<page>.md`.

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

The public site is [rungent.github.io](https://rungent.github.io). After docs changes land on
`main`, run **Publish** on [`rungent/rungent.github.io`](https://github.com/rungent/rungent.github.io/actions)
(or set repo secret `GH_PAGES_TOKEN` so CI can trigger that workflow). The org disables deploy keys,
so the pages repo rebuilds from this repository with its own `GITHUB_TOKEN`.

## Release

`rungent` (PyPI) and `@rungent/sdk` (npm) share one semver. Publishing is done by GitHub Actions
on `v*` tags — do not `uv publish` / `npm publish` from your laptop.

```bash
./scripts/release.sh 0.2.0
git push origin main --tags
```

One-time Trusted Publishing setup: see [`RELEASING.md`](RELEASING.md).
