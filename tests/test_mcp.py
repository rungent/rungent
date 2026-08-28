from typing import Annotated, Any

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from rungent import (
    Agent,
    Identity,
    InteractionRequest,
    ToolContext,
    ToolContinuation,
    ToolResult,
    tool,
)
from rungent.mcp import create_mcp_asgi, invoke_tool, mcp_tool_descriptor
from rungent.state import InteractionQuestion


@tool(effect="read", approval="never", title="Ping")
async def ping(ctx: ToolContext) -> ToolResult:
    """Return the subject."""
    return ToolResult(data={"subject": ctx.identity.subject_id})


@tool(effect="write", approval="always", confirmation="Confirm {name}", title="Ask")
async def ask_more(
    ctx: ToolContext,
    name: Annotated[str | None, "Name"] = None,
) -> ToolResult:
    """Ask for a name when missing."""
    if not name:
        return ToolResult(
            interaction=InteractionRequest(
                kind="form",
                prompt="Name is required",
                questions=[InteractionQuestion(id="name", prompt="Name", kind="text")],
                continuation=ToolContinuation(tool="ask_more", arguments={}),
            )
        )
    return ToolResult(data={"name": name})


@tool(effect="write", approval="never", requires_interaction_response=True, title="Finish")
async def finish_ask(ctx: ToolContext) -> ToolResult:
    """Hidden continuation."""
    return ToolResult(data={"ok": True})


def _agent() -> Agent:
    return Agent(name="demo", instructions="demo", tools=[ping, ask_more, finish_ask])


@pytest.mark.asyncio
async def test_invoke_and_skill_omit_continuation() -> None:
    identity = Identity(subject_id="u1")
    result = await invoke_tool(ping, {}, identity=identity)
    assert result["isError"] is False
    assert "u1" in result["content"][0]["text"]

    missing = await invoke_tool(ask_more, {}, identity=identity)
    assert missing["isError"] is True
    assert "missing_arguments" in missing["content"][0]["text"]

    skill = _agent().export_skill(title="Demo")
    assert "`ping`" in skill
    assert "`ask_more`" in skill
    assert "`finish_ask`" not in skill
    assert "request_input" not in skill


@pytest.mark.asyncio
async def test_mcp_http_lists_and_calls_tools() -> None:
    app = FastAPI()

    async def resolve_identity(request: Request) -> Identity:
        auth = request.headers.get("Authorization") or ""
        if auth != "Bearer good":
            raise PermissionError("login required")
        return Identity(subject_id="u1")

    async def context(_request: Request) -> dict[str, Any]:
        return {"source": "mcp"}

    app.include_router(
        create_mcp_asgi(
            _agent(),
            identity_resolver=resolve_identity,
            context_factory=context,
            resource_metadata_url="https://example.test/.well-known/oauth-protected-resource",
        ),
        prefix="/mcp",
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert denied.status_code == 401
        assert "resource_metadata" in denied.headers["www-authenticate"]

        headers = {"Authorization": "Bearer good"}
        listed = await client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        names = {item["name"] for item in listed.json()["result"]["tools"]}
        assert names == {"ping", "ask_more"}
        tools = listed.json()["result"]["tools"]
        ping_desc = next(item for item in tools if item["name"] == "ping")
        assert ping_desc["annotations"]["readOnlyHint"] is True

        called = await client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "ping", "arguments": {}},
            },
        )
        assert called.json()["result"]["isError"] is False
        assert "u1" in called.json()["result"]["content"][0]["text"]


def test_mcp_descriptor_maps_effects() -> None:
    descriptor = mcp_tool_descriptor(ask_more)
    assert descriptor["annotations"]["readOnlyHint"] is False
    assert descriptor["_meta"]["approval"] == "always"
