"""Tests for System One shortlist and HTTP allowlist tools."""

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field
from rungent import ToolContext, create_openapi_agent
from rungent.http_tools import HttpOperation, HttpToolSettings, tools_from_operations
from rungent.state import Identity
from rungent.systemone import SystemOneClient, SystemOneSettings, shortlist_tools
from rungent.tools import ApprovalPolicy, Tool, ToolEffect


class _Empty(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str | None = Field(default=None)


async def _noop(ctx: ToolContext, **kwargs: Any) -> dict[str, Any]:
    return {"ok": True}


def _tool(name: str, domain: str) -> Tool:
    return Tool(
        name=name,
        description=f"{name} does {domain}",
        function=_noop,
        input_model=_Empty,
        effect=ToolEffect.READ,
        approval=ApprovalPolicy.NEVER,
        confirmation=None,
        approval_ready=None,
        title=name,
        timeout_seconds=30,
        parallel=True,
        deduplicate=True,
        requires_interaction_response=False,
        domain=domain,
    )


@pytest.mark.asyncio
async def test_shortlist_picks_domain_tools():
    settings = SystemOneSettings(base_url="http://systemone.test", confidence_threshold=0.5)
    client = SystemOneClient(settings)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        questions = payload["questions"]
        if "needs_tool" in questions:
            return httpx.Response(
                200,
                json={
                    "answers": {
                        "needs_tool": {"type": "noul", "noul": 0.9},
                        "domain": {
                            "type": "choice",
                            "choice": "vm",
                            "confidence": 0.9,
                            "probabilities": {"vm": 0.9},
                        },
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "answers": {
                    "tool": {
                        "type": "choice",
                        "choice": "list_vms",
                        "confidence": 0.95,
                        "probabilities": {"list_vms": 0.95},
                    }
                }
            },
        )

    transport = httpx.MockTransport(handler)
    client._client = httpx.AsyncClient(base_url=settings.base_url, transport=transport)
    client._owns_client = True

    names = await shortlist_tools(
        client,
        tools=[_tool("list_vms", "vm"), _tool("list_elastic", "elastic")],
        user_input="列出云服务器",
        recent_tool_names=[],
    )
    assert names is not None
    assert "list_vms" in names
    assert "list_elastic" not in names
    await client.aclose()


@pytest.mark.asyncio
async def test_http_get_tool_executes():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/vm"
        return httpx.Response(200, json={"items": [{"id": "1", "name": "a"}]})

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    settings = HttpToolSettings(base_url="http://gw.test", client=async_client)
    tools = tools_from_operations(
        [
            HttpOperation(
                name="list_vms",
                method="GET",
                path="/api/vm",
                summary="List VMs",
                domain="vm",
            )
        ],
        settings,
    )
    assert len(tools) == 1
    assert tools[0].approval is ApprovalPolicy.NEVER
    ctx = ToolContext(
        identity=Identity(subject_id="u1"),
        session_id="s1",
        run_id="r1",
        deps={"Authorization": "Bearer t"},
    )
    result = await tools[0](ctx)
    assert result.data["ok"] is True
    assert result.public["body"]["items"][0]["id"] == "1"
    await async_client.aclose()


@pytest.mark.asyncio
async def test_http_post_wraps_flat_json_string_body():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "d1", "status": "draft"})

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    settings = HttpToolSettings(base_url="http://gw.test", client=async_client)
    tools = tools_from_operations(
        [
            HttpOperation(
                name="create_vm_draft",
                method="POST",
                path="/api/vm/create-drafts",
                summary="Create draft",
                domain="vm",
                approval="never",
                wrap_body="inputs",
            )
        ],
        settings,
    )
    ctx = ToolContext(
        identity=Identity(subject_id="u1"),
        session_id="s1",
        run_id="r1",
        deps={"Authorization": "Bearer t"},
    )
    result = await tools[0](ctx, body=json.dumps({"name": "n1", "zone_id": "z1"}))
    assert result.data["ok"] is True
    assert captured["body"] == {"inputs": {"name": "n1", "zone_id": "z1"}}
    await async_client.aclose()


def test_create_openapi_agent_builds_tools():
    agent = create_openapi_agent(
        name="aidy",
        instructions="You are Aidy.",
        operations=[
            {
                "name": "list_vms",
                "method": "GET",
                "path": "/api/vm",
                "summary": "List VMs",
                "domain": "vm",
            }
        ],
        http=HttpToolSettings(base_url="http://gw.test"),
    )
    assert [tool.name for tool in agent.tools] == ["list_vms"]
