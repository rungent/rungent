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
async def test_shortlist_low_confidence_degrades_to_full_catalog():
    settings = SystemOneSettings(base_url="http://systemone.test", confidence_threshold=0.6)
    client = SystemOneClient(settings)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": {
                    "needs_tool": {"type": "noul", "noul": 0.2},
                    "domain": {
                        "type": "choice",
                        "choice": "vm",
                        "confidence": 0.9,
                        "probabilities": {"vm": 0.9},
                    },
                }
            },
        )

    transport = httpx.MockTransport(handler)
    client._client = httpx.AsyncClient(base_url=settings.base_url, transport=transport)
    client._owns_client = True

    names = await shortlist_tools(
        client,
        tools=[_tool("list_vms", "vm")],
        user_input="你好",
        recent_tool_names=[],
    )
    assert names is None
    await client.aclose()


@pytest.mark.asyncio
async def test_shortlist_unknown_domain_degrades_to_full_catalog():
    settings = SystemOneSettings(base_url="http://systemone.test", confidence_threshold=0.5)
    client = SystemOneClient(settings)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": {
                    "needs_tool": {"type": "noul", "noul": 0.9},
                    "domain": {
                        "type": "choice",
                        "choice": "unknown",
                        "confidence": 0.9,
                        "probabilities": {"unknown": 0.9},
                    },
                }
            },
        )

    transport = httpx.MockTransport(handler)
    client._client = httpx.AsyncClient(base_url=settings.base_url, transport=transport)
    client._owns_client = True

    names = await shortlist_tools(
        client,
        tools=[_tool("list_vms", "vm")],
        user_input="列出云服务器",
        recent_tool_names=[],
    )
    assert names is None
    await client.aclose()


def test_systemone_from_env_strips_v1_suffix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SYSTEMONE_BASE_URL", "https://laya.example/v1")
    settings = SystemOneSettings.from_env()
    assert settings is not None
    assert settings.base_url == "https://laya.example"


@pytest.mark.asyncio
async def test_http_prepare_fetches_revision_when_missing():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.url.path == "/api/vm/create-drafts/d1"
            return httpx.Response(200, json={"id": "d1", "revision": 7, "status": "draft"})
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "d1", "status": "prepared", "revision": 7})

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    settings = HttpToolSettings(base_url="http://gw.test", client=async_client)
    tools = tools_from_operations(
        [
            HttpOperation(
                name="prepare_vm_draft",
                method="POST",
                path="/api/vm/create-drafts/{draft_id}/prepare",
                summary="Prepare draft",
                domain="vm",
                approval="never",
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
    result = await tools[0](ctx, draft_id="d1")
    assert result.data["ok"] is True
    assert captured["path"] == "/api/vm/create-drafts/d1/prepare"
    assert captured["body"] == {"revision": 7}
    await async_client.aclose()


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


@pytest.mark.asyncio
async def test_http_post_top_level_fields_merge_into_wrapped_body():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "d1"})

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
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "zone_id": {"type": "string"},
                        "body": {"type": "object"},
                    },
                },
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
    result = await tools[0](ctx, name="n1", zone_id="z1")
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
