"""Tests for System One shortlist and HTTP allowlist tools."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field
from rungent import ToolContext, create_openapi_agent
from rungent.http_tools import (
    HttpOperation,
    HttpToolSettings,
    load_operations_from_config,
    tools_from_operations,
)
from rungent.state import Identity
from rungent.systemone import (
    MODE_ACT,
    MODE_CHAT,
    MODE_STICKY,
    SystemOneClient,
    SystemOneSettings,
    shortlist_tools,
)
from rungent.tools import ApprovalPolicy, Tool, ToolEffect


class _Empty(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str | None = Field(default=None)


async def _noop(ctx: ToolContext, **kwargs: Any) -> dict[str, Any]:
    return {"ok": True}


def _tool(name: str, domain: str, *, family: str | None = None) -> Tool:
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
        family=family,
    )


def _one_shot_answers(answers: dict[str, Any]):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "tool" in payload["questions"]
        assert "needs_tool" in payload["questions"]
        assert "acts_on_resources" in payload["questions"]
        assert "prose_suffices" in payload["questions"]
        assert payload["questions"]["needs_tool"]["type"] == "noul"
        assert "criteria" in payload["questions"]["needs_tool"]
        return httpx.Response(200, json={"answers": answers})

    return handler


def _high_gate_answers(*, choice: str, confidence: float = 0.95) -> dict[str, Any]:
    return {
        "needs_tool": {"type": "noul", "noul": 0.9},
        "acts_on_resources": {"type": "noul", "noul": 0.9},
        "prose_suffices": {"type": "noul", "noul": 0.1},
        "tool": {
            "type": "choice",
            "choice": choice,
            "confidence": confidence,
            "probabilities": {choice: confidence},
        },
    }


@pytest.mark.asyncio
async def test_shortlist_act_picks_tool_not_other_domain():
    settings = SystemOneSettings(base_url="http://systemone.test", confidence_threshold=0.5)
    client = SystemOneClient(settings)
    client._client = httpx.AsyncClient(
        base_url=settings.base_url,
        transport=httpx.MockTransport(_one_shot_answers(_high_gate_answers(choice="list_vms"))),
    )
    client._owns_client = True

    result = await shortlist_tools(
        client,
        tools=[_tool("list_vms", "vm"), _tool("list_elastic", "elastic")],
        user_input="列出云服务器",
        recent_tool_names=[],
    )
    assert result.mode == MODE_ACT
    assert "list_vms" in result.include
    assert "list_elastic" not in result.include
    await client.aclose()


@pytest.mark.asyncio
async def test_shortlist_act_merges_family_within_max_tools():
    settings = SystemOneSettings(
        base_url="http://systemone.test",
        confidence_threshold=0.5,
        max_tools=4,
    )
    client = SystemOneClient(settings)
    client._client = httpx.AsyncClient(
        base_url=settings.base_url,
        transport=httpx.MockTransport(
            _one_shot_answers(_high_gate_answers(choice="create_vm_draft"))
        ),
    )
    client._owns_client = True

    result = await shortlist_tools(
        client,
        tools=[
            _tool("list_zones", "vm"),
            _tool("list_vms", "vm"),
            _tool("get_vm", "vm"),
            _tool("create_vm_draft", "vm", family="vm_draft"),
            _tool("prepare_vm_draft", "vm", family="vm_draft"),
            _tool("commit_vm_draft", "vm", family="vm_draft"),
            _tool("patch_vm_draft", "vm", family="vm_draft"),
        ],
        user_input="创建一台云服务器",
        recent_tool_names=[],
    )
    assert result.mode == MODE_ACT
    assert "create_vm_draft" in result.include
    assert "prepare_vm_draft" in result.include
    assert "commit_vm_draft" in result.include
    assert "list_zones" not in result.include
    assert len(result.include) <= 4
    await client.aclose()


@pytest.mark.asyncio
async def test_shortlist_low_noul_is_chat_empty_include():
    settings = SystemOneSettings(base_url="http://systemone.test", confidence_threshold=0.6)
    client = SystemOneClient(settings)
    client._client = httpx.AsyncClient(
        base_url=settings.base_url,
        transport=httpx.MockTransport(
            _one_shot_answers(
                {
                    "needs_tool": {"type": "noul", "noul": 0.2},
                    "acts_on_resources": {"type": "noul", "noul": 0.2},
                    "prose_suffices": {"type": "noul", "noul": 0.9},
                    "tool": {
                        "type": "choice",
                        "choice": "list_vms",
                        "confidence": 0.9,
                        "probabilities": {"list_vms": 0.9},
                    },
                }
            )
        ),
    )
    client._owns_client = True

    result = await shortlist_tools(
        client,
        tools=[_tool("list_vms", "vm"), _tool("list_elastic", "elastic")],
        user_input="你好",
        recent_tool_names=[],
    )
    assert result.mode == MODE_CHAT
    assert result.include == set()
    await client.aclose()


@pytest.mark.asyncio
async def test_shortlist_low_confidence_uses_sticky_not_full_catalog():
    settings = SystemOneSettings(
        base_url="http://systemone.test",
        confidence_threshold=0.6,
        max_tools=3,
    )
    client = SystemOneClient(settings)
    client._client = httpx.AsyncClient(
        base_url=settings.base_url,
        transport=httpx.MockTransport(
            _one_shot_answers(
                {
                    **_high_gate_answers(choice="create_vm_draft", confidence=0.4),
                    "tool": {
                        "type": "choice",
                        "choice": "create_vm_draft",
                        "confidence": 0.4,
                        "probabilities": {"create_vm_draft": 0.4, "list_vms": 0.3},
                    },
                }
            )
        ),
    )
    client._owns_client = True

    catalog = [
        _tool("list_zones", "vm"),
        _tool("list_vms", "vm"),
        _tool("create_vm_draft", "vm", family="vm_draft"),
        _tool("prepare_vm_draft", "vm", family="vm_draft"),
        _tool("commit_vm_draft", "vm", family="vm_draft"),
        _tool("list_elastic", "elastic"),
    ]
    result = await shortlist_tools(
        client,
        tools=catalog,
        user_input="创建一台云服务器",
        recent_tool_names=["create_vm_draft"],
        previous_include={"create_vm_draft", "prepare_vm_draft"},
    )
    assert result.mode == MODE_STICKY
    assert result.include == {"create_vm_draft", "prepare_vm_draft"}
    assert "list_elastic" not in result.include
    assert len(result.include) < len(catalog)
    await client.aclose()


@pytest.mark.asyncio
async def test_shortlist_transport_failure_sticky_not_full_catalog():
    settings = SystemOneSettings(base_url="http://systemone.test", max_tools=2)
    client = SystemOneClient(settings)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client._client = httpx.AsyncClient(
        base_url=settings.base_url,
        transport=httpx.MockTransport(handler),
    )
    client._owns_client = True

    catalog = [
        _tool("list_vms", "vm"),
        _tool("create_vm_draft", "vm", family="vm_draft"),
        _tool("prepare_vm_draft", "vm", family="vm_draft"),
        _tool("list_elastic", "elastic"),
    ]
    result = await shortlist_tools(
        client,
        tools=catalog,
        user_input="继续",
        recent_tool_names=["create_vm_draft"],
        previous_include={"create_vm_draft", "prepare_vm_draft", "commit_vm_draft"},
    )
    assert result.mode == MODE_STICKY
    assert result.include == {"create_vm_draft", "prepare_vm_draft"}
    assert "list_elastic" not in result.include
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
async def test_http_patch_wrap_body_fetches_revision():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.url.path == "/api/vm/create-drafts/d1"
            return httpx.Response(200, json={"id": "d1", "revision": 3, "status": "draft"})
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "d1", "revision": 4, "status": "draft"})

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    settings = HttpToolSettings(base_url="http://gw.test", client=async_client)
    tools = tools_from_operations(
        [
            HttpOperation(
                name="patch_vm_draft",
                method="PATCH",
                path="/api/vm/create-drafts/{draft_id}",
                summary="Patch draft",
                domain="vm",
                approval="never",
                wrap_body="inputs",
                parameters={
                    "type": "object",
                    "properties": {
                        "draft_id": {"type": "string"},
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
    result = await tools[0](ctx, draft_id="d1", zone_id="z1")
    assert result.data["ok"] is True
    assert captured["path"] == "/api/vm/create-drafts/d1"
    assert captured["body"] == {"revision": 3, "inputs": {"zone_id": "z1"}}
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


def test_load_operations_from_config_uses_snapshot_and_overrides(tmp_path: Path):
    snapshot = {
        "openapi": "3.1.0",
        "paths": {
            "/vm": {
                "get": {
                    "operationId": "list_vms_openapi",
                    "summary": "List virtual machines",
                    "description": "Paginated VM inventory",
                    "tags": ["虚拟机"],
                    "parameters": [
                        {
                            "name": "page",
                            "in": "query",
                            "schema": {"type": "integer"},
                        }
                    ],
                }
            }
        },
    }
    snapshot_path = tmp_path / "cs.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    config_path = tmp_path / "tools.json"
    config_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "cs",
                        "openapi_file": "cs.json",
                        "path_prefix": "/api",
                    }
                ],
                "include": [["GET", "/vm"]],
                "overrides": {
                    "GET /api/vm": {
                        "name": "list_vms",
                        "domain": "vm",
                        "family": "vm",
                        "strip_ips": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    ops = load_operations_from_config(config_path)
    assert len(ops) == 1
    assert ops[0].name == "list_vms"
    assert ops[0].path == "/api/vm"
    assert ops[0].strip_ips is True
    assert ops[0].family == "vm"
    assert "Paginated" in ops[0].summary
