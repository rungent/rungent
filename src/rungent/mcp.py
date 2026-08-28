"""Optional FastAPI adapter that exposes Agent tools over Streamable HTTP MCP."""

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from .agent import Agent
from .skill import exportable_tools
from .state import Identity, new_id
from .tools import Tool, ToolContext, ToolEffect, validation_error_message

IdentityResolver = Callable[[Any], Identity | Awaitable[Identity]]
ContextFactory = Callable[[Any], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]
ResourceMetadataUrl = str | Callable[[Any], str]


def mcp_annotations(tool: Tool) -> dict[str, Any]:
    effect = tool.effect if isinstance(tool.effect, ToolEffect) else ToolEffect(tool.effect)
    return {
        "title": tool.title,
        "readOnlyHint": effect is ToolEffect.READ,
        "destructiveHint": effect is ToolEffect.DESTRUCTIVE,
        "openWorldHint": effect is ToolEffect.EXTERNAL,
        "idempotentHint": bool(tool.deduplicate and effect is not ToolEffect.READ),
    }


def mcp_tool_descriptor(tool: Tool) -> dict[str, Any]:
    schema = tool.input_model.model_json_schema()
    schema.pop("title", None)
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": schema,
        "annotations": mcp_annotations(tool),
        "_meta": {
            "effect": str(tool.effect),
            "approval": str(tool.approval),
            **({"confirmation": tool.confirmation} if isinstance(tool.confirmation, str) else {}),
        },
    }


async def invoke_tool(
    tool: Tool,
    arguments: Mapping[str, Any] | None,
    *,
    identity: Identity,
    deps: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    ctx = ToolContext(
        identity=identity,
        session_id=session_id or f"mcp_{uuid4().hex}",
        run_id=run_id or new_id("run"),
        deps=deps or {},
    )
    try:
        result = await tool(ctx, **dict(arguments or {}))
    except ValidationError as exc:
        return {
            "isError": True,
            "content": [{"type": "text", "text": validation_error_message(exc)}],
        }
    if result.interaction is not None:
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "error": "missing_arguments",
                            "message": result.interaction.prompt
                            or "Required arguments are missing",
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
        }
    text = result.for_model()
    return {"content": [{"type": "text", "text": text}], "isError": not result.succeeded}


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _jsonrpc_error(rpc_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


def _jsonrpc_result(rpc_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def create_mcp_asgi(
    agent: Agent,
    *,
    identity_resolver: IdentityResolver,
    context_factory: ContextFactory | None = None,
    tools: Sequence[Tool] | None = None,
    server_name: str | None = None,
    server_version: str = "1.0.0",
    protocol_version: str = "2025-03-26",
    resource_metadata_url: ResourceMetadataUrl | None = None,
):
    """Return a FastAPI router that serves MCP JSON-RPC on POST ``/`` (mount at ``/mcp``)."""
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse, Response

    selected = exportable_tools(agent.tools if tools is None else tools)
    catalog = {item.name: item for item in selected}
    router = APIRouter()

    def metadata_url(request: Request) -> str | None:
        if resource_metadata_url is None:
            return None
        if callable(resource_metadata_url):
            return resource_metadata_url(request)
        return resource_metadata_url

    def unauthorized(request: Request) -> JSONResponse:
        headers: dict[str, str] = {"Access-Control-Allow-Origin": "*"}
        url = metadata_url(request)
        if url:
            headers["WWW-Authenticate"] = f'Bearer realm="{agent.name}", resource_metadata="{url}"'
        return JSONResponse({"error": "unauthorized"}, status_code=401, headers=headers)

    async def identity(request: Request) -> Identity:
        return await _resolve(identity_resolver(request))

    async def deps(request: Request) -> Mapping[str, Any]:
        if context_factory is None:
            return {}
        return await _resolve(context_factory(request))

    async def dispatch(request: Request, body: dict[str, Any]) -> dict[str, Any] | None:
        rpc_id = body.get("id")
        method = body.get("method")
        raw_params = body.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        if method == "initialize":
            return _jsonrpc_result(
                rpc_id,
                {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": server_name or agent.name,
                        "version": server_version,
                    },
                },
            )
        if method == "notifications/initialized" or method == "notifications/cancelled":
            return None
        if method == "ping":
            return _jsonrpc_result(rpc_id, {})
        if method == "tools/list":
            return _jsonrpc_result(
                rpc_id,
                {"tools": [mcp_tool_descriptor(item) for item in selected]},
            )
        if method == "tools/call":
            name = str(params.get("name") or "")
            tool = catalog.get(name)
            if tool is None:
                return _jsonrpc_error(rpc_id, -32601, f"Unknown tool: {name}")
            resolved = await identity(request)
            result = await invoke_tool(
                tool,
                params.get("arguments") if isinstance(params.get("arguments"), dict) else {},
                identity=resolved,
                deps=await deps(request),
            )
            return _jsonrpc_result(rpc_id, result)
        return _jsonrpc_error(rpc_id, -32601, f"Unknown method: {method}")

    @router.options("")
    @router.options("/")
    async def options_mcp() -> Response:
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Authorization, Content-Type, MCP-Protocol-Version",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            },
        )

    @router.get("")
    @router.get("/")
    async def get_mcp(request: Request) -> JSONResponse:
        try:
            await identity(request)
        except PermissionError:
            return unauthorized(request)
        return JSONResponse(
            {"error": "method_not_allowed", "hint": "Use POST JSON-RPC"},
            status_code=405,
            headers={"Access-Control-Allow-Origin": "*"},
        )

    @router.post("")
    @router.post("/")
    async def post_mcp(request: Request) -> Response:
        try:
            await identity(request)
        except PermissionError:
            return unauthorized(request)
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                _jsonrpc_error(None, -32700, "Parse error"),
                status_code=400,
                headers={"Access-Control-Allow-Origin": "*"},
            )
        messages = payload if isinstance(payload, list) else [payload]
        replies: list[dict[str, Any]] = []
        for item in messages:
            if not isinstance(item, dict):
                replies.append(_jsonrpc_error(None, -32600, "Invalid Request"))
                continue
            reply = await dispatch(request, item)
            if reply is not None:
                replies.append(reply)
        if not replies:
            return Response(status_code=202, headers={"Access-Control-Allow-Origin": "*"})
        body = replies if isinstance(payload, list) else replies[0]
        return JSONResponse(body, headers={"Access-Control-Allow-Origin": "*"})

    return router
