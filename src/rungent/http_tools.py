"""Build Rungent tools from OpenAPI operations or an explicit allowlist."""

import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, create_model

from .execution import redact
from .state import (
    ApprovalImpact,
    InteractionOption,
    InteractionQuestion,
    InteractionRequest,
    ToolContinuation,
    ToolResult,
)
from .tools import ApprovalPolicy, Tool, ToolContext, ToolEffect, ToolFunction

AuthorizationFactory = Callable[[ToolContext], str | None | Awaitable[str | None]]
RedactFn = Callable[[Any], Any]

_PATH_PARAM = re.compile(r"\{([^/}]+)\}")
_IP_KEYS = frozenset(
    {
        "internal_ip",
        "ip",
        "ip_addr",
        "ip_address",
        "ip_addresses",
        "ipaddress",
    }
)


def default_redact(value: Any, *, strip_ips: bool = True) -> Any:
    cleaned = redact(value)
    if not strip_ips:
        return cleaned
    if isinstance(cleaned, dict):
        return {
            key: default_redact(item, strip_ips=True)
            for key, item in cleaned.items()
            if str(key).lower() not in _IP_KEYS
        }
    if isinstance(cleaned, list):
        return [default_redact(item, strip_ips=True) for item in cleaned]
    return cleaned


@dataclass(frozen=True, slots=True)
class HttpOperation:
    """One allowlisted HTTP operation exposed as a tool."""

    name: str
    method: str
    path: str
    summary: str
    domain: str = "general"
    parameters: dict[str, Any] = field(default_factory=dict)
    effect: str | None = None
    approval: str | None = None
    confirmation: str | None = None
    timeout_seconds: float = 120.0
    strip_ips: bool = False
    # When set (e.g. "inputs"), wrap a flat JSON object as {inputs: ...} if the key is absent.
    wrap_body: str | None = None

    def __post_init__(self) -> None:
        method = self.method.upper()
        object.__setattr__(self, "method", method)
        if not self.name or not self.path.startswith("/"):
            raise ValueError(f"Invalid HTTP operation: {self.name!r} {self.path!r}")


@dataclass(frozen=True, slots=True)
class HttpToolSettings:
    base_url: str
    authorization: AuthorizationFactory | None = None
    redact: RedactFn = default_redact
    client: httpx.AsyncClient | None = None
    default_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))


def _json_type_to_annotation(schema: dict[str, Any]) -> Any:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return Any
    type_name = schema.get("type")
    if isinstance(type_name, list):
        type_name = next((item for item in type_name if item != "null"), "string")
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list[Any],
        "object": dict[str, Any],
    }
    return mapping.get(str(type_name or "string"), Any)


def _input_model_for(operation: HttpOperation) -> type[BaseModel]:
    props = dict(operation.parameters.get("properties") or {})
    required = set(operation.parameters.get("required") or [])
    path_names = set(_PATH_PARAM.findall(operation.path))
    for name in path_names:
        props.setdefault(name, {"type": "string", "description": f"Path parameter {name}"})
        required.add(name)
    fields: dict[str, Any] = {}
    for name, schema in props.items():
        if not isinstance(schema, dict):
            schema = {"type": "string"}
        annotation = _json_type_to_annotation(schema)
        description = str(schema.get("description") or name)
        if name in required:
            fields[name] = (annotation, Field(description=description))
        else:
            fields[name] = (annotation | None, Field(default=None, description=description))
    if operation.method in {"POST", "PUT", "PATCH"} and "body" not in fields:
        fields["body"] = (
            dict[str, Any] | list[Any] | str | None,
            Field(default=None, description="JSON request body"),
        )
    if not fields:
        fields["note"] = (str | None, Field(default=None, description="Unused"))
    return create_model(
        f"{operation.name.title().replace('_', '')}Args",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _format_path(template: str, values: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values or values[key] is None:
            raise ValueError(f"Missing path parameter {key}")
        return quote(str(values[key]), safe="")

    return _PATH_PARAM.sub(replace, template)


def _missing_questions(payload: Any) -> list[InteractionQuestion] | None:
    if not isinstance(payload, dict):
        return None
    missing = payload.get("missing") or payload.get("issues")
    if not isinstance(missing, list) or not missing:
        return None
    questions: list[InteractionQuestion] = []
    for item in missing[:8]:
        if isinstance(item, str):
            questions.append(
                InteractionQuestion(id=item, kind="text", prompt=f"请提供 {item}", required=True)
            )
            continue
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or item.get("id") or "").strip()
        message = str(item.get("message") or item.get("prompt") or field).strip()
        if not field:
            continue
        options_raw = item.get("options")
        if isinstance(options_raw, list) and options_raw:
            options = [
                InteractionOption(
                    id=str(opt.get("id") if isinstance(opt, dict) else opt),
                    label=str(
                        (opt.get("label") if isinstance(opt, dict) else opt) or opt
                    ),
                )
                for opt in options_raw
                if (isinstance(opt, dict) and opt.get("id")) or not isinstance(opt, dict)
            ]
            questions.append(
                InteractionQuestion(
                    id=field,
                    kind="choice",
                    prompt=message or field,
                    required=True,
                    options=options,
                )
            )
        else:
            questions.append(
                InteractionQuestion(id=field, kind="text", prompt=message or field, required=True)
            )
    return questions or None


def _default_effect(method: str) -> ToolEffect:
    return ToolEffect.READ if method in {"GET", "HEAD"} else ToolEffect.WRITE


def _default_approval(method: str) -> ApprovalPolicy:
    return ApprovalPolicy.NEVER if method in {"GET", "HEAD"} else ApprovalPolicy.ALWAYS


def _confirmation_for(operation: HttpOperation) -> str | None:
    if _default_approval(operation.method) is ApprovalPolicy.NEVER and not operation.approval:
        return None
    if operation.confirmation:
        return operation.confirmation
    return f"确认调用 {operation.method} {operation.path}？"


async def _resolve_auth(
    settings: HttpToolSettings, ctx: ToolContext
) -> str | None:
    if settings.authorization is None:
        auth = ctx.deps.get("Authorization") or ctx.deps.get("authorization")
        return str(auth) if auth else None
    value = settings.authorization(ctx)
    if isinstance(value, Awaitable):
        value = await value
    return value


def build_http_tool(operation: HttpOperation, settings: HttpToolSettings) -> Tool:
    input_model = _input_model_for(operation)
    effect = ToolEffect(operation.effect) if operation.effect else _default_effect(operation.method)
    approval = (
        ApprovalPolicy(operation.approval)
        if operation.approval
        else _default_approval(operation.method)
    )
    confirmation = _confirmation_for(operation) if approval is ApprovalPolicy.ALWAYS else None

    async def _confirm(ctx: ToolContext, **arguments: Any) -> ApprovalImpact:
        target = _format_path(operation.path, arguments)
        title = operation.summary or operation.name
        body = arguments.get("body")
        detail = ""
        if isinstance(body, dict) and body:
            detail = json.dumps(settings.redact(body), ensure_ascii=False)[:400]
        effect_text = f"{operation.method} {target}"
        if detail:
            effect_text = f"{effect_text}\n{detail}"
        return ApprovalImpact(title=title, target_label=target, effect=effect_text)

    async def execute(ctx: ToolContext, **arguments: Any) -> ToolResult:
        path_values = {
            name: arguments[name]
            for name in _PATH_PARAM.findall(operation.path)
            if name in arguments and arguments[name] is not None
        }
        url_path = _format_path(operation.path, path_values)
        reserved = set(path_values) | {"body", "note"}
        top_level = {
            key: value
            for key, value in arguments.items()
            if key not in reserved and value is not None
        }
        body = arguments.get("body")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                pass
        # Mutating calls: fold top-level tool args into the JSON body instead of query.
        if operation.method in {"POST", "PUT", "PATCH", "DELETE"}:
            if body is None and top_level:
                body = dict(top_level)
                top_level = {}
            elif isinstance(body, dict) and top_level:
                body = {**top_level, **body}
                top_level = {}
        query = top_level
        if (
            operation.wrap_body
            and isinstance(body, dict)
            and operation.wrap_body not in body
            and body
        ):
            body = {operation.wrap_body: body}
        headers: dict[str, str] = {"Accept": "application/json"}
        auth = await _resolve_auth(settings, ctx)
        if auth:
            headers["Authorization"] = auth if auth.lower().startswith("bearer ") else f"Bearer {auth}"

        client = settings.client
        owns = False
        if client is None:
            client = httpx.AsyncClient(timeout=operation.timeout_seconds)
            owns = True
        try:
            json_body = body if isinstance(body, (dict, list)) else None
            content = None if json_body is not None else (body if isinstance(body, str) else None)
            if json_body is not None:
                headers["Content-Type"] = "application/json"
            response = await client.request(
                operation.method,
                f"{settings.base_url}{url_path}",
                params=query or None,
                json=json_body,
                content=content,
                headers=headers,
            )
        finally:
            if owns:
                await client.aclose()

        try:
            payload: Any = response.json()
        except Exception:
            payload = {"text": response.text[:4000]}

        public = settings.redact(payload)
        if operation.strip_ips:
            public = default_redact(public, strip_ips=True)

        if response.status_code in {400, 422}:
            questions = _missing_questions(payload)
            if questions:
                continuation_args = {
                    key: value for key, value in arguments.items() if value is not None
                }
                return ToolResult(
                    message=str(
                        (payload.get("detail") if isinstance(payload, dict) else None)
                        or "需要补充参数"
                    ),
                    data={"ok": False, "status_code": response.status_code, "body": public},
                    public={"status_code": response.status_code, "body": public},
                    interaction=InteractionRequest(
                        kind="form",
                        prompt="请补充以下信息后继续",
                        questions=questions,
                        continuation=ToolContinuation(
                            tool=operation.name, arguments=continuation_args
                        ),
                    ),
                )

        ok = 200 <= response.status_code < 300
        message = None
        if isinstance(payload, dict):
            message = payload.get("message") or payload.get("detail")
        return ToolResult(
            message=str(message) if message else None,
            data={"ok": ok, "status_code": response.status_code, "body": public},
            public={"status_code": response.status_code, "body": public},
        )

    execute.__name__ = operation.name
    execute.__doc__ = operation.summary
    return Tool(
        name=operation.name,
        description=operation.summary,
        function=execute,
        input_model=input_model,
        effect=effect,
        approval=approval,
        confirmation=_confirm if approval is ApprovalPolicy.ALWAYS else None,
        approval_ready=None,
        title=operation.summary[:80] or operation.name,
        timeout_seconds=operation.timeout_seconds or settings.default_timeout_seconds,
        parallel=operation.method in {"GET", "HEAD"},
        deduplicate=True,
        requires_interaction_response=False,
        domain=operation.domain,
    )


def tools_from_operations(
    operations: Sequence[HttpOperation | Mapping[str, Any]],
    settings: HttpToolSettings,
) -> list[Tool]:
    tools: list[Tool] = []
    for item in operations:
        operation = item if isinstance(item, HttpOperation) else HttpOperation(**dict(item))
        tools.append(build_http_tool(operation, settings))
    names = [tool.name for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate HTTP tool names in allowlist")
    return tools


def operations_from_openapi(
    spec: Mapping[str, Any],
    *,
    allowlist: Sequence[tuple[str, str]] | None = None,
    require_aidy_flag: bool = False,
    path_prefix: str = "",
) -> list[HttpOperation]:
    """Convert OpenAPI paths into HttpOperation entries.

    allowlist entries are (METHOD, path) with OpenAPI path templates.
    When require_aidy_flag is true, only operations with ``x-aidy: true`` / ``aidy: true`` are kept.
    """
    allowed = {(method.upper(), path) for method, path in allowlist} if allowlist else None
    paths = spec.get("paths") or {}
    operations: list[HttpOperation] = []
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        full_path = f"{path_prefix.rstrip('/')}{path}" if path_prefix else path
        for method, operation in methods.items():
            if method.startswith("x-") or not isinstance(operation, dict):
                continue
            method_u = method.upper()
            if method_u not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}:
                continue
            if allowed is not None and (method_u, path) not in allowed and (method_u, full_path) not in allowed:
                continue
            extra = operation.get("x-aidy") or operation.get("aidy")
            if require_aidy_flag and extra is not True:
                continue
            tags = operation.get("tags") or []
            domain = str(tags[0]).lower().replace(" ", "_") if tags else "general"
            name = str(operation.get("operationId") or f"{method_u.lower()}_{path.strip('/').replace('/', '_').replace('{', '').replace('}', '')}")
            name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            summary = str(operation.get("summary") or operation.get("description") or name)
            properties: dict[str, Any] = {}
            required: list[str] = []
            for param in operation.get("parameters") or []:
                if not isinstance(param, dict):
                    continue
                param_name = str(param.get("name") or "")
                if not param_name:
                    continue
                schema = param.get("schema") or {"type": "string"}
                if param.get("description"):
                    schema = {**schema, "description": param["description"]}
                properties[param_name] = schema
                if param.get("required"):
                    required.append(param_name)
            body = (operation.get("requestBody") or {}).get("content", {})
            json_body = body.get("application/json") if isinstance(body, dict) else None
            if isinstance(json_body, dict) and json_body.get("schema"):
                properties["body"] = {
                    "type": "object",
                    "description": "JSON request body",
                }
                if operation.get("requestBody", {}).get("required"):
                    required.append("body")
            operations.append(
                HttpOperation(
                    name=name,
                    method=method_u,
                    path=full_path,
                    summary=summary[:300],
                    domain=domain,
                    parameters={"type": "object", "properties": properties, "required": required},
                )
            )
    return operations


def load_operations(data: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[HttpOperation]:
    if isinstance(data, Mapping):
        items = data.get("operations") or data.get("allowlist") or []
    else:
        items = data
    return [item if isinstance(item, HttpOperation) else HttpOperation(**dict(item)) for item in items]
