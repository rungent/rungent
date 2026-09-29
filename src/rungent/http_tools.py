"""Build Rungent tools from OpenAPI operations or an explicit allowlist."""

import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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
    confirmation: str | Mapping[str, Any] | None = None
    timeout_seconds: float = 120.0
    strip_ips: bool = False
    # When set (e.g. "inputs"), wrap a flat JSON object as {inputs: ...} if the key is absent.
    wrap_body: str | None = None
    family: str | None = None

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


def _create_draft_resource_path(url_path: str) -> str | None:
    """Draft GET path for create-drafts mutate (PATCH / prepare / commit)."""
    path = url_path.rstrip("/")
    for suffix in ("/prepare", "/commit"):
        if not path.endswith(suffix):
            continue
        parent = path[: -len(suffix)]
        if "/create-drafts/" in parent:
            return parent
    # PATCH/PUT on .../create-drafts/{draft_id}
    if "/create-drafts/" in path:
        after = path.split("/create-drafts/", 1)[1]
        if after and "/" not in after:
            return path
    return None


class _FormatMap(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _http_error_message(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "请求失败"
    detail = payload.get("detail")
    if isinstance(detail, list):
        return "参数不合法，请检查后重试"
    if isinstance(detail, dict):
        message = detail.get("msg") or detail.get("message") or detail.get("detail")
        if message:
            return str(message).strip()[:500] or "请求失败"
    for key in ("message", "detail", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:500]
    return "请求失败"


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

    async def _confirm(ctx: ToolContext, **arguments: Any) -> ApprovalImpact:
        path_values = {
            name: arguments[name]
            for name in _PATH_PARAM.findall(operation.path)
            if name in arguments and arguments[name] is not None
        }
        try:
            url_path = _format_path(operation.path, path_values)
        except ValueError:
            url_path = operation.path

        draft_path = _create_draft_resource_path(url_path)
        if draft_path and url_path.rstrip("/").endswith("/commit"):
            headers: dict[str, str] = {"Accept": "application/json"}
            auth = await _resolve_auth(settings, ctx)
            if auth:
                headers["Authorization"] = (
                    auth if auth.lower().startswith("bearer ") else f"Bearer {auth}"
                )
            client = settings.client
            owns = False
            if client is None:
                client = httpx.AsyncClient(timeout=operation.timeout_seconds)
                owns = True
            try:
                response = await client.request(
                    "GET",
                    f"{settings.base_url}{draft_path}",
                    headers=headers,
                )
                try:
                    payload: Any = response.json()
                except Exception:
                    payload = None
            finally:
                if owns:
                    await client.aclose()
            if (
                200 <= response.status_code < 300
                and isinstance(payload, dict)
                and isinstance(payload.get("confirmation"), dict)
            ):
                confirmation = payload["confirmation"]
                fields = confirmation.get("fields") or {}
                if not isinstance(fields, dict):
                    fields = {}
                title = str(confirmation.get("title") or operation.summary or "确认创建云服务器")
                target = str(fields.get("名称") or arguments.get("draft_id") or title)
                changes = [f"{key}：{value}" for key, value in fields.items()]
                effect_text = "提交后将占用配额并开始开通"
                if isinstance(operation.confirmation, str) and operation.confirmation:
                    effect_text = (
                        operation.confirmation.format_map(_FormatMap(arguments)).strip()
                        or effect_text
                    )
                return ApprovalImpact(
                    title=title,
                    target_label=target,
                    effect=effect_text,
                    changes=changes,
                )

        if isinstance(operation.confirmation, Mapping):
            conf = dict(operation.confirmation)
            title = str(conf.get("title") or operation.summary or operation.name)
            title = title.format_map(_FormatMap(arguments)).strip() or title
            fields_spec = conf.get("fields") or []
            changes: list[str] = []
            target = title
            if isinstance(fields_spec, list):
                for item in fields_spec:
                    if not isinstance(item, Mapping):
                        continue
                    key = str(item.get("key") or "")
                    label = str(item.get("label") or key)
                    source = str(item.get("source") or "args")
                    value = arguments.get(key) if source == "args" else None
                    if value is None and key in arguments:
                        value = arguments[key]
                    if value is not None:
                        changes.append(f"{label}：{value}")
                        if target == title:
                            target = str(value)
            effect = str(conf.get("effect") or title)
            return ApprovalImpact(
                title=title,
                target_label=str(target),
                effect=effect,
                changes=changes,
            )

        if isinstance(operation.confirmation, str) and operation.confirmation:
            title = operation.confirmation.format_map(_FormatMap(arguments)).strip()
            if not title:
                title = operation.summary or operation.name
            target = str(
                arguments.get("id")
                or arguments.get("vm_id")
                or arguments.get("system_id")
                or arguments.get("elastic_id")
                or arguments.get("draft_id")
                or arguments.get("name")
                or title
            )
            return ApprovalImpact(title=title, target_label=target, effect=title)

        title = operation.summary or operation.name
        target = str(
            arguments.get("id")
            or arguments.get("vm_id")
            or arguments.get("system_id")
            or arguments.get("elastic_id")
            or arguments.get("draft_id")
            or arguments.get("name")
            or title
        )
        return ApprovalImpact(title=title, target_label=target, effect=title)

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
            elif body is None and operation.method in {"POST", "PUT", "PATCH"}:
                body = {}
        query = top_level
        if (
            operation.wrap_body
            and isinstance(body, dict)
            and operation.wrap_body not in body
            and body
        ):
            # Keep protocol fields (e.g. draft revision) outside the wrapped payload.
            keep = {"revision", "vg_ssh_keys"}
            outer = {key: body[key] for key in keep if key in body}
            inner = {key: value for key, value in body.items() if key not in keep}
            body = {**outer, operation.wrap_body: inner} if inner else outer
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
            draft_path = _create_draft_resource_path(url_path)
            if (
                draft_path
                and operation.method in {"POST", "PUT", "PATCH"}
                and (not isinstance(body, dict) or body.get("revision") is None)
            ):
                draft_response = await client.request(
                    "GET",
                    f"{settings.base_url}{draft_path}",
                    headers=headers,
                )
                try:
                    draft_payload: Any = draft_response.json()
                except Exception:
                    draft_payload = None
                if (
                    200 <= draft_response.status_code < 300
                    and isinstance(draft_payload, dict)
                    and draft_payload.get("revision") is not None
                ):
                    body = {
                        **(body if isinstance(body, dict) else {}),
                        "revision": draft_payload["revision"],
                    }
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
                    message=_http_error_message(payload)
                    if isinstance(payload, dict) and isinstance(payload.get("detail"), list)
                    else str(
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
        message = None if ok else _http_error_message(payload)
        if ok and isinstance(payload, dict):
            raw = payload.get("message")
            message = str(raw) if raw else None
        return ToolResult(
            message=message,
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
        family=operation.family,
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
    When require_aidy_flag is true, only operations with ``x-aidy: true`` or an
    ``x-aidy`` object (or ``aidy``) are kept. Prefer a single ``x-aidy`` object
    on the owning service for domain/family/approval/confirmation/effect/
    strip_ips/wrap_body; flat ``x-aidy-*`` keys are still read as a fallback.
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
            if allowed is not None:
                aliases = _path_aliases(path) | _path_aliases(full_path)
                allowed_hit = any(
                    method_u == want_method and bool(aliases & _path_aliases(want_path))
                    for want_method, want_path in allowed
                )
                if not allowed_hit:
                    continue
            extra = operation.get("x-aidy") or operation.get("aidy")
            if require_aidy_flag:
                if extra is True:
                    meta: Mapping[str, Any] = {}
                elif isinstance(extra, Mapping):
                    meta = extra
                else:
                    continue
            else:
                meta = extra if isinstance(extra, Mapping) else {}
            tags = operation.get("tags") or []
            domain = str(
                operation.get("x-aidy-domain")
                or meta.get("domain")
                or (str(tags[0]).lower().replace(" ", "_") if tags else "general")
            )
            name = str(
                operation.get("operationId")
                or f"{method_u.lower()}_{path.strip('/').replace('/', '_').replace('{', '').replace('}', '')}"
            )
            name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            summary_parts = [
                str(part).strip()
                for part in (operation.get("summary"), operation.get("description"))
                if part
            ]
            summary = " — ".join(dict.fromkeys(summary_parts)) if summary_parts else name
            properties: dict[str, Any] = {}
            required: list[str] = []
            for param in operation.get("parameters") or []:
                if not isinstance(param, dict):
                    continue
                if str(param.get("in") or "").lower() in {"header", "cookie"}:
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
                body_schema = json_body["schema"]
                desc = "JSON request body"
                if isinstance(body_schema, dict) and body_schema.get("description"):
                    desc = str(body_schema["description"])
                if isinstance(body_schema, dict) and body_schema.get("type") == "object":
                    for prop_name, prop_schema in (body_schema.get("properties") or {}).items():
                        if prop_name in properties:
                            continue
                        if isinstance(prop_schema, dict):
                            properties[prop_name] = prop_schema
                        else:
                            properties[prop_name] = {"type": "string"}
                    for req in body_schema.get("required") or []:
                        if req not in required:
                            required.append(str(req))
                properties.setdefault(
                    "body",
                    {"type": "object", "description": desc},
                )
                if operation.get("requestBody", {}).get("required") and "body" not in required:
                    # Prefer flattened fields; body remains optional unless no properties.
                    if len(properties) == 1:
                        required.append("body")
            family = operation.get("x-aidy-family") or meta.get("family")
            wrap_body = operation.get("x-aidy-wrap-body") or meta.get("wrap_body")
            strip_ips = bool(
                operation.get("x-aidy-strip-ips")
                or meta.get("strip_ips")
                or False
            )
            approval = operation.get("x-aidy-approval") or meta.get("approval")
            confirmation = operation.get("x-aidy-confirmation") or meta.get("confirmation")
            effect = operation.get("x-aidy-effect") or meta.get("effect")
            name_override = meta.get("name")
            if name_override:
                name = re.sub(r"[^a-zA-Z0-9_]", "_", str(name_override))
            if isinstance(confirmation, Mapping):
                confirmation_value: str | Mapping[str, Any] | None = dict(confirmation)
            elif confirmation:
                confirmation_value = str(confirmation)
            else:
                confirmation_value = None
            operations.append(
                HttpOperation(
                    name=name,
                    method=method_u,
                    path=full_path,
                    summary=summary[:500],
                    domain=domain,
                    parameters={"type": "object", "properties": properties, "required": required},
                    strip_ips=strip_ips,
                    wrap_body=str(wrap_body) if wrap_body else None,
                    family=str(family) if family else None,
                    approval=str(approval) if approval else None,
                    confirmation=confirmation_value,
                    effect=str(effect) if effect else None,
                )
            )
    return operations


def load_operations(data: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[HttpOperation]:
    if isinstance(data, Mapping):
        items = data.get("operations") or data.get("allowlist") or []
    else:
        items = data
    return [item if isinstance(item, HttpOperation) else HttpOperation(**dict(item)) for item in items]


_ENV_VAR = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(template: str, env: Mapping[str, str] | None = None) -> str:
    source = env if env is not None else os.environ

    def repl(match: re.Match[str]) -> str:
        return str(source.get(match.group(1), "") or "")

    return _ENV_VAR.sub(repl, template)


def _resolve_openapi_url(template: str, env: Mapping[str, str] | None = None) -> str:
    expanded = _expand_env(template, env).strip()
    if not expanded or "://" not in expanded:
        return ""
    return expanded


@dataclass(frozen=True, slots=True)
class OpenApiHost:
    """One upstream service whose OpenAPI feeds the HTTP tool catalog.

    Prefer ``base_url`` (service root); rungent appends ``openapi_path``.
    Use ``spec`` for in-proc documents (e.g. the host's own ``app.openapi()``).
    ``openapi_url`` overrides ``base_url`` when set.
    """

    id: str
    base_url: str | None = None
    openapi_url: str | None = None
    openapi_path: str = "/openapi.json"
    path_prefix: str = ""
    spec: Mapping[str, Any] | None = None

    def resolved_openapi_url(self, env: Mapping[str, str] | None = None) -> str:
        if self.openapi_url:
            return _resolve_openapi_url(self.openapi_url, env)
        if not self.base_url:
            return ""
        base = _expand_env(self.base_url, env).strip().rstrip("/")
        if not base or "://" not in base:
            return ""
        path = self.openapi_path if self.openapi_path.startswith("/") else f"/{self.openapi_path}"
        return f"{base}{path}"


def _coerce_openapi_host(value: OpenApiHost | Mapping[str, Any]) -> OpenApiHost:
    if isinstance(value, OpenApiHost):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"OpenAPI host must be OpenApiHost or mapping, got {type(value)!r}")
    data = dict(value)
    if "id" not in data:
        raise ValueError("OpenAPI host requires id")
    allowed = {
        "id",
        "base_url",
        "openapi_url",
        "openapi_path",
        "path_prefix",
        "spec",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown OpenApiHost fields: {sorted(unknown)}")
    return OpenApiHost(**data)  # type: ignore[arg-type]


def load_operations_from_hosts(
    hosts: Sequence[OpenApiHost | Mapping[str, Any]],
    *,
    require_aidy_flag: bool = True,
    env: Mapping[str, str] | None = None,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
    timeout_seconds: float = 15.0,
) -> list[HttpOperation]:
    """Load x-aidy tools from service hosts (base URL or in-proc spec).

    Hosts pass service roots; this resolves ``{base_url}{openapi_path}`` and
    fetches live OpenAPI. Pass ``spec=`` (or ``specs[id]``) to skip the network.
    """
    merged_specs: dict[str, Mapping[str, Any]] = dict(specs or {})
    sources: list[dict[str, Any]] = []
    for raw in hosts:
        host = _coerce_openapi_host(raw)
        if host.spec is not None:
            merged_specs[host.id] = host.spec
        entry: dict[str, Any] = {"id": host.id, "path_prefix": host.path_prefix}
        if host.id not in merged_specs:
            url = host.resolved_openapi_url(env)
            if not url:
                raise ValueError(
                    f"OpenAPI host {host.id!r} needs base_url, openapi_url, or spec"
                )
            entry["openapi_url"] = url
        sources.append(entry)
    return load_operations_from_sources(
        sources,
        require_aidy_flag=require_aidy_flag,
        env=env,
        specs=merged_specs or None,
        timeout_seconds=timeout_seconds,
    )


def _normalize_template(path: str) -> str:
    return _PATH_PARAM.sub("{}", path)


def _override_key(method: str, path: str) -> str:
    return f"{method.upper()} {path}"


def _lookup_override(
    overrides: Mapping[str, Mapping[str, Any]],
    *,
    method: str,
    path: str,
    name: str,
) -> Mapping[str, Any]:
    exact = overrides.get(_override_key(method, path))
    if exact:
        return exact
    norm = _normalize_template(path)
    for key, value in overrides.items():
        if " " not in key:
            continue
        key_method, key_path = key.split(" ", 1)
        if key_method.upper() == method.upper() and _normalize_template(key_path) == norm:
            return value
    return overrides.get(name) or {}


def _path_aliases(path: str) -> set[str]:
    norms = {_normalize_template(path)}
    if path.startswith("/api/") or path == "/api":
        norms.add(_normalize_template(path.removeprefix("/api") or "/"))
    else:
        norms.add(_normalize_template(f"/api{path}" if path.startswith("/") else f"/api/{path}"))
    return norms


def _include_matches(method: str, path: str, include: Sequence[tuple[str, str]]) -> bool:
    aliases = _path_aliases(path)
    for want_method, want_path in include:
        if method != want_method:
            continue
        if aliases & _path_aliases(want_path):
            return True
    return False


def _apply_operation_overrides(
    operations: list[HttpOperation],
    overrides: Mapping[str, Mapping[str, Any]] | None,
) -> list[HttpOperation]:
    if not overrides:
        return operations
    result: list[HttpOperation] = []
    for operation in operations:
        patch = _lookup_override(
            overrides, method=operation.method, path=operation.path, name=operation.name
        )
        if not patch:
            result.append(operation)
            continue
        data = {
            "name": operation.name,
            "method": operation.method,
            "path": operation.path,
            "summary": operation.summary,
            "domain": operation.domain,
            "parameters": operation.parameters,
            "effect": operation.effect,
            "approval": operation.approval,
            "confirmation": operation.confirmation,
            "timeout_seconds": operation.timeout_seconds,
            "strip_ips": operation.strip_ips,
            "wrap_body": operation.wrap_body,
            "family": operation.family,
        }
        data.update({key: value for key, value in patch.items() if value is not None})
        result.append(HttpOperation(**data))
    return result


def _fetch_openapi_spec(url: str, *, timeout_seconds: float = 15.0) -> dict[str, Any]:
    response = httpx.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError(f"OpenAPI response is not an object: {url}")
    return dict(payload)


def _read_openapi_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"OpenAPI file is not an object: {path}")
    return dict(payload)


def load_operations_from_sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    require_aidy_flag: bool = True,
    include: Sequence[tuple[str, str]] | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    static_operations: Sequence[Mapping[str, Any] | HttpOperation] | None = None,
    env: Mapping[str, str] | None = None,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
    timeout_seconds: float = 15.0,
) -> list[HttpOperation]:
    """Load HTTP tools from live OpenAPI sources (no snapshot fallback).

    When ``require_aidy_flag`` is true (default), only operations marked
    ``x-aidy: true`` (or an object) are kept. Prefer putting tool metadata on
    the owning service's OpenAPI instead of a host-side include list.
    """
    payload: dict[str, Any] = {
        "sources": list(sources),
        "require_aidy_flag": require_aidy_flag,
    }
    if include is not None:
        payload["include"] = [list(item) for item in include]
    if overrides is not None:
        payload["overrides"] = dict(overrides)
    if static_operations:
        payload["static_operations"] = [
            item
            if isinstance(item, Mapping)
            else {
                "name": item.name,
                "method": item.method,
                "path": item.path,
                "summary": item.summary,
                "domain": item.domain,
                "parameters": item.parameters,
                "effect": item.effect,
                "approval": item.approval,
                "confirmation": item.confirmation,
                "timeout_seconds": item.timeout_seconds,
                "strip_ips": item.strip_ips,
                "wrap_body": item.wrap_body,
                "family": item.family,
            }
            for item in static_operations
        ]
    # Write through load_operations_from_config via an in-memory path by
    # reusing the same parsing body — call internal by temp config file avoided.
    return _load_operations_payload(
        payload,
        config_dir=Path("."),
        env=env,
        specs=specs,
        timeout_seconds=timeout_seconds,
    )


def load_operations_from_config(
    path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
    timeout_seconds: float = 15.0,
) -> list[HttpOperation]:
    """Load HTTP operations from a thin OpenAPI tools config.

    Config shape::

        {
          "sources": [
            {
              "id": "cs",
              "openapi_url": "${AIDY_CS_OPENAPI_URL}",
              "path_prefix": "/api"
            }
          ],
          "require_aidy_flag": true
        }

    For each source: use ``specs[id]`` if provided; else HTTP ``openapi_url``;
    else ``openapi_file`` relative to the config directory (tests / offline only).
    Live URL fetch failures raise — no snapshot fallback.
    """
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Tools config must be an object: {config_path}")

    # Legacy allowlist shape still supported.
    if "operations" in payload and "sources" not in payload:
        return load_operations(payload)

    return _load_operations_payload(
        dict(payload),
        config_dir=config_path.parent,
        env=env,
        specs=specs,
        timeout_seconds=timeout_seconds,
    )


def _load_operations_payload(
    payload: Mapping[str, Any],
    *,
    config_dir: Path,
    env: Mapping[str, str] | None = None,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
    timeout_seconds: float = 15.0,
) -> list[HttpOperation]:
    include_raw = payload.get("include") or []
    include: list[tuple[str, str]] | None = None
    if include_raw:
        include = []
        for item in include_raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                include.append((str(item[0]).upper(), str(item[1])))
            elif isinstance(item, Mapping):
                include.append((str(item["method"]).upper(), str(item["path"])))
            else:
                raise ValueError(f"Invalid include entry: {item!r}")

    # Default on for source-based configs: catalog = x-aidy on owning OpenAPI.
    if "require_aidy_flag" in payload:
        require_aidy_flag = bool(payload["require_aidy_flag"])
    else:
        require_aidy_flag = bool(payload.get("sources"))
    operations: list[HttpOperation] = []
    seen: set[tuple[str, str]] = set()

    for source in payload.get("sources") or []:
        if not isinstance(source, Mapping):
            raise ValueError(f"Invalid OpenAPI source: {source!r}")
        source_id = str(
            source.get("id") or source.get("openapi_url") or source.get("openapi_file") or "source"
        )
        path_prefix = str(source.get("path_prefix") or "")
        spec: Mapping[str, Any] | None = None
        if specs and source_id in specs:
            spec = specs[source_id]
        else:
            url_template = str(source.get("openapi_url") or "")
            url = _resolve_openapi_url(url_template, env) if url_template else ""
            file_name = str(source.get("openapi_file") or "").strip()
            file_path: Path | None = None
            if file_name:
                file_path = Path(file_name)
                if not file_path.is_absolute():
                    file_path = config_dir / file_path
            if url:
                try:
                    spec = _fetch_openapi_spec(url, timeout_seconds=timeout_seconds)
                except Exception as exc:
                    raise RuntimeError(f"Failed to fetch OpenAPI from {url}: {exc}") from exc
            else:
                if file_path is None:
                    raise ValueError(
                        f"OpenAPI source {source_id!r} needs openapi_url, openapi_file, "
                        f"or specs[{source_id!r}]"
                    )
                spec = _read_openapi_file(file_path)
        parsed = operations_from_openapi(
            spec,
            allowlist=include,
            require_aidy_flag=require_aidy_flag,
            path_prefix=path_prefix,
        )
        for operation in parsed:
            key = (operation.method, operation.path)
            if key in seen:
                continue
            seen.add(key)
            operations.append(operation)

    for item in payload.get("static_operations") or []:
        operation = item if isinstance(item, HttpOperation) else HttpOperation(**dict(item))
        key = (operation.method, operation.path)
        if key in seen:
            continue
        seen.add(key)
        operations.append(operation)

    overrides = payload.get("overrides")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("overrides must be an object")
    operations = _apply_operation_overrides(
        operations, overrides if isinstance(overrides, Mapping) else None
    )

    if include is not None:
        operations = [
            op for op in operations if _include_matches(op.method, op.path, include)
        ]

    if not operations:
        raise ValueError("No HTTP operations loaded from OpenAPI sources")
    names = [op.name for op in operations]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate HTTP tool names after OpenAPI load: {names}")
    return operations


