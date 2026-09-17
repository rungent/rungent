"""Redacted tool execution facts for ACS events."""

from typing import Any

from pydantic import BaseModel

_REDACT_KEYS = {
    "access_token",
    "api_key",
    "kubeconfig",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
    "ticket",
    "token",
    "vg_ssh_keys",
}


class ResourceRef(BaseModel):
    resource_type: str
    resource_id: str


class ExecutionError(BaseModel):
    code: str
    message: str
    fields: list[dict[str, str]] | None = None


class ToolExecution(BaseModel):
    arguments: dict[str, Any] | None = None
    targets: list[ResourceRef] | None = None
    created_resources: list[ResourceRef] | None = None
    status: str | None = None
    request_id: str | None = None
    operation_id: str | None = None
    error: ExecutionError | None = None


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in _REDACT_KEYS:
                continue
            redacted[key] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def public_validation_message(details: list[dict[str, str]]) -> str:
    if not details:
        return "参数不合法"
    parts = [
        f"{item['path']}: {item['message']}" if item.get("path") else item["message"]
        for item in details
        if item.get("message")
    ]
    text = "；".join(parts).strip()
    return text[:500] if text else "参数不合法"


def execution_payload(
    *,
    arguments: dict[str, Any] | None = None,
    host: ToolExecution | None = None,
    status: str | None = None,
    error: ExecutionError | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if host is not None:
        payload.update(host.model_dump(mode="json", exclude_none=True))
    if arguments is not None:
        payload["arguments"] = redact(arguments)
    if status is not None and "status" not in payload:
        payload["status"] = status
    if error is not None and "error" not in payload:
        payload["error"] = error.model_dump(mode="json")
    elif "error" not in payload:
        payload["error"] = None
    return payload
