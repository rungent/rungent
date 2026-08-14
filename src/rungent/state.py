"""Persistent runtime state shared by the harness, stores, and transports."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def now() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    WAITING_EXTERNAL = "waiting_external"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Identity(BaseModel):
    subject_id: str
    tenant_id: str | None = None


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    id: str = Field(default_factory=lambda: new_id("msg"))
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    created_at: datetime = Field(default_factory=now)

    def model_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": __import__("json").dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
        if self.name:
            message["name"] = self.name
        return message


class Session(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ses"))
    agent_name: str
    subject_id: str
    tenant_id: str | None = None
    title: str | None = None
    resource: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)

    def model_post_init(self, __context: Any) -> None:
        if self.title is None:
            return
        title = self.title.strip()
        if not title or len(title) > 200:
            raise ValueError("Session title must contain 1 to 200 characters")
        self.title = title


class InteractionOption(BaseModel):
    id: str
    label: str
    description: str | None = None
    recommended: bool = False


class InteractionQuestion(BaseModel):
    id: str
    prompt: str
    kind: Literal["text", "choice"]
    options: list[InteractionOption] = Field(default_factory=list)
    multiple: bool = False
    allow_custom: bool = False
    required: bool = True


class Interaction(BaseModel):
    id: str = Field(default_factory=lambda: new_id("int"))
    kind: Literal["text", "choice", "form", "approval"]
    prompt: str
    options: list[InteractionOption] = Field(default_factory=list)
    questions: list[InteractionQuestion] = Field(default_factory=list)
    multiple: bool = False
    allow_custom: bool = False
    allow_skip: bool = False
    skip_label: str | None = None
    tool_call_id: str
    resolved: bool = False


class ToolContinuation(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class InteractionRequest(BaseModel):
    kind: Literal["text", "choice", "form"]
    prompt: str
    options: list[InteractionOption] = Field(default_factory=list)
    questions: list[InteractionQuestion] = Field(default_factory=list)
    multiple: bool = False
    allow_custom: bool = False
    allow_skip: bool = False
    skip_label: str | None = None
    continuation: ToolContinuation


class DeferredRequest(BaseModel):
    """A durable host-owned task that must finish before the model continues."""

    task_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=280)
    public: Any = None


class PendingCall(BaseModel):
    kind: Literal["model", "approval", "continuation"]
    call: ToolCall
    interaction: Interaction


class PendingExternal(BaseModel):
    call: ToolCall
    task: DeferredRequest


class Run(BaseModel):
    id: str = Field(default_factory=lambda: new_id("run"))
    session_id: str
    status: RunStatus = RunStatus.QUEUED
    input: str = ""
    event_seq: int = 0
    model_steps: int = 0
    idempotency_key: str | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    pending_call: PendingCall | None = None
    pending_external: PendingExternal | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class InteractionResponse(BaseModel):
    interaction_id: str
    value: Any


class TrustedInteractionResponse(BaseModel):
    """A normalized response accepted by the runtime for the current interaction."""

    interaction_id: str
    kind: Literal["text", "choice", "form"]
    prompt: str
    value: Any


class ToolResult(BaseModel):
    data: Any = None
    message: str | None = None
    public: Any = None
    interaction: InteractionRequest | None = None
    deferred: DeferredRequest | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.interaction is not None and self.deferred is not None:
            raise ValueError("ToolResult cannot request interaction and deferred work together")

    @property
    def succeeded(self) -> bool:
        return not (isinstance(self.data, dict) and self.data.get("ok") is False)

    def for_model(self) -> str:
        import json

        payload: dict[str, Any] = {"tool_status": "success" if self.succeeded else "error"}
        if self.data is not None:
            payload["data"] = self.data
        if self.message:
            payload["message"] = self.message
        return json.dumps(payload, ensure_ascii=False, default=str)


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool = False

    def for_model(self) -> str:
        import json

        return json.dumps(
            {"tool_status": "error", **self.model_dump()},
            ensure_ascii=False,
        )
