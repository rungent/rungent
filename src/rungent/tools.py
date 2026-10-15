import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import UnionType
from typing import Annotated, Any, Union, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .state import ApprovalImpact, Identity, ToolResult, TrustedInteractionResponse

logger = logging.getLogger(__name__)


def _union_members(annotation: Any) -> list[Any]:
    origin = get_origin(annotation)
    if origin not in {UnionType, Union}:
        return [annotation]
    return [item for item in get_args(annotation) if item is not type(None)]


def _json_container(value: str) -> Any:
    text = value.strip()
    if len(text) < 2 or text[0] not in "[{":
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, (list, dict)) else None


def _matches(annotation: Any, value: Any) -> bool:
    origin = get_origin(annotation)
    if annotation is Any:
        return True
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is str:
        return isinstance(value, str)
    if origin is list or annotation is list:
        return isinstance(value, list)
    if origin is dict or annotation is dict:
        return isinstance(value, dict)
    return False


def coerce_value(annotation: Any, value: Any) -> Any:
    """Turn model-emitted JSON strings into the declared tool argument type."""
    if value is None:
        return None
    members = _union_members(annotation)
    if len(members) != 1:
        if isinstance(value, str):
            parsed = _json_container(value)
            if parsed is not None:
                for member in members:
                    if _matches(member, parsed):
                        return coerce_value(member, parsed)
        for member in members:
            if member is str:
                continue
            coerced = coerce_value(member, value)
            if coerced is not value and _matches(member, coerced):
                return coerced
        return value
    target = members[0]
    origin = get_origin(target)
    if origin is list or target is list:
        if isinstance(value, str):
            parsed = _json_container(value)
            value = parsed if isinstance(parsed, list) else [value]
        item_type = get_args(target)[0] if get_args(target) else Any
        if isinstance(value, list):
            return [coerce_value(item_type, item) for item in value]
        return value
    if origin is dict or target is dict:
        if isinstance(value, str):
            parsed = _json_container(value)
            if isinstance(parsed, dict):
                return parsed
        return value
    if target is int and isinstance(value, str):
        text = value.strip()
        if text and (text.isdigit() or (text[0] == "-" and text[1:].isdigit())):
            return int(text)
        return value
    if target is float and isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            return value
        return number
    if target is bool and isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return value


def coerce_arguments(model: type[BaseModel], arguments: Mapping[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for name, value in arguments.items():
        declared = model.model_fields.get(name)
        if declared is None:
            coerced[name] = value
            continue
        coerced[name] = coerce_value(declared.annotation, value)
    return coerced


@dataclass(frozen=True, slots=True)
class ToolFollow:
    """Run another tool after this one succeeds, using fields from its result."""

    tool: str
    arguments: dict[str, str]
    when: dict[str, Any]


def parse_tool_follow(value: Any) -> ToolFollow | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return ToolFollow(tool=value.strip(), arguments={}, when={})
    if not isinstance(value, Mapping):
        raise ValueError("tool follow must name the next tool")
    tool_name = str(value.get("tool") or "").strip()
    if not tool_name:
        raise ValueError("tool follow requires tool")
    raw_arguments = value.get("arguments") or {}
    raw_when = value.get("when") or {}
    if not isinstance(raw_arguments, Mapping) or not isinstance(raw_when, Mapping):
        raise ValueError("tool follow arguments and when must be objects")
    return ToolFollow(
        tool=tool_name,
        arguments={str(key): str(source) for key, source in raw_arguments.items()},
        when={str(key): expected for key, expected in raw_when.items()},
    )


class ToolEffect(StrEnum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    EXTERNAL = "external"


class ApprovalPolicy(StrEnum):
    NEVER = "never"
    ALWAYS = "always"


@dataclass(slots=True)
class ToolContext:
    identity: Identity
    session_id: str
    run_id: str
    current_input: str = ""
    resource: Mapping[str, Any] = field(default_factory=dict)
    deps: Mapping[str, Any] = field(default_factory=dict)
    interaction_response: TrustedInteractionResponse | None = None
    _progress_reporter: Callable[[str, Any], Awaitable[None]] | None = None

    async def report_progress(self, message: str, *, public: Any = None) -> None:
        """Emit safe progress from inside a long-running host tool."""
        message = message.strip()
        if not message or len(message) > 280:
            raise ValueError("Progress must be a non-empty string of at most 280 characters")
        if self._progress_reporter is not None:
            await self._progress_reporter(message, public)


ToolFunction = Callable[..., Awaitable[Any]]
ConfirmationFunction = Callable[..., str | ApprovalImpact | Awaitable[str | ApprovalImpact]]
ApprovalReady = Callable[..., bool | Awaitable[bool]]


def _schema_annotation(annotation: Any) -> Any:
    if get_origin(annotation) is not Annotated:
        return annotation
    base, *metadata = get_args(annotation)
    normalized = [Field(description=item) if isinstance(item, str) else item for item in metadata]
    return cast(Any, Annotated).__class_getitem__((base, *normalized))


def _input_model(fn: ToolFunction, name: str) -> type[BaseModel]:
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"Tool {name} must be an async function")
    signature = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    parameters = list(signature.parameters.items())
    if not parameters:
        raise TypeError(f"Tool {name} must accept ToolContext as its first parameter")
    context_name, context_parameter = parameters[0]
    context_type = hints.get(context_name, context_parameter.annotation)
    if context_type is not ToolContext:
        raise TypeError(f"Tool {name} first parameter must be annotated as ToolContext")
    fields: dict[str, tuple[Any, Any]] = {}
    for index, (parameter_name, parameter) in enumerate(parameters):
        if index == 0:
            continue
        annotation = _schema_annotation(hints.get(parameter_name, parameter.annotation))
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Tool parameter {name}.{parameter_name} must have a type annotation")
        default = parameter.default
        if default is inspect.Parameter.empty:
            default = ...
        fields[parameter_name] = (annotation, default)
    return cast(
        type[BaseModel],
        create_model(
            f"{''.join(part.title() for part in name.split('_'))}Input",
            __config__=ConfigDict(extra="forbid"),
            **cast(Any, fields),
        ),
    )


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    function: ToolFunction
    input_model: type[BaseModel]
    effect: ToolEffect
    approval: ApprovalPolicy
    confirmation: str | ConfirmationFunction | None
    approval_ready: ApprovalReady | None
    title: str
    timeout_seconds: float
    parallel: bool
    deduplicate: bool
    requires_interaction_response: bool
    domain: str | None = None
    family: str | None = None
    follow: ToolFollow | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        arguments = coerce_arguments(self.input_model, arguments)
        allowed = set(self.input_model.model_fields)
        unknown = sorted(key for key in arguments if key not in allowed)
        if unknown:
            logger.warning("tool %s dropping unknown arguments: %s", self.name, unknown)
            arguments = {key: value for key, value in arguments.items() if key in allowed}
        value = self.input_model.model_validate(arguments)
        return {name: getattr(value, name) for name in self.input_model.model_fields}

    def normalize(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.input_model.model_validate(self.validate(arguments)).model_dump(mode="json")

    async def confirmation_prompt(
        self,
        ctx: ToolContext,
        arguments: dict[str, Any],
    ) -> str:
        impact = await self.approval_impact(ctx, arguments)
        return impact.as_prompt() if impact is not None else ""

    async def approval_impact(
        self,
        ctx: ToolContext,
        arguments: dict[str, Any],
    ) -> ApprovalImpact | None:
        if self.confirmation is None:
            raise RuntimeError(f"Tool {self.name} has no approval confirmation")
        if isinstance(self.confirmation, str):
            prompt = self.confirmation.format_map(arguments).strip()
            if not prompt:
                raise ValueError(f"Tool {self.name} produced an empty approval confirmation")
            if len(prompt) > 2000:
                raise ValueError(f"Tool {self.name} approval confirmation exceeds 2000 characters")
            return ApprovalImpact(
                title=prompt,
                target_label=prompt,
                effect=prompt,
            )
        result = self.confirmation(ctx, **self.validate(arguments))
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ApprovalImpact):
            prompt = result.as_prompt()
            if not prompt.strip():
                raise ValueError(f"Tool {self.name} produced an empty approval confirmation")
            if len(prompt) > 2000:
                raise ValueError(f"Tool {self.name} approval confirmation exceeds 2000 characters")
            return result
        if not isinstance(result, str) or not result.strip():
            raise ValueError(f"Tool {self.name} produced an empty approval confirmation")
        prompt = result.strip()
        if len(prompt) > 2000:
            raise ValueError(f"Tool {self.name} approval confirmation exceeds 2000 characters")
        return ApprovalImpact(title=prompt, target_label=prompt, effect=prompt)

    async def __call__(self, ctx: ToolContext, **arguments: Any) -> ToolResult:
        validated = self.validate(arguments)
        result = await self.function(ctx, **validated)
        if isinstance(result, ToolResult):
            return result
        return ToolResult(data=result)


def tool(
    *,
    effect: ToolEffect | str,
    approval: ApprovalPolicy | str,
    name: str | None = None,
    description: str | None = None,
    title: str | None = None,
    confirmation: str | ConfirmationFunction | None = None,
    approval_ready: ApprovalReady | None = None,
    timeout_seconds: float = 60,
    parallel: bool = False,
    deduplicate: bool = True,
    requires_interaction_response: bool = False,
    domain: str | None = None,
    family: str | None = None,
    follow: str | Mapping[str, Any] | None = None,
) -> Callable[[ToolFunction], Tool]:
    def decorate(fn: ToolFunction) -> Tool:
        tool_name = name or fn.__name__
        tool_description = description or inspect.getdoc(fn)
        if not tool_description:
            raise TypeError(f"Tool {tool_name} needs a docstring or description")
        approval_policy = ApprovalPolicy(approval)
        if approval_policy is ApprovalPolicy.ALWAYS and confirmation is None:
            raise TypeError(f"Tool {tool_name} with approval='always' needs a confirmation")
        if approval_policy is ApprovalPolicy.NEVER and confirmation is not None:
            raise TypeError(f"Tool {tool_name} with approval='never' cannot declare a confirmation")
        return Tool(
            name=tool_name,
            description=tool_description,
            function=fn,
            input_model=_input_model(fn, tool_name),
            effect=ToolEffect(effect),
            approval=approval_policy,
            confirmation=confirmation,
            approval_ready=approval_ready,
            title=title or tool_name.replace("_", " ").title(),
            timeout_seconds=timeout_seconds,
            parallel=parallel,
            deduplicate=deduplicate,
            requires_interaction_response=requires_interaction_response,
            domain=domain,
            family=family,
            follow=parse_tool_follow(follow),
        )

    return decorate


def validation_error_message(exc: ValidationError) -> str:
    return __import__("json").dumps(
        {"error": "invalid_tool_arguments", "details": validation_error_details(exc)},
        ensure_ascii=False,
    )


def validation_error_details(exc: ValidationError) -> list[dict[str, str]]:
    return [
        {"path": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
        for item in exc.errors()
    ]


REQUEST_INPUT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "request_input",
        "description": (
            "Pause and ask the user for missing information. Use form to group independent "
            "questions that are already known; ask dependent questions sequentially. Use choice "
            "when a short list of valid answers exists; set multiple=true for multi-select and "
            "allow_custom=true when the user may also enter their own answer. Set allow_skip=true "
            "only when ignoring the question is safe. Never put numbered alternatives inside a "
            "text prompt; the runtime rejects them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["text", "choice", "form"]},
                "prompt": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"},
                            "description": {"type": "string"},
                            "recommended": {"type": "boolean", "default": False},
                        },
                        "required": ["id", "label"],
                        "additionalProperties": False,
                    },
                },
                "multiple": {
                    "type": "boolean",
                    "description": "Allow selecting more than one option. Only valid for choice.",
                    "default": False,
                },
                "allow_custom": {
                    "type": "boolean",
                    "description": (
                        "Allow a free-text custom answer. For multi-select it may accompany "
                        "options."
                    ),
                    "default": False,
                },
                "allow_skip": {
                    "type": "boolean",
                    "description": "Allow the user to explicitly skip this choice.",
                    "default": False,
                },
                "skip_label": {"type": "string"},
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "description": "Independent questions shown and submitted as one form.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "prompt": {"type": "string"},
                            "kind": {"type": "string", "enum": ["text", "choice"]},
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "label": {"type": "string"},
                                        "description": {"type": "string"},
                                        "recommended": {
                                            "type": "boolean",
                                            "default": False,
                                        },
                                    },
                                    "required": ["id", "label"],
                                    "additionalProperties": False,
                                },
                            },
                            "multiple": {"type": "boolean", "default": False},
                            "allow_custom": {"type": "boolean", "default": False},
                            "required": {"type": "boolean", "default": True},
                        },
                        "required": ["id", "prompt", "kind"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["kind", "prompt"],
            "additionalProperties": False,
        },
    },
}


REPORT_PROGRESS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "report_progress",
        "description": (
            "Share one short, user-visible progress update during a multi-step task. Report only "
            "the current plan, action, or finding; never include hidden reasoning, prompts, "
            "secrets, or private tool data. Do not use for simple answers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 280,
                }
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },
}
