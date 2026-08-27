"""Context-window usage estimation and provider calibration."""

import json
from typing import Any

CATEGORY_LABELS: dict[str, str] = {
    "instructions": "Instructions",
    "runtime": "Runtime",
    "context": "Context",
    "tool_definitions": "Tool definitions",
    "conversation": "Conversation",
}

DEFAULT_CONTEXT_BUDGET_TOKENS = 80_000


def approx_text_tokens(text: str) -> int:
    """Heuristic token count tuned for mixed CJK/Latin agent content."""
    if not text:
        return 0
    cjk = 0
    latin = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            cjk += 1
        else:
            latin += 1
    return max(1, cjk + latin // 4)


def approx_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += approx_text_tokens(content)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += approx_text_tokens(json.dumps(tool_calls, ensure_ascii=False))
        total += 4
    return total


def parse_provider_usage(raw: dict[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt_tokens")
    if not isinstance(prompt, int) or prompt < 0:
        return None
    completion = raw.get("completion_tokens")
    total = raw.get("total_tokens")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": int(completion) if isinstance(completion, int) else 0,
        "total_tokens": int(total) if isinstance(total, int) else prompt,
    }


def _percent(used: int, budget: int) -> int:
    if budget <= 0:
        return 0
    return min(100, int(round(used / budget * 100)))


def _category(category_id: str, tokens: int) -> dict[str, Any]:
    return {
        "id": category_id,
        "label": CATEGORY_LABELS[category_id],
        "tokens": tokens,
    }


def estimate_context_usage(
    *,
    instructions: str,
    runtime: str,
    context: str,
    conversation: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    budget: int,
) -> dict[str, Any]:
    tool_tokens = approx_text_tokens(json.dumps(tool_schemas, ensure_ascii=False))
    categories = [
        _category("instructions", approx_text_tokens(instructions)),
        _category("runtime", approx_text_tokens(runtime)),
        _category("context", approx_text_tokens(context)),
        _category("tool_definitions", tool_tokens),
        _category("conversation", approx_tokens(conversation) if conversation else 0),
    ]
    categories = [item for item in categories if item["tokens"] > 0]
    used_tokens = sum(item["tokens"] for item in categories)
    return {
        "budget_tokens": budget,
        "used_tokens": used_tokens,
        "used_percent": _percent(used_tokens, budget),
        "categories": categories,
        "source": "estimated",
    }


def calibrate_context_usage(
    estimated: dict[str, Any],
    *,
    provider_usage: dict[str, Any] | None,
    budget: int,
) -> dict[str, Any]:
    parsed = parse_provider_usage(provider_usage)
    if not parsed:
        return {**estimated, "source": "estimated"}

    prompt_tokens = parsed["prompt_tokens"]
    est_used = int(estimated.get("used_tokens") or 0)
    categories = list(estimated.get("categories") or [])

    if est_used > 0 and categories:
        ratio = prompt_tokens / est_used
        scaled: list[dict[str, Any]] = []
        for cat in categories:
            scaled.append({**cat, "tokens": max(0, int(round(cat["tokens"] * ratio)))})
        drift = prompt_tokens - sum(item["tokens"] for item in scaled)
        if drift and scaled:
            scaled[-1] = {**scaled[-1], "tokens": scaled[-1]["tokens"] + drift}
        categories = scaled

    return {
        "budget_tokens": budget,
        "used_tokens": prompt_tokens,
        "used_percent": _percent(prompt_tokens, budget),
        "categories": categories,
        "source": "provider",
        "estimated_tokens": est_used,
        "prompt_tokens": prompt_tokens,
    }
