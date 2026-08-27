from rungent.usage import (
    DEFAULT_CONTEXT_BUDGET_TOKENS,
    approx_text_tokens,
    calibrate_context_usage,
    estimate_context_usage,
    parse_provider_usage,
)


def test_approx_text_tokens_counts_cjk_and_latin():
    assert approx_text_tokens("") == 0
    assert approx_text_tokens("你好") == 2
    assert approx_text_tokens("abcd") == 1


def test_estimate_context_usage_has_baseline_without_conversation():
    usage = estimate_context_usage(
        instructions="You are a console assistant.",
        runtime="Report progress briefly.",
        context="",
        conversation=[],
        tool_schemas=[{"type": "function", "function": {"name": "inspect_vm"}}],
        budget=DEFAULT_CONTEXT_BUDGET_TOKENS,
    )
    assert usage["budget_tokens"] == DEFAULT_CONTEXT_BUDGET_TOKENS
    assert usage["used_tokens"] > 0
    assert usage["source"] == "estimated"
    assert {item["id"] for item in usage["categories"]} == {
        "instructions",
        "runtime",
        "tool_definitions",
    }
    assert all(item["tokens"] > 0 for item in usage["categories"])
    assert usage["used_tokens"] == sum(item["tokens"] for item in usage["categories"])


def test_estimate_includes_context_and_conversation():
    usage = estimate_context_usage(
        instructions="Help",
        runtime="Progress",
        context="Current context:\nuser=alice",
        conversation=[{"role": "user", "content": "列出虚拟机"}],
        tool_schemas=[],
        budget=1000,
    )
    ids = {item["id"] for item in usage["categories"]}
    assert {"instructions", "runtime", "context", "conversation"} <= ids


def test_calibrate_scales_categories_to_prompt_tokens():
    estimated = estimate_context_usage(
        instructions="AAAA",
        runtime="BBBB",
        context="",
        conversation=[],
        tool_schemas=[],
        budget=100,
    )
    calibrated = calibrate_context_usage(
        estimated,
        provider_usage={"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45},
        budget=100,
    )
    assert calibrated["source"] == "provider"
    assert calibrated["used_tokens"] == 40
    assert calibrated["prompt_tokens"] == 40
    assert calibrated["estimated_tokens"] == estimated["used_tokens"]
    assert sum(item["tokens"] for item in calibrated["categories"]) == 40
    assert calibrated["used_percent"] == 40


def test_calibrate_keeps_estimate_without_provider_usage():
    estimated = {"budget_tokens": 10, "used_tokens": 3, "categories": []}
    assert parse_provider_usage({"total_tokens": 9}) is None
    calibrated = calibrate_context_usage(estimated, provider_usage={"total_tokens": 9}, budget=10)
    assert calibrated["source"] == "estimated"
