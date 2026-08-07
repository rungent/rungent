import json

import httpx
import pytest

from rungent.llm import ModelCompleted, ModelRetrying, OpenAICompatibleModel, TextDelta
from rungent.state import ToolCall


async def collect(model: OpenAICompatibleModel):
    return [event async for event in model.stream(messages=[], tools=[])]


async def test_transport_retries_network_failures_and_reports_progress():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleModel(
            api_key="test",
            base_url="https://example.test/v1",
            model="test",
            max_retries=3,
            retry_backoff_seconds=0,
            client=client,
        )
        events = await collect(model)

    assert attempts == 3
    assert events == [
        ModelRetrying(retry=1, max_retries=3, delay_seconds=0, reason="network"),
        ModelRetrying(retry=2, max_retries=3, delay_seconds=0, reason="network"),
        TextDelta("ok"),
        ModelCompleted(text="ok"),
    ]


async def test_transport_does_not_retry_non_retryable_client_errors():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleModel(
            api_key="test",
            base_url="https://example.test/v1",
            model="test",
            max_retries=3,
            retry_backoff_seconds=0,
            client=client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await collect(model)

    assert attempts == 1


async def test_transport_stops_after_configured_retries():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleModel(
            api_key="test",
            base_url="https://example.test/v1",
            model="test",
            max_retries=3,
            retry_backoff_seconds=0,
            client=client,
        )
        with pytest.raises(httpx.ReadTimeout):
            await collect(model)

    assert attempts == 4


async def test_transport_retries_malformed_tool_arguments_with_compactness_hint():
    attempts = 0
    requests: list[dict] = []

    def response(arguments: str) -> httpx.Response:
        chunk = {
            "choices": [
                {
                    "delta": {
                        "content": "working",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "initialize", "arguments": arguments},
                            }
                        ],
                    }
                }
            ]
        }
        return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        requests.append(json.loads(request.content))
        if attempts == 1:
            return response('{"candidate":')
        return response('{"candidate": {}}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleModel(
            api_key="test",
            base_url="https://example.test/v1",
            model="test",
            max_retries=2,
            retry_backoff_seconds=0,
            client=client,
        )
        events = await collect(model)

    assert attempts == 2
    assert events == [
        TextDelta("working"),
        ModelRetrying(
            retry=1,
            max_retries=2,
            delay_seconds=0,
            reason="invalid_tool_arguments",
            tool_name="initialize",
            arguments_chars=13,
            error_kind="JSONDecodeError",
            error_position=13,
        ),
        TextDelta("working"),
        ModelCompleted(
            text="working",
            tool_calls=[ToolCall(id="call-1", name="initialize", arguments={"candidate": {}})],
        ),
    ]
    assert "compact, valid JSON" in requests[1]["messages"][-1]["content"]


async def test_transport_adds_trusted_provider_specific_request_fields():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, text="data: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleModel(
            api_key="test",
            base_url="https://example.test/v1",
            model="test",
            extra_body={"enable_thinking": False},
            client=client,
        )
        await collect(model)

    assert captured["enable_thinking"] is False


def test_transport_rejects_extra_body_overriding_core_fields():
    with pytest.raises(ValueError, match="cannot override"):
        OpenAICompatibleModel(
            api_key="test",
            base_url="https://example.test/v1",
            model="test",
            extra_body={"messages": []},
        )
