from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from rungent import Agent, Identity, Runtime
from rungent.fastapi import create_router
from rungent.llm import ModelCompleted
from rungent.state import ToolCall
from rungent.store import MemoryStore
from rungent.testing import ScriptedModel


async def test_fastapi_router_exposes_session_and_stream_contract():
    runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Reply")],
        model=ScriptedModel([ModelCompleted(text="Hello")]),
        store=MemoryStore(),
    )

    async def identity(_request: Request) -> Identity:
        return Identity(subject_id="user")

    app = FastAPI()
    app.include_router(create_router(runtime, identity_resolver=identity), prefix="/assistant")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        session = await client.post("/assistant/sessions", json={"resource": {"trip_id": "t1"}})
        assert session.status_code == 200
        session_id = session.json()["id"]
        response = await client.post(
            f"/assistant/sessions/{session_id}/runs",
            json={"input": "Hi"},
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        streamed = await client.get(f"/assistant/runs/{run_id}/events/stream")
        assert '"type": "run.completed"' in streamed.text
        runs = await client.get(f"/assistant/sessions/{session_id}/runs")
        assert runs.json()[0]["id"] == run_id
        assert runs.json()[0]["model_steps"] == 1
        events = await client.get(f"/assistant/runs/{run_id}/events", params={"after_seq": 2})
        assert all(event["seq"] > 2 for event in events.json())
        assert events.json()[-1]["type"] == "run.completed"
        detail = await client.get(f"/assistant/sessions/{session_id}")
        assert [message["role"] for message in detail.json()["messages"]] == [
            "user",
            "assistant",
        ]
        titled = await client.patch(
            f"/assistant/sessions/{session_id}", json={"title": "Tokyo trip"}
        )
        assert titled.json()["title"] == "Tokyo trip"
        listed = await client.get("/assistant/sessions")
        assert [item["id"] for item in listed.json()] == [session_id]
        assert listed.json()[0]["title"] == "Tokyo trip"


async def test_fastapi_cancel_endpoint_persists_terminal_event():
    runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Ask")],
        model=ScriptedModel(
            [
                ModelCompleted(
                    tool_calls=[
                        ToolCall(
                            id="ask",
                            name="request_input",
                            arguments={"kind": "text", "prompt": "Which city?"},
                        )
                    ]
                )
            ]
        ),
        store=MemoryStore(),
    )

    async def identity(_request: Request) -> Identity:
        return Identity(subject_id="user")

    app = FastAPI()
    app.include_router(create_router(runtime, identity_resolver=identity), prefix="/assistant")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        session = (await client.post("/assistant/sessions", json={})).json()
        response = await client.post(
            f"/assistant/sessions/{session['id']}/runs", json={"input": "Plan"}
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        streamed = await client.get(f"/assistant/runs/{run_id}/events/stream")
        assert '"type": "interaction.requested"' in streamed.text
        cancelled = await client.post(f"/assistant/runs/{run_id}/cancel")
        assert cancelled.json()["status"] == "cancelled"
        events = (await client.get(f"/assistant/runs/{run_id}/events")).json()
        assert events[-1]["type"] == "run.cancelled"


async def test_run_creation_is_idempotent_and_conflicts_include_active_run():
    class WaitingModel:
        def __init__(self) -> None:
            self.release = __import__("asyncio").Event()

        async def stream(self, *, messages, tools, model=None):
            await self.release.wait()
            yield ModelCompleted(text="Done")

    model = WaitingModel()
    runtime = Runtime(
        agents=[Agent(name="assistant", instructions="Reply")],
        model=model,
        store=MemoryStore(),
    )

    async def identity(_request: Request) -> Identity:
        return Identity(subject_id="user")

    app = FastAPI()
    app.include_router(create_router(runtime, identity_resolver=identity), prefix="/assistant")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        session = (await client.post("/assistant/sessions", json={})).json()
        path = f"/assistant/sessions/{session['id']}/runs"
        first = await client.post(
            path,
            json={"input": "Hi"},
            headers={"Idempotency-Key": "request-1"},
        )
        duplicate = await client.post(
            path,
            json={"input": "Hi"},
            headers={"Idempotency-Key": "request-1"},
        )
        conflict = await client.post(path, json={"input": "Another request"})

        assert duplicate.json() == first.json()
        assert conflict.status_code == 409
        detail = conflict.json()["detail"]
        assert detail["code"] == "active_run_conflict"
        assert detail["run_id"] == first.json()["run_id"]
        assert detail["status"] in {"queued", "running"}
        await client.post(f"/assistant/runs/{first.json()['run_id']}/cancel")
