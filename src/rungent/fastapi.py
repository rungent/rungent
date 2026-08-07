from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .acs import encode_sse
from .runtime import ActiveRunConflict, Runtime
from .state import Identity, InteractionResponse, Run, Session

IdentityResolver = Callable[[Request], Identity | Awaitable[Identity]]
ContextFactory = Callable[
    [Request, Session, Run | None], Mapping[str, Any] | Awaitable[Mapping[str, Any]]
]


class CreateSessionRequest(BaseModel):
    agent: str | None = None
    resource: dict[str, Any] = Field(default_factory=dict)


class CreateRunRequest(BaseModel):
    input: str


class SubmitInteractionRequest(BaseModel):
    interaction_id: str
    value: Any


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def create_router(
    runtime: Runtime,
    *,
    identity_resolver: IdentityResolver,
    context_factory: ContextFactory | None = None,
) -> APIRouter:
    router = APIRouter()

    async def identity(request: Request) -> Identity:
        return await _resolve(identity_resolver(request))

    async def deps(request: Request, session: Session, run: Run | None = None) -> Mapping[str, Any]:
        if context_factory is None:
            return {}
        return await _resolve(context_factory(request, session, run))

    def translate_error(exc: Exception) -> HTTPException:
        if isinstance(exc, PermissionError):
            return HTTPException(status_code=403, detail=str(exc))
        if isinstance(exc, KeyError):
            return HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, ActiveRunConflict):
            return HTTPException(
                status_code=409,
                detail={
                    "code": "active_run_conflict",
                    "run_id": exc.run.id,
                    "status": exc.run.status,
                },
            )
        if isinstance(exc, ValueError):
            return HTTPException(status_code=409, detail=str(exc))
        return HTTPException(status_code=500, detail="Rungent request failed")

    @router.post("/sessions")
    async def create_session(body: CreateSessionRequest, request: Request):
        try:
            session = await runtime.create_session(
                identity=await identity(request),
                agent_name=body.agent,
                resource=body.resource,
            )
            return session.model_dump(mode="json")
        except Exception as exc:
            raise translate_error(exc) from exc

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        try:
            session, messages = await runtime.get_session(
                session_id, identity=await identity(request)
            )
            return {
                **session.model_dump(mode="json"),
                "messages": [message.model_dump(mode="json") for message in messages],
            }
        except Exception as exc:
            raise translate_error(exc) from exc

    @router.get("/sessions/{session_id}/runs")
    async def list_runs(session_id: str, request: Request):
        try:
            runs = await runtime.list_session_runs(session_id, identity=await identity(request))
            return [
                {
                    "id": run.id,
                    "session_id": run.session_id,
                    "status": run.status,
                    "event_seq": run.event_seq,
                    "model_steps": run.model_steps,
                    "created_at": run.created_at,
                    "updated_at": run.updated_at,
                }
                for run in runs
            ]
        except Exception as exc:
            raise translate_error(exc) from exc

    @router.get("/runs/{run_id}/events")
    async def list_events(
        run_id: str,
        request: Request,
        after_seq: int = Query(default=0, ge=0),
    ):
        try:
            events = await runtime.get_run_events(
                run_id,
                identity=await identity(request),
                after_seq=after_seq,
            )
            return [event.model_dump(mode="json") for event in events]
        except Exception as exc:
            raise translate_error(exc) from exc

    @router.get("/runs/{run_id}/events/stream")
    async def stream_events(
        run_id: str,
        request: Request,
        after_seq: int = Query(default=0, ge=0),
    ):
        try:
            resolved_identity = await identity(request)
            await runtime.get_run_events(
                run_id,
                identity=resolved_identity,
                after_seq=after_seq,
            )
        except Exception as exc:
            raise translate_error(exc) from exc

        async def stream():
            async for event in runtime.stream_events(
                run_id,
                identity=resolved_identity,
                after_seq=after_seq,
            ):
                yield encode_sse(event)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, request: Request):
        try:
            run = await runtime.cancel_run(run_id, identity=await identity(request))
            return {"id": run.id, "status": run.status}
        except Exception as exc:
            raise translate_error(exc) from exc

    @router.post("/sessions/{session_id}/runs")
    async def create_run(session_id: str, body: CreateRunRequest, request: Request):
        try:
            resolved_identity = await identity(request)
            session, _ = await runtime.get_session(session_id, identity=resolved_identity)
            if not body.input.strip():
                raise ValueError("Run input cannot be empty")
            resolved_deps = await deps(request, session)
            run = await runtime.create_run(
                session_id=session_id,
                content=body.input,
                identity=resolved_identity,
                idempotency_key=request.headers.get("Idempotency-Key"),
                deps=resolved_deps,
            )
        except Exception as exc:
            raise translate_error(exc) from exc
        return JSONResponse(
            status_code=202,
            content={"run_id": run.id, "status": run.status},
        )

    @router.post("/runs/{run_id}/responses")
    async def submit_response(run_id: str, body: SubmitInteractionRequest, request: Request):
        try:
            resolved_identity = await identity(request)
            run = await runtime.store.get_run(run_id)
            session, _ = await runtime.get_session(run.session_id, identity=resolved_identity)
            if run.status != "waiting_input" or run.pending_call is None:
                raise ValueError("Run is not waiting for input")
            if run.pending_call.interaction.id != body.interaction_id:
                raise ValueError("Interaction does not belong to this run")
            resolved_deps = await deps(request, session, run)
        except Exception as exc:
            raise translate_error(exc) from exc

        try:
            submitted = await runtime.submit_response(
                run_id=run_id,
                response=InteractionResponse(
                    interaction_id=body.interaction_id,
                    value=body.value,
                ),
                identity=resolved_identity,
                deps=resolved_deps,
            )
        except Exception as exc:
            raise translate_error(exc) from exc
        return JSONResponse(
            status_code=202,
            content={"run_id": submitted.id, "status": "running"},
        )

    return router
