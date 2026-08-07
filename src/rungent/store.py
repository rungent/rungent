from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from datetime import timedelta
from typing import Protocol

from .acs import Event
from .state import Message, Run, RunStatus, Session, now


class Store(Protocol):
    async def create_session(self, session: Session) -> None: ...

    async def get_session(self, session_id: str) -> Session: ...

    async def append_message(self, session_id: str, message: Message) -> None: ...

    async def list_messages(self, session_id: str) -> Sequence[Message]: ...

    async def append_event(self, event: Event) -> None: ...

    async def list_events(self, run_id: str, *, after_seq: int = 0) -> Sequence[Event]: ...

    async def create_run(self, run: Run) -> None: ...

    async def get_run(self, run_id: str) -> Run: ...

    async def find_active_run(self, session_id: str) -> Run | None: ...

    async def find_run_by_idempotency_key(
        self, session_id: str, idempotency_key: str
    ) -> Run | None: ...

    async def list_recoverable_runs(self) -> Sequence[Run]: ...

    async def claim_run(
        self,
        run_id: str,
        owner: str,
        lease_seconds: float,
        *,
        expected_status: RunStatus = RunStatus.QUEUED,
    ) -> Run | None: ...

    async def renew_run_lease(self, run_id: str, owner: str, lease_seconds: float) -> bool: ...

    async def release_run_lease(self, run_id: str, owner: str) -> None: ...

    async def list_runs(self, session_id: str) -> Sequence[Run]: ...

    async def save_run(self, run: Run) -> None: ...


class MemoryStore:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.messages: dict[str, list[Message]] = defaultdict(list)
        self.events: dict[str, list[Event]] = defaultdict(list)
        self.runs: dict[str, Run] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, session: Session) -> None:
        async with self._lock:
            if session.id in self.sessions:
                raise ValueError(f"Session already exists: {session.id}")
            self.sessions[session.id] = session.model_copy(deep=True)

    async def get_session(self, session_id: str) -> Session:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session: {session_id}")
        return session.model_copy(deep=True)

    async def append_message(self, session_id: str, message: Message) -> None:
        async with self._lock:
            if session_id not in self.sessions:
                raise KeyError(f"Unknown session: {session_id}")
            self.messages[session_id].append(message.model_copy(deep=True))
            self.sessions[session_id].updated_at = now()

    async def list_messages(self, session_id: str) -> Sequence[Message]:
        return [item.model_copy(deep=True) for item in self.messages.get(session_id, [])]

    async def append_event(self, event: Event) -> None:
        async with self._lock:
            if event.run_id not in self.runs:
                raise KeyError(f"Unknown run: {event.run_id}")
            events = self.events[event.run_id]
            if events and event.seq <= events[-1].seq:
                raise ValueError(f"Event sequence must increase for run {event.run_id}")
            events.append(event.model_copy(deep=True))

    async def list_events(self, run_id: str, *, after_seq: int = 0) -> Sequence[Event]:
        if run_id not in self.runs:
            raise KeyError(f"Unknown run: {run_id}")
        return [
            item.model_copy(deep=True)
            for item in self.events.get(run_id, [])
            if item.seq > after_seq
        ]

    async def create_run(self, run: Run) -> None:
        async with self._lock:
            if run.id in self.runs:
                raise ValueError(f"Run already exists: {run.id}")
            self.runs[run.id] = run.model_copy(deep=True)

    async def get_run(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise KeyError(f"Unknown run: {run_id}")
        return run.model_copy(deep=True)

    async def find_active_run(self, session_id: str) -> Run | None:
        active = {
            RunStatus.QUEUED,
            RunStatus.RUNNING,
            RunStatus.WAITING_INPUT,
            RunStatus.WAITING_EXTERNAL,
        }
        for run in reversed(self.runs.values()):
            if run.session_id == session_id and run.status in active:
                return run.model_copy(deep=True)
        return None

    async def find_run_by_idempotency_key(
        self, session_id: str, idempotency_key: str
    ) -> Run | None:
        for run in reversed(self.runs.values()):
            if run.session_id == session_id and run.idempotency_key == idempotency_key:
                return run.model_copy(deep=True)
        return None

    async def list_recoverable_runs(self) -> Sequence[Run]:
        current = now()
        return [
            run.model_copy(deep=True)
            for run in self.runs.values()
            if run.status is RunStatus.QUEUED
            or (
                run.status is RunStatus.RUNNING
                and (run.lease_expires_at is None or run.lease_expires_at <= current)
            )
        ]

    async def claim_run(
        self,
        run_id: str,
        owner: str,
        lease_seconds: float,
        *,
        expected_status: RunStatus = RunStatus.QUEUED,
    ) -> Run | None:
        async with self._lock:
            run = self.runs.get(run_id)
            if run is None:
                raise KeyError(f"Unknown run: {run_id}")
            if run.status is not expected_status:
                return None
            claimed = run.model_copy(deep=True)
            claimed.status = RunStatus.RUNNING
            claimed.lease_owner = owner
            claimed.lease_expires_at = now() + timedelta(seconds=lease_seconds)
            claimed.updated_at = now()
            self.runs[run_id] = claimed.model_copy(deep=True)
            return claimed

    async def renew_run_lease(self, run_id: str, owner: str, lease_seconds: float) -> bool:
        async with self._lock:
            run = self.runs.get(run_id)
            if run is None or run.lease_owner != owner or run.status is not RunStatus.RUNNING:
                return False
            run.lease_expires_at = now() + timedelta(seconds=lease_seconds)
            return True

    async def release_run_lease(self, run_id: str, owner: str) -> None:
        async with self._lock:
            run = self.runs.get(run_id)
            if run is not None and run.lease_owner == owner:
                run.lease_owner = None
                run.lease_expires_at = None

    async def list_runs(self, session_id: str) -> Sequence[Run]:
        return [
            run.model_copy(deep=True) for run in self.runs.values() if run.session_id == session_id
        ]

    async def save_run(self, run: Run) -> None:
        async with self._lock:
            if run.id not in self.runs:
                raise KeyError(f"Unknown run: {run.id}")
            run.updated_at = now()
            self.runs[run.id] = run.model_copy(deep=True)
