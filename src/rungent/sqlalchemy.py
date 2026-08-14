from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, cast

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    or_,
    select,
    update,
)
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .acs import Event
from .state import Identity, Message, Run, RunStatus, Session, now


class RungentBase(DeclarativeBase):
    pass


class SessionRow(RungentBase):
    __tablename__ = "rungent_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(256), index=True, nullable=False)
    tenant_id: Mapped[str | None] = mapped_column(String(256), index=True)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    resource: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MessageRow(RungentBase):
    __tablename__ = "rungent_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("rungent_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RunRow(RungentBase):
    __tablename__ = "rungent_runs"
    __table_args__ = (UniqueConstraint("session_id", "idempotency_key"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("rungent_sessions.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EventRow(RungentBase):
    __tablename__ = "rungent_events"
    __table_args__ = (UniqueConstraint("run_id", "seq"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("rungent_runs.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SQLAlchemyStore:
    """Persistent Store using application-owned SQLAlchemy async sessions.

    Add ``RungentBase.metadata`` to the application's Alembic target metadata. Rungent never calls
    ``create_all`` in production.
    """

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def create_session(self, session: Session) -> None:
        async with self.sessions.begin() as db:
            db.add(SessionRow(**session.model_dump()))

    async def get_session(self, session_id: str) -> Session:
        async with self.sessions() as db:
            row = await db.get(SessionRow, session_id)
            if row is None:
                raise KeyError(f"Unknown session: {session_id}")
            return self._session_from_row(row)

    async def list_sessions(self, identity: Identity) -> Sequence[Session]:
        async with self.sessions() as db:
            rows = (
                await db.execute(
                    select(SessionRow)
                    .where(
                        SessionRow.subject_id == identity.subject_id,
                        SessionRow.tenant_id == identity.tenant_id,
                    )
                    .order_by(SessionRow.updated_at.desc(), SessionRow.id.desc())
                )
            ).scalars()
            return [self._session_from_row(row) for row in rows]

    async def save_session(self, session: Session) -> None:
        async with self.sessions.begin() as db:
            row = await db.get(SessionRow, session.id)
            if row is None:
                raise KeyError(f"Unknown session: {session.id}")
            session.updated_at = now()
            row.agent_name = session.agent_name
            row.subject_id = session.subject_id
            row.tenant_id = session.tenant_id
            row.title = session.title
            row.resource = session.resource
            row.updated_at = session.updated_at

    @staticmethod
    def _session_from_row(row: SessionRow) -> Session:
        return Session(
            id=row.id,
            agent_name=row.agent_name,
            subject_id=row.subject_id,
            tenant_id=row.tenant_id,
            title=row.title,
            resource=row.resource,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def append_message(self, session_id: str, message: Message) -> None:
        payload = message.model_dump(mode="json")
        async with self.sessions.begin() as db:
            session = await db.get(SessionRow, session_id)
            if session is None:
                raise KeyError(f"Unknown session: {session_id}")
            session.updated_at = message.created_at
            db.add(
                MessageRow(
                    id=message.id,
                    session_id=session_id,
                    role=message.role,
                    content=message.content,
                    payload=payload,
                    created_at=message.created_at,
                )
            )

    async def list_messages(self, session_id: str) -> Sequence[Message]:
        async with self.sessions() as db:
            rows = (
                await db.execute(
                    select(MessageRow)
                    .where(MessageRow.session_id == session_id)
                    .order_by(MessageRow.created_at, MessageRow.id)
                )
            ).scalars()
            return [Message.model_validate(row.payload) for row in rows]

    async def append_event(self, event: Event) -> None:
        async with self.sessions.begin() as db:
            if await db.get(RunRow, event.run_id) is None:
                raise KeyError(f"Unknown run: {event.run_id}")
            db.add(
                EventRow(
                    id=event.id,
                    run_id=event.run_id,
                    seq=event.seq,
                    type=event.type,
                    payload=event.model_dump(mode="json"),
                    created_at=event.created_at,
                )
            )

    async def list_events(self, run_id: str, *, after_seq: int = 0) -> Sequence[Event]:
        async with self.sessions() as db:
            if await db.get(RunRow, run_id) is None:
                raise KeyError(f"Unknown run: {run_id}")
            rows = (
                await db.execute(
                    select(EventRow)
                    .where(EventRow.run_id == run_id, EventRow.seq > after_seq)
                    .order_by(EventRow.seq)
                )
            ).scalars()
            return [Event.model_validate(row.payload) for row in rows]

    async def create_run(self, run: Run) -> None:
        async with self.sessions.begin() as db:
            db.add(
                RunRow(
                    id=run.id,
                    session_id=run.session_id,
                    status=run.status,
                    idempotency_key=run.idempotency_key,
                    lease_owner=run.lease_owner,
                    lease_expires_at=run.lease_expires_at,
                    payload=run.model_dump(mode="json"),
                    created_at=run.created_at,
                    updated_at=run.updated_at,
                )
            )

    async def get_run(self, run_id: str) -> Run:
        async with self.sessions() as db:
            row = await db.get(RunRow, run_id)
            if row is None:
                raise KeyError(f"Unknown run: {run_id}")
            return Run.model_validate(row.payload)

    async def find_active_run(self, session_id: str) -> Run | None:
        active = [
            RunStatus.QUEUED,
            RunStatus.RUNNING,
            RunStatus.WAITING_INPUT,
            RunStatus.WAITING_EXTERNAL,
        ]
        async with self.sessions() as db:
            row = (
                await db.execute(
                    select(RunRow)
                    .where(RunRow.session_id == session_id, RunRow.status.in_(active))
                    .order_by(RunRow.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            return None if row is None else Run.model_validate(row.payload)

    async def find_run_by_idempotency_key(
        self, session_id: str, idempotency_key: str
    ) -> Run | None:
        async with self.sessions() as db:
            row = (
                await db.execute(
                    select(RunRow).where(
                        RunRow.session_id == session_id,
                        RunRow.idempotency_key == idempotency_key,
                    )
                )
            ).scalar_one_or_none()
            return None if row is None else Run.model_validate(row.payload)

    async def list_recoverable_runs(self) -> Sequence[Run]:
        async with self.sessions() as db:
            current = now()
            rows = (
                await db.execute(
                    select(RunRow).where(
                        or_(
                            RunRow.status == RunStatus.QUEUED,
                            (
                                (RunRow.status == RunStatus.RUNNING)
                                & (
                                    (RunRow.lease_expires_at.is_(None))
                                    | (RunRow.lease_expires_at <= current)
                                )
                            ),
                        )
                    )
                )
            ).scalars()
            return [Run.model_validate(row.payload) for row in rows]

    async def claim_run(
        self,
        run_id: str,
        owner: str,
        lease_seconds: float,
        *,
        expected_status: RunStatus = RunStatus.QUEUED,
    ) -> Run | None:
        async with self.sessions.begin() as db:
            row = await db.get(RunRow, run_id)
            if row is None:
                raise KeyError(f"Unknown run: {run_id}")
            run = Run.model_validate(row.payload)
            run.status = RunStatus.RUNNING
            run.lease_owner = owner
            run.lease_expires_at = now() + timedelta(seconds=lease_seconds)
            run.updated_at = now()
            result = cast(
                CursorResult[Any],
                await db.execute(
                    update(RunRow)
                    .where(RunRow.id == run_id, RunRow.status == expected_status)
                    .values(
                        status=run.status,
                        lease_owner=owner,
                        lease_expires_at=run.lease_expires_at,
                        payload=run.model_dump(mode="json"),
                        updated_at=run.updated_at,
                    )
                ),
            )
            return run if result.rowcount == 1 else None

    async def renew_run_lease(self, run_id: str, owner: str, lease_seconds: float) -> bool:
        expires = now() + timedelta(seconds=lease_seconds)
        async with self.sessions.begin() as db:
            row = await db.get(RunRow, run_id)
            if row is None or row.status != RunStatus.RUNNING or row.lease_owner != owner:
                return False
            run = Run.model_validate(row.payload)
            run.lease_expires_at = expires
            row.lease_expires_at = expires
            row.payload = run.model_dump(mode="json")
            return True

    async def release_run_lease(self, run_id: str, owner: str) -> None:
        async with self.sessions.begin() as db:
            row = await db.get(RunRow, run_id)
            if row is None or row.lease_owner != owner:
                return
            run = Run.model_validate(row.payload)
            run.lease_owner = None
            run.lease_expires_at = None
            row.lease_owner = None
            row.lease_expires_at = None
            row.payload = run.model_dump(mode="json")

    async def list_runs(self, session_id: str) -> Sequence[Run]:
        async with self.sessions() as db:
            rows = (
                await db.execute(
                    select(RunRow)
                    .where(RunRow.session_id == session_id)
                    .order_by(RunRow.created_at, RunRow.id)
                )
            ).scalars()
            return [Run.model_validate(row.payload) for row in rows]

    async def save_run(self, run: Run) -> None:
        async with self.sessions.begin() as db:
            row = await db.get(RunRow, run.id)
            if row is None:
                raise KeyError(f"Unknown run: {run.id}")
            row.status = run.status
            row.idempotency_key = run.idempotency_key
            row.lease_owner = run.lease_owner
            row.lease_expires_at = run.lease_expires_at
            row.payload = run.model_dump(mode="json")
            row.updated_at = run.updated_at
