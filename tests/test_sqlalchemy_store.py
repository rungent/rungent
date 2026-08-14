import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from rungent.acs import EventEmitter
from rungent.sqlalchemy import RungentBase, SQLAlchemyStore
from rungent.state import Identity, Message, Run, RunStatus, Session


async def test_sqlalchemy_store_round_trip():
    pytest.importorskip("aiosqlite")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(RungentBase.metadata.create_all)
    store = SQLAlchemyStore(async_sessionmaker(engine, expire_on_commit=False))
    session = Session(agent_name="trip", subject_id="user")
    await store.create_session(session)
    await store.append_message(session.id, Message(role="user", content="hello"))
    run = Run(session_id=session.id, status=RunStatus.RUNNING)
    await store.create_run(run)
    emitter = EventEmitter(session.id, run.id)
    event = emitter.emit("run.started", status="running")
    await store.append_event(event)
    active_run = await store.find_active_run(session.id)
    assert active_run is not None
    assert active_run.id == run.id
    run.status = RunStatus.COMPLETED
    await store.save_run(run)

    assert (await store.get_session(session.id)).subject_id == "user"
    assert [item.content for item in await store.list_messages(session.id)] == ["hello"]
    assert (await store.get_run(run.id)).status == RunStatus.COMPLETED
    assert [item.id for item in await store.list_events(run.id)] == [event.id]
    assert [item.id for item in await store.list_runs(session.id)] == [run.id]
    assert await store.find_active_run(session.id) is None
    session.title = "Tokyo"
    await store.save_session(session)
    listed = await store.list_sessions(Identity(subject_id="user"))
    assert [item.id for item in listed] == [session.id]
    assert listed[0].title == "Tokyo"
    assert await store.list_sessions(Identity(subject_id="other")) == []
    await engine.dispose()


async def test_waiting_external_run_is_active():
    pytest.importorskip("aiosqlite")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(RungentBase.metadata.create_all)
    store = SQLAlchemyStore(async_sessionmaker(engine, expire_on_commit=False))
    session = Session(agent_name="agent", subject_id="user")
    await store.create_session(session)
    run = Run(session_id=session.id, status=RunStatus.WAITING_EXTERNAL)
    await store.create_run(run)

    active = await store.find_active_run(session.id)

    assert active is not None
    assert active.id == run.id
    await engine.dispose()


async def test_claim_run_atomically_transitions_expected_status():
    pytest.importorskip("aiosqlite")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(RungentBase.metadata.create_all)
    store = SQLAlchemyStore(async_sessionmaker(engine, expire_on_commit=False))
    session = Session(agent_name="agent", subject_id="user")
    await store.create_session(session)
    run = Run(session_id=session.id, status=RunStatus.WAITING_EXTERNAL)
    await store.create_run(run)

    claimed = await store.claim_run(
        run.id,
        "worker-1",
        30,
        expected_status=RunStatus.WAITING_EXTERNAL,
    )
    duplicate = await store.claim_run(
        run.id,
        "worker-2",
        30,
        expected_status=RunStatus.WAITING_EXTERNAL,
    )

    assert claimed is not None
    assert claimed.status is RunStatus.RUNNING
    assert claimed.lease_owner == "worker-1"
    assert duplicate is None
    await engine.dispose()
