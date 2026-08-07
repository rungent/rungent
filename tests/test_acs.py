import json

from rungent.acs import EventEmitter, encode_sse


def test_acs_envelope_is_versioned_and_sequenced():
    emitter = EventEmitter("ses_1", "run_1")
    first = emitter.emit("run.started", status="running")
    second = emitter.emit("message.delta", delta="hello")
    assert (first.seq, second.seq) == (1, 2)
    wire = encode_sse(second)
    payload = json.loads(wire.removeprefix("data: ").strip())
    assert payload == {
        "v": 1,
        "id": second.id,
        "seq": 2,
        "type": "message.delta",
        "session_id": "ses_1",
        "run_id": "run_1",
        "created_at": second.created_at.isoformat().replace("+00:00", "Z"),
        "data": {"delta": "hello"},
    }
