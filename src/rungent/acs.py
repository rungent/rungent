from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .state import new_id, now


class Event(BaseModel):
    v: int = 1
    id: str = Field(default_factory=lambda: new_id("evt"))
    seq: int
    type: str
    session_id: str
    run_id: str
    created_at: datetime = Field(default_factory=now)
    data: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class EventEmitter:
    session_id: str
    run_id: str
    sequence: int = 0

    def emit(self, event_type: str, **data: Any) -> Event:
        self.sequence += 1
        return Event(
            seq=self.sequence,
            type=event_type,
            session_id=self.session_id,
            run_id=self.run_id,
            data=data,
        )


def encode_sse(event: Event) -> str:
    return f"data: {json.dumps(event.model_dump(mode='json'), ensure_ascii=False)}\n\n"
