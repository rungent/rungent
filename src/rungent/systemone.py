"""System One (Jev-compatible) decision client for tool shortlisting."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .tools import Tool

logger = logging.getLogger(__name__)

_BUILTIN = frozenset({"request_input", "report_progress"})

MODE_CHAT = "chat"
MODE_ACT = "act"

# Product-neutral Noul questions (skill-suggestion style). Gate uses the mean.
_NOUL_QUESTIONS: dict[str, dict[str, Any]] = {
    "needs_tool": {
        "type": "noul",
        "instructions": "Does `turn` require calling any API or tool, rather than answering in prose alone?",
        "criteria": {
            "true": "The user asks to list, create, change, start/stop, delete, or otherwise query external resources via tools",
            "false": "Greeting, conceptual explanation, or an answer that needs no new tool call",
        },
    },
    "acts_on_resources": {
        "type": "noul",
        "instructions": "Does `turn` intend to inspect or change external resources (not pure explanation)?",
        "criteria": {
            "true": "The user wants resource state, inventory, or a mutating action",
            "false": "The user only wants concepts, opinions, or chatter",
        },
    },
    "prose_suffices": {
        "type": "noul",
        "instructions": "Can `turn` be fully answered with natural language and existing context, with no tool call?",
        "criteria": {
            "true": "No tool is needed; prose or prior context is enough",
            "false": "At least one tool call is required to satisfy the user",
        },
    },
}


@dataclass(frozen=True, slots=True)
class SystemOneSettings:
    """Connection settings for a Jev-compatible System One endpoint (e.g. Laya)."""

    base_url: str
    model: str = "multilingual"
    timeout_seconds: float = 5.0
    api_key: str | None = None
    needs_tool_threshold: float = 0.5

    @classmethod
    def from_env(cls, *, prefix: str = "SYSTEMONE_") -> "SystemOneSettings | None":
        base_url = (os.environ.get(f"{prefix}BASE_URL") or "").strip()
        if not base_url:
            return None
        normalized = base_url.rstrip("/")
        # OpenAI-style ".../v1" bases are common; client posts "/v1/systemone".
        if normalized.endswith("/v1"):
            normalized = normalized[: -len("/v1")].rstrip("/")
        return cls(
            base_url=normalized,
            model=(os.environ.get(f"{prefix}MODEL") or "multilingual").strip(),
            timeout_seconds=float(os.environ.get(f"{prefix}TIMEOUT_SECONDS") or "5"),
            api_key=(os.environ.get(f"{prefix}API_KEY") or "").strip() or None,
            needs_tool_threshold=float(os.environ.get(f"{prefix}NEEDS_TOOL_THRESHOLD") or "0.5"),
        )


@dataclass(frozen=True, slots=True)
class ShortlistResult:
    """Chat hides business tools. Act gives the model the whole catalog.

    The caller's token is the permission check. A gate must not drop a catalog tool.
    """

    mode: str
    include: set[str] = field(default_factory=set)
    noul: float | None = None


class SystemOneClient:
    def __init__(self, settings: SystemOneSettings, *, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._client = client
        self._owns_client = client is None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.settings.api_key:
                headers["Authorization"] = f"Bearer {self.settings.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_seconds,
                headers=headers,
            )
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def evaluate(
        self, *, state: Any, questions: dict[str, Any]
    ) -> dict[str, Any]:
        http = await self._http()
        response = await http.post(
            "/v1/systemone",
            json={"model": self.settings.model, "state": state, "questions": questions},
        )
        response.raise_for_status()
        payload = response.json()
        answers = payload.get("answers") or payload.get("result", {}).get("answers") or {}
        if not isinstance(answers, dict):
            raise ValueError("System One response missing answers")
        return answers


def _catalog_names(tools: list[Tool]) -> set[str]:
    return {tool.name for tool in tools}


def _noul_value(answers: dict[str, Any], key: str) -> float:
    raw = answers.get(key) or {}
    if isinstance(raw, dict):
        return float(raw.get("noul") or 0)
    return 0.0


def _gate_noul(answers: dict[str, Any]) -> float:
    needs = _noul_value(answers, "needs_tool")
    acts = _noul_value(answers, "acts_on_resources")
    not_prose = 1.0 - _noul_value(answers, "prose_suffices")
    # Strongest affirmative signal: one well-calibrated Noul can open the gate.
    # Mean of all three under-triggers on Laya for list/create Chinese turns.
    return max(needs, acts, not_prose)


async def shortlist_tools(
    client: SystemOneClient,
    *,
    tools: list[Tool],
    user_input: str,
) -> ShortlistResult:
    """Decide chat vs the whole catalog for this step."""
    if not tools:
        return ShortlistResult(mode=MODE_CHAT, include=set(), noul=None)

    catalog = _catalog_names(tools)
    try:
        answers = await client.evaluate(
            state={"turn": user_input[:2000]},
            questions=_NOUL_QUESTIONS,
        )
    except Exception:
        logger.exception(
            "System One gate failed; mode=%s include_size=%d",
            MODE_ACT,
            len(catalog),
        )
        return ShortlistResult(mode=MODE_ACT, include=catalog)

    noul = _gate_noul(answers)
    if noul < client.settings.needs_tool_threshold:
        logger.info("System One mode=%s noul=%.3f include_size=0", MODE_CHAT, noul)
        return ShortlistResult(mode=MODE_CHAT, include=set(), noul=noul)

    logger.info(
        "System One mode=%s noul=%.3f include_size=%d",
        MODE_ACT,
        noul,
        len(catalog),
    )
    return ShortlistResult(mode=MODE_ACT, include=catalog, noul=noul)


def is_builtin_tool_name(name: str) -> bool:
    return name in _BUILTIN
