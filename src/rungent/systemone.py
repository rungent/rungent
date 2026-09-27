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
MODE_STICKY = "sticky"

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
    confidence_threshold: float = 0.6
    max_tools: int = 8

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
            confidence_threshold=float(os.environ.get(f"{prefix}CONFIDENCE_THRESHOLD") or "0.6"),
            max_tools=int(os.environ.get(f"{prefix}MAX_TOOLS") or "8"),
        )


@dataclass(frozen=True, slots=True)
class ShortlistResult:
    """Harness shortlist for one model step. Never means 'full catalog'."""

    mode: str
    include: set[str] = field(default_factory=set)
    noul: float | None = None
    choice: str | None = None
    confidence: float | None = None


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


def _cap_names(names: list[str], *, limit: int) -> set[str]:
    if limit <= 0:
        return set()
    return set(names[:limit])


def _domain_cap(tools: list[Tool], *, domain: str, limit: int) -> set[str]:
    ordered = [tool.name for tool in tools if (tool.domain or "general") == domain]
    return _cap_names(ordered, limit=limit)


def _family_of(tool: Tool | None) -> str | None:
    if tool is None:
        return None
    family = getattr(tool, "family", None)
    if family is None:
        return None
    text = str(family).strip()
    return text or None


def _family_expand(tools: list[Tool], seeds: list[str] | set[str]) -> set[str]:
    by_name = {tool.name: tool for tool in tools}
    families = {_family_of(by_name.get(name)) for name in seeds}
    families.discard(None)
    if not families:
        return set()
    return {
        tool.name
        for tool in tools
        if _family_of(tool) in families
    }


def _expand_act_include(
    tools: list[Tool],
    *,
    winner: str,
    max_tools: int,
) -> set[str]:
    by_name = {tool.name: tool for tool in tools}
    winner_tool = by_name.get(winner)
    ordered: list[str] = [winner]

    family = _family_expand(tools, [winner])
    for name in sorted(family):
        if name not in ordered:
            ordered.append(name)

    if winner_tool is None:
        return _cap_names(ordered, limit=max_tools)

    winner_tokens = set(winner.split("_"))
    domain = winner_tool.domain or "general"
    siblings = [
        tool
        for tool in tools
        if tool.name not in ordered and (tool.domain or "general") == domain
    ]

    def _affinity(tool: Tool) -> tuple[int, int]:
        shared = len(winner_tokens & set(tool.name.split("_")))
        return (-shared, siblings.index(tool))

    for tool in sorted(siblings, key=_affinity):
        if len(ordered) >= max_tools:
            break
        ordered.append(tool.name)
    return _cap_names(ordered, limit=max_tools)


def _sticky_include(
    tools: list[Tool],
    *,
    previous: set[str] | None,
    recent_tool_names: list[str],
    winner: str | None,
    max_tools: int,
) -> set[str]:
    if previous:
        alive = {name for name in previous if any(tool.name == name for tool in tools)}
        if alive:
            return _cap_names(sorted(alive), limit=max_tools)

    seeds = list(recent_tool_names)
    if winner:
        seeds.append(winner)
    family = _family_expand(tools, seeds)
    if family:
        return _cap_names(sorted(family), limit=max_tools)

    if winner:
        by_name = {tool.name: tool for tool in tools}
        tool = by_name.get(winner)
        if tool is not None:
            return _domain_cap(tools, domain=tool.domain or "general", limit=max_tools)

    for name in reversed(recent_tool_names):
        by_name = {tool.name: tool for tool in tools}
        tool = by_name.get(name)
        if tool is not None:
            return _domain_cap(tools, domain=tool.domain or "general", limit=max_tools)

    return _cap_names([tool.name for tool in tools], limit=max_tools)


def _noul_value(answers: dict[str, Any], key: str) -> float:
    raw = answers.get(key) or {}
    if isinstance(raw, dict):
        return float(raw.get("noul") or 0)
    return 0.0


def _gate_noul(answers: dict[str, Any]) -> float:
    needs = _noul_value(answers, "needs_tool")
    acts = _noul_value(answers, "acts_on_resources")
    prose = _noul_value(answers, "prose_suffices")
    # prose_suffices is inverted: high prose → lower tool need.
    return (needs + acts + (1.0 - prose)) / 3.0


async def shortlist_tools(
    client: SystemOneClient,
    *,
    tools: list[Tool],
    user_input: str,
    recent_tool_names: list[str],
    focus_summary: str = "",
    pending: str = "",
    previous_include: set[str] | None = None,
) -> ShortlistResult:
    """Return a harness shortlist for this model step.

    Aligns with the Jev agent-harness contract:
    - low gate noul → chat (no business tools)
    - high noul + confident choice → act (winner + family/domain expand)
    - failure / low confidence → sticky (previous or capped family), never full catalog
    """
    if not tools:
        return ShortlistResult(mode=MODE_CHAT, include=set(), noul=None)

    tool_criteria = {
        item.name: (item.description or item.title or item.name)[:200] for item in tools
    }
    state = {
        "turn": user_input[:2000],
        "recent_tools": recent_tool_names[-6:],
        "focus": focus_summary[:500],
        "pending": pending[:500],
    }
    questions: dict[str, Any] = {
        "tool": {
            "type": "choice",
            "instructions": "Which API or tool best completes `turn`?",
            "criteria": tool_criteria,
        },
        **_NOUL_QUESTIONS,
    }
    try:
        answers = await client.evaluate(state=state, questions=questions)
    except Exception:
        include = _sticky_include(
            tools,
            previous=previous_include,
            recent_tool_names=recent_tool_names,
            winner=None,
            max_tools=client.settings.max_tools,
        )
        logger.exception(
            "System One shortlist failed; mode=%s include_size=%d",
            MODE_STICKY,
            len(include),
        )
        return ShortlistResult(mode=MODE_STICKY, include=include)

    noul = _gate_noul(answers)
    pick = answers.get("tool") or {}
    winner = str(pick.get("choice") or "") if isinstance(pick, dict) else ""
    confidence = float(pick.get("confidence") or 0) if isinstance(pick, dict) else 0.0

    if noul < client.settings.needs_tool_threshold:
        logger.info(
            "System One mode=%s noul=%.3f choice=%r confidence=%.3f include_size=0",
            MODE_CHAT,
            noul,
            winner,
            confidence,
        )
        return ShortlistResult(
            mode=MODE_CHAT,
            include=set(),
            noul=noul,
            choice=winner or None,
            confidence=confidence,
        )

    if confidence >= client.settings.confidence_threshold and winner:
        include = _expand_act_include(
            tools, winner=winner, max_tools=client.settings.max_tools
        )
        logger.info(
            "System One mode=%s noul=%.3f choice=%r confidence=%.3f include_size=%d",
            MODE_ACT,
            noul,
            winner,
            confidence,
            len(include),
        )
        return ShortlistResult(
            mode=MODE_ACT,
            include=include,
            noul=noul,
            choice=winner,
            confidence=confidence,
        )

    include = _sticky_include(
        tools,
        previous=previous_include,
        recent_tool_names=recent_tool_names,
        winner=winner or None,
        max_tools=client.settings.max_tools,
    )
    logger.info(
        "System One mode=%s noul=%.3f choice=%r confidence=%.3f include_size=%d",
        MODE_STICKY,
        noul,
        winner,
        confidence,
        len(include),
    )
    return ShortlistResult(
        mode=MODE_STICKY,
        include=include,
        noul=noul,
        choice=winner or None,
        confidence=confidence,
    )


def is_builtin_tool_name(name: str) -> bool:
    return name in _BUILTIN
