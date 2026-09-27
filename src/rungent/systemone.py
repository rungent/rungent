"""System One (Jev-compatible) decision client for tool shortlisting."""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from .tools import Tool

logger = logging.getLogger(__name__)

_BUILTIN = frozenset({"request_input", "report_progress"})
_DRAFT_NAME = re.compile(
    r"^(?P<action>create|prepare|commit|patch|get)_(?P<resource>.+)_draft$"
)

MODE_CHAT = "chat"
MODE_ACT = "act"
MODE_STICKY = "sticky"


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


def _draft_resource(name: str) -> str | None:
    match = _DRAFT_NAME.match(name)
    return match.group("resource") if match else None


def _draft_family(tools: list[Tool], seeds: list[str] | set[str]) -> set[str]:
    resources = {res for name in seeds if (res := _draft_resource(name))}
    if not resources:
        return set()
    return {
        tool.name
        for tool in tools
        if (res := _draft_resource(tool.name)) and res in resources
    }


def _cap_names(names: list[str], *, limit: int) -> set[str]:
    if limit <= 0:
        return set()
    return set(names[:limit])


def _domain_cap(tools: list[Tool], *, domain: str, limit: int) -> set[str]:
    ordered = [tool.name for tool in tools if (tool.domain or "general") == domain]
    return _cap_names(ordered, limit=limit)


def _expand_act_include(
    tools: list[Tool],
    *,
    winner: str,
    max_tools: int,
) -> set[str]:
    by_name = {tool.name: tool for tool in tools}
    if winner not in by_name:
        # Unknown winner: keep draft family from name alone if parseable, else empty act set.
        family = _draft_family(tools, [winner])
        return _cap_names([winner, *sorted(family)], limit=max_tools) if family else set()

    family = _draft_family(tools, [winner])
    ordered: list[str] = [winner]
    for name in sorted(family):
        if name != winner and name not in ordered:
            ordered.append(name)

    winner_tokens = set(winner.split("_"))
    domain = by_name[winner].domain or "general"
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
    family = _draft_family(tools, seeds)
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
    - low ``needs_tool`` noul → chat (no business tools)
    - high noul + confident choice → act (winner + draft family)
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
            "instructions": "哪个 API 操作最能完成 `turn`？",
            "criteria": tool_criteria,
        },
        "needs_tool": {
            "type": "noul",
            "instructions": "`turn` 是否需要调用控制台 API 工具，而不是直接用自然语言回答？",
            "criteria": {
                "true": "用户要求列出、创建、修改、开关机或查询控制台资源，需要新的 API 调用",
                "false": "用户寒暄、询问概念解释，或仅凭已有上下文即可回答",
            },
        },
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

    needs = answers.get("needs_tool") or {}
    noul = float(needs.get("noul") if isinstance(needs, dict) else 0)
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
