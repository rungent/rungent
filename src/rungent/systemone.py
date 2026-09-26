"""System One (Jev-compatible) decision client for tool shortlisting."""

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from .tools import Tool

logger = logging.getLogger(__name__)

_BUILTIN = frozenset({"request_input", "report_progress"})


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


async def shortlist_tools(
    client: SystemOneClient,
    *,
    tools: list[Tool],
    user_input: str,
    recent_tool_names: list[str],
    focus_summary: str = "",
) -> set[str] | None:
    """Return business tool names to expose this step, or None to keep the full catalog.

    On transport/API failure returns None (degrade to full allowlist).
    """
    if not tools:
        return set()
    by_domain: dict[str, list[Tool]] = {}
    for tool in tools:
        by_domain.setdefault(tool.domain or "general", []).append(tool)
    domain_criteria = {
        domain: f"{len(items)} operations: "
        + ", ".join(item.name for item in items[:12])
        for domain, items in by_domain.items()
    }
    state = {
        "turn": user_input[:2000],
        "recent_tools": recent_tool_names[-6:],
        "focus": focus_summary[:500],
    }
    questions: dict[str, Any] = {
        "domain": {
            "type": "choice",
            "instructions": "Which product domain best serves `turn`?",
            "criteria": domain_criteria,
        },
        "needs_tool": {
            "type": "noul",
            "instructions": (
                "Does `turn` require calling a console API tool rather than a direct answer?"
            ),
        },
    }
    try:
        answers = await client.evaluate(state=state, questions=questions)
    except Exception:
        logger.exception("System One shortlist failed; using full tool catalog")
        return None

    needs = answers.get("needs_tool") or {}
    noul = float(needs.get("noul") if isinstance(needs, dict) else 0)
    if noul < client.settings.needs_tool_threshold:
        # Prefer full catalog over an empty tool set (model would have zero ops).
        logger.info("System One needs_tool=%.3f below threshold; using full catalog", noul)
        return None

    domain_answer = answers.get("domain") or {}
    domain = str(domain_answer.get("choice") or "")
    confidence = float(domain_answer.get("confidence") or 0)
    if confidence < client.settings.confidence_threshold or domain not in by_domain:
        logger.info(
            "System One domain=%r confidence=%.3f; using full catalog",
            domain,
            confidence,
        )
        return None

    candidates = by_domain[domain]
    tool_criteria = {item.name: (item.description or item.title)[:200] for item in candidates}
    try:
        tool_answers = await client.evaluate(
            state=state,
            questions={
                "tool": {
                    "type": "choice",
                    "instructions": "Which API operation best serves `turn`?",
                    "criteria": tool_criteria,
                }
            },
        )
    except Exception:
        logger.exception("System One tool pick failed; exposing domain tools")
        return {item.name for item in candidates[: client.settings.max_tools]}

    pick = tool_answers.get("tool") or {}
    winner = str(pick.get("choice") or "")
    pick_confidence = float(pick.get("confidence") or 0)
    if pick_confidence < client.settings.confidence_threshold or not winner:
        return {item.name for item in candidates[: client.settings.max_tools]}

    ordered = [winner]
    for item in candidates:
        if item.name != winner and len(ordered) < client.settings.max_tools:
            ordered.append(item.name)
    return set(ordered)


def is_builtin_tool_name(name: str) -> bool:
    return name in _BUILTIN
