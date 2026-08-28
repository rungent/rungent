"""Export an Agent as a Skill markdown draft for an external coding agent."""

from collections.abc import Sequence
from typing import Any

from .agent import Agent
from .tools import Tool, ToolEffect


def exportable_tools(tools: Sequence[Tool]) -> list[Tool]:
    return [item for item in tools if not item.requires_interaction_response]


def export_skill(
    agent: Agent,
    *,
    title: str | None = None,
    extra: str = "",
    tools: Sequence[Tool] | None = None,
) -> str:
    """Return a SKILL.md draft. Hosts must rewrite product voice and auth steps."""
    selected = exportable_tools(agent.tools if tools is None else tools)
    heading = title or f"{agent.name} tools"
    lines = [
        f"# {heading}",
        "",
        "This draft is generated from a Rungent Agent. It is not a skill router.",
        "Call the remote MCP tools listed below. Do not invent REST paths.",
        "Do not paste first-party session JWTs into the agent.",
        "If a tool is missing, complete the host OAuth login in the MCP client.",
        "",
        "## Tools",
        "",
    ]
    for tool in selected:
        effect = tool.effect.value if isinstance(tool.effect, ToolEffect) else str(tool.effect)
        lines.append(f"### `{tool.name}`")
        lines.append("")
        lines.append(f"{tool.description.strip()}")
        lines.append("")
        lines.append(f"- effect: `{effect}`")
        lines.append(f"- approval: `{tool.approval}`")
        if tool.approval == "always" and isinstance(tool.confirmation, str):
            lines.append(f"- confirmation: {tool.confirmation}")
        schema: dict[str, Any] = tool.input_model.model_json_schema()
        required = schema.get("required") or []
        properties = schema.get("properties") or {}
        if properties:
            lines.append("- arguments:")
            for name, spec in properties.items():
                mark = "required" if name in required else "optional"
                desc = spec.get("description") or spec.get("type") or ""
                lines.append(f"  - `{name}` ({mark}): {desc}")
        lines.append("")
    extra = extra.strip()
    if extra:
        lines.extend(["## Host notes", "", extra, ""])
    return "\n".join(lines)
