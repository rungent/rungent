from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.resources
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import Agent
from .testing import BaselineSuite


def _load_value(target: str) -> Any:
    module_name, separator, attribute = target.partition(":")
    if not separator:
        raise ValueError("Target must use module:attribute syntax")
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    return getattr(importlib.import_module(module_name), attribute)


def _load_agent(target: str) -> Agent:
    value = _load_value(target)
    if not isinstance(value, Agent):
        raise TypeError(f"{target} is not a rungent.Agent")
    return value


def doctor(target: str) -> int:
    agent = _load_agent(target)
    payload = {
        "agent": agent.name,
        "tools": [
            {
                "name": item.name,
                "effect": item.effect,
                "approval": item.approval,
                "schema": item.schema()["function"]["parameters"],
            }
            for item in agent.tools
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def run_baseline_suite(target: str, case_names: list[str] | None = None) -> int:
    suite = _load_value(target)
    if not isinstance(suite, BaselineSuite):
        raise TypeError(f"{target} is not a rungent.testing.BaselineSuite")
    if case_names:
        selected = [case for case in suite.cases if case.name in case_names]
        missing = set(case_names) - {case.name for case in selected}
        if missing:
            raise ValueError(f"Unknown baseline cases: {', '.join(sorted(missing))}")
        suite = replace(suite, cases=selected)
    report = asyncio.run(suite.run())
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.passed else 1


def init_project(path: str) -> int:
    destination = Path(path) / "RUNGENT.md"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    source = importlib.resources.files("rungent").joinpath("RUNGENT.md")
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Created {destination}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="rungent")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor", help="Validate and print an Agent definition")
    doctor_parser.add_argument("agent", help="Python target in module:attribute form")
    init_parser = commands.add_parser("init", help="Add Rungent's AI integration guide")
    init_parser.add_argument("path", nargs="?", default=".")
    baseline_parser = commands.add_parser("baseline", help="Run an application BaselineSuite")
    baseline_parser.add_argument("suite", help="Python target in module:attribute form")
    baseline_parser.add_argument(
        "--case", dest="cases", action="append", help="Run one exact case name; repeatable"
    )
    args = parser.parse_args()
    if args.command == "doctor":
        return doctor(args.agent)
    if args.command == "baseline":
        return run_baseline_suite(args.suite, args.cases)
    return init_project(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
