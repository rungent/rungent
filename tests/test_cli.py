import json

from rungent import Agent, Runtime
from rungent import cli as cli_module
from rungent.llm import ModelCompleted
from rungent.store import MemoryStore
from rungent.testing import BaselineCase, BaselineSuite, ScriptedModel


def test_baseline_cli_runs_suite_and_prints_machine_readable_report(monkeypatch, capsys):
    suite = BaselineSuite(
        runtime=Runtime(
            agents=[Agent(name="assistant", instructions="Reply")],
            model=ScriptedModel([ModelCompleted(text="Done")]),
            store=MemoryStore(),
        ),
        cases=[BaselineCase(name="plain", input="Hello")],
    )
    monkeypatch.setattr(cli_module, "_load_value", lambda _target: suite)

    assert cli_module.run_baseline_suite("application:suite", ["plain"]) == 0
    assert json.loads(capsys.readouterr().out)["pass_rate"] == 1.0
