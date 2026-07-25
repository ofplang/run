"""Smoke tests for the ofp-run CLI scaffold.

These pin the CLI's shape and exit-code contract while the library is built out;
they are intentionally light and will grow as `run` gains real behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ofplang.run.cli import EXIT_USAGE, main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_BROKEN_YAML = "a: [1, 2\nb: :::\n"


def test_help_exits_zero(capsys):
    # `--help` is handled by argparse and exits with code 0.
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "ofp-run" in capsys.readouterr().out


def test_missing_subcommand_is_usage_error():
    # A subcommand is required; omitting it is a usage error (argparse -> exit 2).
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == EXIT_USAGE


def test_run_requires_env():
    # `run` needs --env; omitting it is an argparse usage error (exit 2).
    with pytest.raises(SystemExit) as exc:
        main(["run", "workflow.yaml"])
    assert exc.value.code == EXIT_USAGE


def test_run_missing_workflow_is_usage_error(capsys):
    # A workflow path that does not exist is an input (usage) error, not a failure.
    assert main(["run", "does_not_exist.yaml", "--env", "nope.yaml"]) == EXIT_USAGE
    assert "workflow not found" in capsys.readouterr().err


def test_replay_missing_plan_is_usage_error(capsys):
    # `replay` reads a plan; a missing plan file is a usage error.
    assert main(["replay", "does_not_exist.yaml", "--env", "nope.yaml"]) == EXIT_USAGE
    assert "cannot read plan" in capsys.readouterr().err


def test_run_malformed_workflow_is_usage_error(tmp_path, capsys):
    # A malformed workflow (valid env) is an input error (exit 2), not an
    # uncaught traceback -- the runner is the untrusted boundary.
    bad = tmp_path / "broken.workflow.yaml"
    bad.write_text(_BROKEN_YAML, encoding="utf-8")
    code = main(["run", str(bad), "--env", str(EXAMPLES / "count_chain.env.yaml")])
    assert code == EXIT_USAGE
    assert "invalid input" in capsys.readouterr().err


def test_run_malformed_contract_is_usage_error(tmp_path, capsys):
    # An unparsable contract expression is an input error (exit 2), not a
    # traceback or a spurious "execution failed".
    wf_text = (EXAMPLES / "count_chain.workflow.yaml").read_text(encoding="utf-8")
    wf_text = wf_text.replace(
        "  inc:\n    kind: atomic\n",
        (
            '  inc:\n    kind: atomic\n    contracts:\n'
            '      requires:\n        - expr: "inputs.x.view.value >>>"\n'
        ),
    )
    wf = tmp_path / "bad_contract.workflow.yaml"
    wf.write_text(wf_text, encoding="utf-8")
    code = main(["run", str(wf), "--env", str(EXAMPLES / "count_chain.env.yaml")])
    assert code == EXIT_USAGE
    assert "invalid input" in capsys.readouterr().err
