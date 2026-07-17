"""Smoke tests for the ofp-run CLI scaffold.

These pin the CLI's shape and exit-code contract while the library is built out;
they are intentionally light and will grow as `run` gains real behaviour.
"""

from __future__ import annotations

import pytest

from ofplang.run.cli import EXIT_USAGE, main


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
