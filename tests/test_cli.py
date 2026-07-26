"""Smoke tests for the ofp-run CLI scaffold.

These pin the CLI's shape and exit-code contract while the library is built out;
they are intentionally light and will grow as `run` gains real behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ofplang.run.cli import EXIT_OK, EXIT_USAGE, main

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


def test_run_malformed_workflow_is_caught_by_front_door(tmp_path, capsys):
    # A malformed workflow is rejected by the ofplang-validate front door. For
    # `run`, anything that stops it before execution is a usage error (exit 2);
    # the failure is reported with its validate error code.
    bad = tmp_path / "broken.workflow.yaml"
    bad.write_text(_BROKEN_YAML, encoding="utf-8")
    code = main(["run", str(bad), "--env", str(EXAMPLES / "count_chain.env.yaml")])
    assert code == EXIT_USAGE
    assert "wrong_value_kind" in capsys.readouterr().err


def test_run_no_validate_bypasses_front_door_on_malformed(tmp_path, capsys):
    # With --no-validate the front door is skipped, so a malformed workflow falls
    # through to the runner's own input guard: still a usage error (exit 2).
    bad = tmp_path / "broken.workflow.yaml"
    bad.write_text(_BROKEN_YAML, encoding="utf-8")
    code = main(
        ["run", str(bad), "--env", str(EXAMPLES / "count_chain.env.yaml"), "--no-validate"]
    )
    assert code == EXIT_USAGE
    assert "invalid input" in capsys.readouterr().err


def test_run_malformed_contract_is_caught_by_front_door(tmp_path, capsys):
    # An unparsable contract expression is caught by the front door (exit 2),
    # reported with its validate code rather than reaching the runner.
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
    assert "contract_parse_error" in capsys.readouterr().err


def _count_chain_with_bogus_key(tmp_path) -> Path:
    """A parseable count_chain workflow that is invalid v0 (unknown top-level key)
    but that the runner still tolerates — it reads only the keys it needs."""
    src = (EXAMPLES / "count_chain.workflow.yaml").read_text(encoding="utf-8")
    wf = tmp_path / "bogus.workflow.yaml"
    wf.write_text("bogus_key: 1\n" + src, encoding="utf-8")
    return wf


def test_run_invalid_v0_workflow_is_usage_error(tmp_path, capsys):
    # A parseable-but-invalid-v0 workflow (not just malformed YAML) is caught by
    # the front door with its specific validate code.
    wf = _count_chain_with_bogus_key(tmp_path)
    code = main(["run", str(wf), "--env", str(EXAMPLES / "count_chain.env.yaml")])
    assert code == EXIT_USAGE
    assert "unknown_key" in capsys.readouterr().err


def test_run_no_validate_runs_invalid_v0(tmp_path):
    # --no-validate lets an invalid-v0-but-runnable workflow through: the runner
    # ignores the unknown key and drives it to completion.
    wf = _count_chain_with_bogus_key(tmp_path)
    out = tmp_path / "status.yaml"
    code = main(
        [
            "run",
            str(wf),
            "--env",
            str(EXAMPLES / "count_chain.env.yaml"),
            "--no-validate",
            "-o",
            str(out),
        ]
    )
    assert code == EXIT_OK
