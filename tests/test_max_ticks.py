"""The non-termination guard is a setting, not a wall (R7).

`max_ticks` bounds the loop iterations before a run is declared non-terminating. One
iteration is one poll interval, so under fixed-interval polling the guard also bounds the
makespan a run can reach -- and until this was settable, a workflow longer than the default
could not be run at all from the CLI, at any interval that kept its observed times honest.

The guard still exists, because the failure it catches is real: a backend whose clock does
not advance would otherwise loop forever. `0` on the CLI (None in the library) is how a
caller says it accepts that risk for a long run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, main  # noqa: E402
from ofplang.run.runner import DEFAULT_MAX_TICKS, RollingRunner, RunnerError  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")

# simple: source (2) -> transport (1) -> target (2); makespan 5, so 6 ticks at interval 1.


def test_the_default_is_the_documented_one():
    assert DEFAULT_MAX_TICKS == 100_000
    assert RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0).max_ticks == DEFAULT_MAX_TICKS


def test_a_run_longer_than_the_limit_is_given_up_on():
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0, max_ticks=3)
    with pytest.raises(RunnerError) as excinfo:
        runner.run()
    message = str(excinfo.value)
    # The message has to say which knob and what it bounds: the run is fine, the limit is
    # not, and nothing else in the output would tell the reader that.
    assert "exceeded max ticks (3)" in message
    assert "--max-ticks" in message


def test_no_limit_runs_a_workflow_the_limit_would_have_stopped():
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0, max_ticks=None)
    status = runner.run()
    assert status["now"] == 5
    assert runner.ticks == 6  # more than the limit that failed above


# --- the CLI translation ------------------------------------------------------


def _run(tmp_path, *extra):
    out = tmp_path / "status.yaml"
    return main(
        ["run", SIMPLE_WF, "--env", SIMPLE_ENV, "--seed", "0", "-o", str(out), *extra]
    )


def test_cli_reports_a_run_that_hits_the_limit_as_a_failure(tmp_path, capsys):
    # An execution failure (exit 1), not a usage error: the invocation was fine.
    assert _run(tmp_path, "--max-ticks", "3") == EXIT_FAILED
    assert "exceeded max ticks (3)" in capsys.readouterr().err


def test_cli_zero_means_no_limit(tmp_path):
    assert _run(tmp_path, "--max-ticks", "0") == EXIT_OK


def test_cli_rejects_a_negative_limit(tmp_path, capsys):
    # Not a limit at all, so it is an input error rather than something to interpret.
    assert _run(tmp_path, "--max-ticks", "-1") == EXIT_USAGE
    assert "--max-ticks must not be negative" in capsys.readouterr().err


def test_cli_default_completes_the_run(tmp_path):
    assert _run(tmp_path) == EXIT_OK
