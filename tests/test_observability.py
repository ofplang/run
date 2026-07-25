"""Observability of failures and contract checks (dev-notes design.md D36).

Three facets:

* F-reason -- a run that stops records a structured `RollingRunner.failure` with a
  machine-readable `kind` (reason code), a human-readable `detail`, the `subject`
  that failed, and the time; distinct codes for contract / script / injected
  failures.
* F-stderr -- the CLI prints that reason (code + detail) to stderr.
* F-trace -- an optional `contract_observer` callback is called for every contract
  check (held or violated), for tracing / debugging.

The reason is deliberately NOT in the status document (which stays a valid §6
document); it is exposed via the runner attribute and stderr.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.cli import EXIT_FAILED, main  # noqa: E402
from ofplang.run.runner import RollingRunner  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
# contract.workflow.yaml: `score` has requires (raw >= 0) and ensures
# (margin == raw - threshold), threshold = 60; downstream `report`.
WF = str(FIXTURES / "contract.workflow.yaml")
ENV = str(FIXTURES / "contract.env.yaml")


def _variant(tmp_path, old, new):
    """A copy of the contract workflow with `old` replaced by `new` (to inject a
    specific failure)."""
    wf = tmp_path / "variant.workflow.yaml"
    wf.write_text(Path(WF).read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    return str(wf)


def _run(workflow, raw):
    runner = RollingRunner(
        workflow,
        ENV,
        {"boundary": {"inputs": {"raw": {"view": raw}}}},
        poll_interval=None,
        random_seed=0,
    )
    runner.run()
    return runner


# -- F-reason: reason codes --------------------------------------------------


def test_requires_violation_reason():
    runner = _run(WF, -5)  # requires: inputs.raw.view >= 0
    assert runner.failed
    assert runner.failure.kind == "contract_requires"
    assert runner.failure.subject == "Score"
    assert "inputs.raw.view >= 0" in runner.failure.detail
    assert runner.failure.now == 0  # checked at dispatch (t=0)


def test_ensures_violation_reason(tmp_path):
    wf = _variant(
        tmp_path,
        'return {"margin": raw - threshold}',
        'return {"margin": raw + threshold}',
    )
    runner = _run(wf, 72)
    assert runner.failed
    assert runner.failure.kind == "contract_ensures"
    assert runner.failure.subject == "Score"


def test_script_error_reason(tmp_path):
    wf = _variant(tmp_path, 'return {"margin": raw - threshold}', 'return {"margin": raw // 0}')
    runner = _run(wf, 72)
    assert runner.failed
    assert runner.failure.kind == "script_error"
    assert "ZeroDivisionError" in runner.failure.detail


def test_script_output_names_reason(tmp_path):
    wf = _variant(tmp_path, 'return {"margin": raw - threshold}', 'return {"wrong": 0}')
    runner = _run(wf, 72)
    assert runner.failed
    assert runner.failure.kind == "script_output_names"


def test_successful_run_has_no_failure():
    runner = _run(WF, 72)
    assert not runner.failed
    assert runner.failure is None


# -- F-stderr: CLI prints the reason -----------------------------------------


def test_cli_prints_failure_reason(tmp_path, capsys):
    boundary = tmp_path / "boundary.yaml"
    boundary.write_text("boundary:\n  inputs:\n    raw: {view: -5}\n", encoding="utf-8")
    out = tmp_path / "status.yaml"
    code = main(["run", WF, "--env", ENV, "--boundary", str(boundary), "-o", str(out)])
    assert code == EXIT_FAILED
    err = capsys.readouterr().err
    assert "contract_requires" in err
    assert "inputs.raw.view >= 0" in err


# -- F-trace: contract observer ----------------------------------------------


def test_contract_observer_records_every_check():
    trace: list[dict] = []
    runner = RollingRunner(
        WF, ENV, {"boundary": {"inputs": {"raw": {"view": 72}}}},
        poll_interval=None, random_seed=0, contract_observer=trace.append,
    )
    runner.run()

    assert not runner.failed
    # Every record carries the full shape.
    assert all({"subject", "process", "section", "expr", "held", "now"} <= set(r) for r in trace)
    seen = {(r["subject"], r["section"], r["held"]) for r in trace}
    # `score`'s requires and ensures were both checked and held.
    assert ("Score", "requires", True) in seen
    assert ("Score", "ensures", True) in seen


def test_contract_observer_records_a_violation():
    trace: list[dict] = []
    runner = RollingRunner(
        WF, ENV, {"boundary": {"inputs": {"raw": {"view": -5}}}},
        poll_interval=None, random_seed=0, contract_observer=trace.append,
    )
    runner.run()

    assert runner.failed
    # The observer saw the requires check fail (held is False).
    assert any(
        r["subject"] == "Score"
        and r["section"] == "requires"
        and r["held"] is False
        for r in trace
    )
