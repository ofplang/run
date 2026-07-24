"""Run-start preflight of run-phase preconditions (spec §5.6 / §9; dev-notes D37).

An atomic `requires` expression that references only run/graph-phase inputs is
knowable at run start, so it is checked as a preflight -- before any work is
dispatched -- instead of when the (possibly late-dispatched) process would run. A
violation stops the run at t=0 with no activity having run. `requires` that reads a
data-phase input stays a runtime check at dispatch (D32), unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
WF = str(FIXTURES / "preflight.workflow.yaml")
ENV = str(FIXTURES / "preflight.env.yaml")
# contract.workflow's `score` has a requires over a data-phase input (stays runtime).
CONTRACT_WF = str(FIXTURES / "contract.workflow.yaml")
CONTRACT_ENV = str(FIXTURES / "contract.env.yaml")


def _run(limit, poll_interval=None):
    b = {"boundary": {"inputs": {"seed": {"view": 10}, "limit": {"view": limit}}}}
    runner = RollingRunner(WF, ENV, b, poll_interval=poll_interval, random_seed=0)
    status = runner.run()
    return runner, status


# -- classification ----------------------------------------------------------


def test_run_phase_requires_is_classified_preflight():
    runner = RollingRunner(WF, ENV, random_seed=0)
    # `check`'s requires reads only the run-phase `limit`, so it is hoisted to preflight
    # (and is not left as a dispatch-time `requires`).
    assert "requires_preflight" in runner._contract_asts["check"]
    assert "requires" not in runner._contract_asts["check"]


def test_data_phase_requires_stays_runtime():
    # `score`'s requires reads a data-phase input, so it is NOT preflighted -- it stays
    # a dispatch-time check.
    runner = RollingRunner(CONTRACT_WF, CONTRACT_ENV, {"boundary": {"inputs": {"raw": {"view": 1}}}}, random_seed=0)
    assert "requires" in runner._contract_asts["score"]
    assert "requires_preflight" not in runner._contract_asts["score"]


# -- behavior ----------------------------------------------------------------


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_run_phase_precondition_holds_completes(poll_interval):
    runner, status = _run(3, poll_interval)
    assert not runner.failed
    assert runner.outputs == {"ok": 14}  # Slow.d = 11, ok = 11 + 3
    assert all(a["status"] == "completed" for a in status["activities"])


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_run_phase_precondition_violation_stops_at_run_start(poll_interval):
    # limit = -1 violates `requires: inputs.limit.view >= 0`. Because `limit` is
    # run-phase, the check is hoisted to run start: the run fails at t=0, before the
    # 5-unit `Slow` (or anything) runs, so no activity ran.
    runner, status = _run(-1, poll_interval)
    assert runner.failed
    assert status["activities"] == []          # nothing ran (preflight, before dispatch)
    assert status["now"] == 0
    assert runner.failure.kind == "contract_requires_preflight"
    assert runner.failure.subject == "Check"
    assert runner.failure.now == 0


def test_preflight_violation_is_observed_at_run_start():
    trace: list[dict] = []
    b = {"boundary": {"inputs": {"seed": {"view": 10}, "limit": {"view": -1}}}}
    runner = RollingRunner(WF, ENV, b, poll_interval=None, random_seed=0, contract_observer=trace.append)
    runner.run()
    # The observer saw the preflight check fail, at run start (t=0).
    assert any(
        r["subject"] == "Check" and r["section"] == "requires_preflight" and r["held"] is False and r["now"] == 0
        for r in trace
    )
