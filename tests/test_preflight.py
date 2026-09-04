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
# producer_run_phase's `check` has a run-phase input fed by a producer (Src), so its
# preflight-candidate requires must be deferred to dispatch (review #2).
PROD_WF = str(FIXTURES / "producer_run_phase.workflow.yaml")
PROD_ENV = str(FIXTURES / "producer_run_phase.env.yaml")


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
    assert "requires_preflight" in runner.jobs[0].contract_asts["check"]
    assert "requires" not in runner.jobs[0].contract_asts["check"]


def test_data_phase_requires_stays_runtime():
    # `score`'s requires reads a data-phase input, so it is NOT preflighted -- it stays
    # a dispatch-time check.
    runner = RollingRunner(
        CONTRACT_WF,
        CONTRACT_ENV,
        {"boundary": {"inputs": {"raw": {"view": 1}}}},
        random_seed=0,
    )
    assert "requires" in runner.jobs[0].contract_asts["score"]
    assert "requires_preflight" not in runner.jobs[0].contract_asts["score"]


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


def test_producer_fed_run_phase_requires_is_deferred_to_dispatch():
    # `check`'s requires reads a run-phase input, so it is a preflight *candidate*
    # (process-level, phase-based classification retained). But at the Check node that
    # input is fed by the producer Src, so it is not fixed at run start: the per-node
    # split defers it (it is not checkable at run start, and is checked at dispatch).
    runner = RollingRunner(
        PROD_WF,
        PROD_ENV,
        {"boundary": {"inputs": {"seed": {"view": 5}}}},
        random_seed=0,
    )
    assert "requires_preflight" in runner.jobs[0].contract_asts["check"]
    checkable, deferred = runner._split_preflight(("Check",), "check")
    assert not checkable and len(deferred) == 1


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_producer_fed_run_phase_violation_caught_at_dispatch(poll_interval):
    # Src produces r = -1, violating `check`'s `requires: inputs.r.view >= 0`. The
    # value is only known after Src runs, so the check must fire at dispatch (against
    # the real -1), not at run start against the typed default 0 (which would hold and
    # hide the violation). So Src completes first, then Check fails.
    b = {"boundary": {"inputs": {"seed": {"view": -1}}}}
    runner = RollingRunner(PROD_WF, PROD_ENV, b, poll_interval=poll_interval, random_seed=0)
    status = runner.run()
    assert runner.failed
    assert runner.failure.kind == "contract_requires"  # a dispatch-time check, not preflight
    assert runner.failure.now > 0                       # after Src ran, not at t=0
    by_process = {a["process"]: a["status"] for a in status["activities"] if "process" in a}
    assert by_process.get("src") == "completed"
    assert by_process.get("check") == "failed"


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_producer_fed_run_phase_holds_completes(poll_interval):
    # Src produces r = 5, satisfying the precondition -> the run completes.
    b = {"boundary": {"inputs": {"seed": {"view": 5}}}}
    runner = RollingRunner(PROD_WF, PROD_ENV, b, poll_interval=poll_interval, random_seed=0)
    status = runner.run()
    assert not runner.failed
    assert runner.outputs == {"ok": 5}
    assert all(a["status"] == "completed" for a in status["activities"])


def test_preflight_violation_is_observed_at_run_start():
    trace: list[dict] = []
    b = {"boundary": {"inputs": {"seed": {"view": 10}, "limit": {"view": -1}}}}
    runner = RollingRunner(
        WF, ENV, b, poll_interval=None, random_seed=0, contract_observer=trace.append
    )
    runner.run()
    # The observer saw the preflight check fail, at run start (t=0).
    assert any(
        r["subject"] == "Check"
        and r["section"] == "requires_preflight"
        and r["held"] is False
        and r["now"] == 0
        for r in trace
    )
