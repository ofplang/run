"""Tests for the failure stop policy (dev-notes design.md D25 / the D-b milestone).

v0 stops the whole run on any activity failure: once an operation is observed
`failed`, the runner dispatches no more work and only waits for what is still
running to finish (it never sends an abort). The final status marks the failed
activity `failed` and every activity that never started `cancelled`; the run
counts as failed (`runner.failed`, CLI exit 1).

Failures are injected on the simulator by capability, up front (like device
faults, not via the CLI): `schedule_process_failure(process, mode)` /
`schedule_transport_failure(transporter, from, to)`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.schedule import validate_document  # noqa: E402

from ofplang.run.cli import EXIT_FAILED, main  # noqa: E402
from ofplang.run.runner import RollingRunner, serialize_document  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
FAIL_WF = str(FIXTURES / "failure.workflow.yaml")
FAIL_ENV = str(FIXTURES / "failure.env.yaml")
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")


def _by_node(status: dict) -> dict:
    return {tuple(a["node"]): a for a in status["activities"] if a.get("node")}


def _assert_terminal(status: dict) -> None:
    # No activity is left running or pending: every one reached a terminal state.
    assert all(a["status"] in ("completed", "failed", "cancelled") for a in status["activities"])


# -- process failure -----------------------------------------------------------

def test_process_failure_stops_drains_and_cancels():
    # The `bad` chain's source fails; the parallel long `slow` source is still
    # running and must be drained (awaited, not aborted); everything downstream and
    # unstarted is cancelled. Event-boundary advance gives exact times.
    runner = RollingRunner(FAIL_WF, FAIL_ENV, poll_interval=None, random_seed=0)
    runner.sim.schedule_process_failure("src_bad", "m0")
    status = runner.run()

    assert runner.failed
    acts = _by_node(status)
    # The failing source is failed at its end; the long parallel source completed.
    assert acts[("SrcBad",)]["status"] == "failed" and acts[("SrcBad",)]["end"] == 2
    assert acts[("SrcSlow",)]["status"] == "completed" and acts[("SrcSlow",)]["end"] == 6
    # Everything else -- the failed chain's sink and the slow chain's (never
    # dispatched) sink -- is cancelled.
    assert acts[("SinkBad",)]["status"] == "cancelled"
    assert acts[("SinkSlow",)]["status"] == "cancelled"
    # Both pending transports are cancelled too.
    cancelled_transports = [
        a for a in status["activities"] if a["kind"] == "transport" and a["status"] == "cancelled"
    ]
    assert len(cancelled_transports) == 2
    _assert_terminal(status)
    # The clock rests at the last running operation's finish (the drained slow op).
    assert status["now"] == 6


def test_process_failure_fixed_interval_stops_too():
    # The same outcome under the standard fixed-interval polling.
    runner = RollingRunner(FAIL_WF, FAIL_ENV, random_seed=0)  # default poll_interval=1
    runner.sim.schedule_process_failure("src_bad", "m0")
    status = runner.run()
    assert runner.failed
    acts = _by_node(status)
    assert acts[("SrcBad",)]["status"] == "failed"
    assert acts[("SrcSlow",)]["status"] == "completed"
    assert acts[("SinkBad",)]["status"] == "cancelled"
    assert acts[("SinkSlow",)]["status"] == "cancelled"
    _assert_terminal(status)


def test_failed_status_is_a_valid_but_terminal_document(tmp_path):
    # The final status of a failed run is a valid execution document (§6.2 accepts
    # failed / cancelled), but it is terminal: feeding it back to the scheduler is
    # rejected, because a stopped run has no remaining work to plan (D25).
    runner = RollingRunner(FAIL_WF, FAIL_ENV, poll_interval=None, random_seed=0)
    runner.sim.schedule_process_failure("src_bad", "m0")
    status = runner.run()

    out = tmp_path / "status.yaml"
    out.write_text(serialize_document(status), encoding="utf-8")
    assert validate_document(out).ok, [(d.code, d.path) for d in validate_document(out).errors]

    from ofplang.schedule import schedule

    report = schedule(FAIL_WF, FAIL_ENV, document_path=out)
    assert not report.ok
    assert "terminal_status_not_replannable" in {d.code for d in report.diagnostics}


# -- transport failure ---------------------------------------------------------

def test_transport_failure_stops_run():
    # Failing the transport of the simple chain: the source completes, the transport
    # fails, and the target (which needed the delivered object) is cancelled.
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, poll_interval=None, random_seed=0)
    runner.sim.schedule_transport_failure("transport", "station_0.core", "station_1.core")
    status = runner.run()

    assert runner.failed
    acts = _by_node(status)
    assert acts[("SampleSource",)]["status"] == "completed"
    assert acts[("SampleTarget",)]["status"] == "cancelled"
    transport = next(a for a in status["activities"] if a["kind"] == "transport")
    assert transport["status"] == "failed"
    _assert_terminal(status)


def test_source_failure_cancels_the_whole_chain():
    # Failing the very first activity cancels the entire remaining linear chain.
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, poll_interval=None, random_seed=0)
    runner.sim.schedule_process_failure("source", "m0")
    status = runner.run()

    assert runner.failed
    acts = _by_node(status)
    assert acts[("SampleSource",)]["status"] == "failed"
    assert acts[("SampleTarget",)]["status"] == "cancelled"
    assert all(a["status"] != "completed" for a in status["activities"])
    # The source failed at its planned end; nothing ran afterwards.
    assert status["now"] == 2


# -- CLI mapping ---------------------------------------------------------------

def test_cli_maps_failed_run_to_exit_failed(tmp_path, monkeypatch, capsys):
    # A failed run is not an exception: the CLI still writes the status but returns
    # EXIT_FAILED. Failure is not CLI-injectable (Python-only), so a fake runner
    # stands in for one whose run failed.
    import ofplang.run.cli as cli

    class _FakeRunner:
        failed = True

        def __init__(self, *a, **k):
            pass

        def run(self):
            return {"now": 2, "activities": [{"kind": "processing", "status": "failed",
                                              "start": 0, "end": 2, "process": "source",
                                              "mode": "m0", "node": ["S"]}]}

    monkeypatch.setattr(cli, "RollingRunner", _FakeRunner)
    out = tmp_path / "status.yaml"
    code = main(["run", SIMPLE_WF, "--env", SIMPLE_ENV, "-o", str(out)])
    assert code == EXIT_FAILED
    assert out.is_file()  # the status is still emitted
    assert "execution failed" in capsys.readouterr().err
