"""Tests for the rolling-horizon runner (milestone 2b-1).

These drive real v0 workflows to completion through `ofplang.schedule` (called
in-process) against the simulator, exercising the replan round-trip: every tick
the committed history is fed back to the scheduler, fixed, and the rest
re-optimised. The scheduler is a required dependency here; if it is not installed
these tests are skipped.

Fixtures live in tests/fixtures/ (self-contained copies of the schedule examples).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.cli import EXIT_FAILED, EXIT_OK, main  # noqa: E402
from ofplang.run.runner import RollingRunner, RunnerError, load_document  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")
IFACE_WF = str(FIXTURES / "interface_load.workflow.yaml")
IFACE_ENV = str(FIXTURES / "interface_load.env.yaml")
NO_TARGET_ENV = str(FIXTURES / "simple_no_target.env.yaml")


def test_simple_workflow_runs_to_completion():
    # source -> transport -> target; no interface (source creates the Object).
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0)
    status = runner.run()
    assert all(a["status"] == "completed" for a in status["activities"])
    assert status["now"] == 5  # the plan's makespan


def test_round_trip_replans_with_committed_history():
    # The run must go through several replan cycles, each feeding committed history
    # back to the scheduler -- that is what 2b-1 validates. A single cycle would
    # mean the history round-trip was never exercised.
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0)
    runner.run()
    assert runner.ticks > 1


def test_committed_activities_carry_provenance():
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0)
    status = runner.run()
    kinds = [a["kind"] for a in status["activities"]]
    assert kinds.count("processing") == 2  # source + target
    assert "transport" in kinds
    # Provenance is preserved for the scheduler to match on replans.
    procs = [a for a in status["activities"] if a["kind"] == "processing"]
    assert {tuple(a["node"]) for a in procs} == {("SampleSource",), ("SampleTarget",)}


def test_interface_workflow_seeds_and_delivers():
    # A workflow with an Object-bearing entry input: the runner seeds `sample` at
    # loader.stage (interface.inputs) and drives it through heat to output.slot.
    doc = load_document(FIXTURES / "interface_load.document.yaml")
    interface = doc["interface"]
    runner = RollingRunner(IFACE_WF, IFACE_ENV, interface, random_seed=0)
    status = runner.run()
    assert all(a["status"] == "completed" for a in status["activities"])
    # The interface is carried through unchanged (§6.8).
    assert status["interface"]["inputs"] == {"sample": "loader.stage"}
    # loader.stage -> heater.stage (2) + heat (10) + heater.stage -> output.slot (2).
    assert status["now"] == 14


def test_infeasible_workflow_raises():
    # The environment has no capability for `target`, so no plan can be produced.
    runner = RollingRunner(SIMPLE_WF, NO_TARGET_ENV, random_seed=0)
    with pytest.raises(RunnerError):
        runner.run()


# -- CLI end to end --------------------------------------------------------

def test_cli_run_simple(tmp_path, capsys):
    out = tmp_path / "status.yaml"
    code = main(["run", SIMPLE_WF, "--env", SIMPLE_ENV, "--seed", "0", "-o", str(out)])
    assert code == EXIT_OK
    status = load_document(out)
    assert status["now"] == 5
    assert all(a["status"] == "completed" for a in status["activities"])


def test_cli_run_interface(tmp_path):
    out = tmp_path / "status.yaml"
    iface_doc = str(FIXTURES / "interface_load.document.yaml")
    code = main(
        ["run", IFACE_WF, "--env", IFACE_ENV, "--interface", iface_doc, "--seed", "0", "-o", str(out)]
    )
    assert code == EXIT_OK
    status = load_document(out)
    assert status["now"] == 14


def test_cli_run_infeasible_reports_failure(capsys):
    code = main(["run", SIMPLE_WF, "--env", NO_TARGET_ENV, "--seed", "0"])
    assert code == EXIT_FAILED
    assert "execution failed" in capsys.readouterr().err
