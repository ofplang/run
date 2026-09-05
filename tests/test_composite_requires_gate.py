"""A nested composite's `requires` gates its whole body (spec §9; review #1).

D34 checks a nested composite's `requires` at its value boundary -- once its inputs
are available. A body node that is *independent* of those inputs (here `Make`, which
reads none of the composite's inputs) has no dataflow dependency on them, so the
scheduler places it at run start. Without a gate it would run before the composite's
`requires` was ever evaluated. The runner therefore defers every body activity until
its composite's `requires` has been checked: on a violation, no body node runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
WF = str(FIXTURES / "composite_requires_gate.workflow.yaml")
ENV = str(FIXTURES / "composite_requires_gate.env.yaml")


def _boundary(a):
    return {"boundary": {"inputs": {"a": {"view": a}}}}


def _by_node(status):
    return {
        "/".join(a["node"]): a["status"]
        for a in status["activities"]
        if a.get("kind") == "processing"
    }


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_input_independent_body_node_is_gated_on_violation(poll_interval):
    # inp = -1 violates wrap's `requires: inputs.inp.view >= 0`. `Make` is independent
    # of `inp`, so the scheduler would start it at t=0; the gate must hold it until
    # wrap's `requires` is checked -- which fails -- so `Make` never runs.
    runner = RollingRunner(WF, ENV, _boundary(-1), poll_interval=poll_interval, random_seed=0)
    status = runner.run()
    assert runner.failed
    assert runner.failure.kind == "contract_requires"
    by_node = _by_node(status)
    assert by_node["Pre"] == "completed"      # outside wrap: runs, produces wrap's input
    assert by_node["W/Make"] == "cancelled"   # gated: must NOT run before the violated requires
    assert by_node["W/Combine"] == "cancelled"


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_body_runs_when_requires_holds(poll_interval):
    # inp = 5 satisfies the precondition, so wrap's body runs and the workflow
    # completes: out = inp + Make.m = 5 + 1 = 6.
    runner = RollingRunner(WF, ENV, _boundary(5), poll_interval=poll_interval, random_seed=0)
    status = runner.run()
    assert not runner.failed
    assert runner.outputs == {"r": 6}
    assert all(a["status"] == "completed" for a in status["activities"])


def test_gate_helper_reflects_checked_requires():
    # The gate is closed for a body activity of `W` until wrap's `requires` is checked,
    # and open for an activity outside any requires-bearing composite.
    runner = RollingRunner(WF, ENV, _boundary(5), random_seed=0)
    job = runner.jobs[0]
    assert runner._requires_gate_open(job, ("Pre",)) is True   # not under a nested composite
    assert runner._requires_gate_open(job, ("W", "Make")) is False  # W's requires unchecked
    job.checked_requires.add(("W",))
    assert runner._requires_gate_open(job, ("W", "Make")) is True  # opens once it is checked
