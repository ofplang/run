"""Contracts on a nested composite (spec §9; dev-notes design.md D34 / Phase 2).

A composite invoked inside the workflow carries `requires` / `ensures`; the runner
checks them at the composite's value boundary even though the composite is
flattened away -- `requires` once the composite's inputs are available (which, when
an input is fed by an upstream atomic, is mid-run, before the composite's body
runs), `ensures` once its outputs are available. A violation stops the run
gracefully (D25) at the composite boundary: no single activity is marked failed,
and the composite's not-yet-run body is cancelled.

Relies on the scheduler exposing each nested composite's boundary (`Workflow.
composites`, D34); the boundary mapping itself is unit-tested in ofplang-schedule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
WF = str(FIXTURES / "nested_composite_contract.workflow.yaml")
ENV = str(FIXTURES / "nested_composite_contract.env.yaml")


def _boundary(a):
    return {"boundary": {"inputs": {"a": {"view": a}}}}


def _by_node(status):
    return {tuple(a["node"]): a["status"] for a in status["activities"] if a.get("kind") == "processing"}


def test_nested_composite_boundary_is_exposed_to_the_runner():
    # The runner sees the nested composite `W`'s boundary (from the scheduler): its
    # input `inp` fed by the atomic `Pre`, its output `out` produced by the atomic
    # `Inc` inside it.
    runner = RollingRunner(WF, ENV, _boundary(5), random_seed=0)
    assert ("W",) in runner.dataflow.composites
    boundary = runner.dataflow.composites[("W",)]
    assert boundary.process == "wrap"
    assert boundary.inputs == {"inp": (("Pre",), "p")}
    assert boundary.outputs == {"out": (("W", "Inc"), "y")}


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_nested_contracts_hold(poll_interval):
    # a = 5: wrap.inp = 5 satisfies requires (>= 0), and wrap.out = 6 satisfies
    # ensures (== inp + 1).
    runner = RollingRunner(WF, ENV, _boundary(5), poll_interval=poll_interval, random_seed=0)
    status = runner.run()

    assert not runner.failed
    assert runner.outputs == {"r": 6}
    assert _by_node(status) == {("Pre",): "completed", ("W", "Inc"): "completed"}


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_nested_requires_violation_stops_mid_run(poll_interval):
    # a = -2: wrap.inp is fed by the upstream `Pre`, so wrap.requires (inp >= 0) is
    # checked after `Pre` completes and before wrap's body (`Inc`) runs. The violation
    # stops the run: `Pre` stays completed, `Inc` is cancelled, and no output is
    # produced.
    runner = RollingRunner(WF, ENV, _boundary(-2), poll_interval=poll_interval, random_seed=0)
    status = runner.run()

    assert runner.failed
    assert _by_node(status) == {("Pre",): "completed", ("W", "Inc"): "cancelled"}
    assert runner.outputs == {}


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_nested_ensures_violation_fails_at_completion(poll_interval, tmp_path):
    # The inner script adds 2 instead of 1, so wrap.out (= inp + 2) breaks
    # wrap.ensures (out == inp + 1). The body runs to completion, then the ensures
    # fails: the activities stay completed but the run is marked failed.
    wf = tmp_path / "bad_nested_ensures.workflow.yaml"
    wf.write_text(
        Path(WF).read_text(encoding="utf-8").replace('return {"y": x + 1}', 'return {"y": x + 2}'),
        encoding="utf-8",
    )
    runner = RollingRunner(str(wf), ENV, _boundary(5), poll_interval=poll_interval, random_seed=0)
    status = runner.run()

    assert runner.failed
    assert _by_node(status) == {("Pre",): "completed", ("W", "Inc"): "completed"}
    assert runner.outputs == {"r": 7}
