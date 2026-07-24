"""Type-level static view values (spec §7.4; dev-notes design.md D35).

A view field declaring `value:` is a type-level constant -- the same for every value
of that nominal type. The runner projects it onto every view record it routes, so a
Python script reading the field and a contract referencing it both see the static
value rather than a stale runtime default. This covers the value-layer core
(`default_value` / `with_static_views`) and the end-to-end runner path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ofplang.run.runner.contracts import Contracts, default_value, with_static_views

FIXTURES = Path(__file__).parent / "fixtures"
WF = str(FIXTURES / "static_view.workflow.yaml")
ENV = str(FIXTURES / "static_view.env.yaml")


# -- value-layer core --------------------------------------------------------


def _spec_type():
    return Contracts.from_workflow(WF).types["Spec"]


def test_static_view_value_is_captured():
    spec = _spec_type()
    assert spec.static_view == {"capacity": 96}


def test_default_value_carries_static_view_value():
    # A synthesised default of a type with a static view field takes the static value,
    # not the primitive default (0).
    assert default_value(_spec_type()) == {"capacity": 96}


def test_with_static_views_forces_the_static_field():
    spec = _spec_type()
    # A differing value is overwritten with the static value (option A); a matching one
    # is unchanged.
    assert with_static_views({"capacity": 5}, spec) == {"capacity": 96}
    assert with_static_views({"capacity": 96}, spec) == {"capacity": 96}


# -- end to end --------------------------------------------------------------

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_static_view_value_flows_to_script_and_contract(poll_interval):
    # `Tag` emits a Spec with capacity 5 (conflicting); the runner forces it to the
    # static 96 on record. `Use`'s script reads s["capacity"] == 96 -> n = 192, and
    # its contracts (requires capacity == 96, ensures n == capacity * 2) both hold. So
    # the run completes only because static view values are honored (D35).
    runner = RollingRunner(WF, ENV, poll_interval=poll_interval, random_seed=0)
    status = runner.run()

    assert not runner.failed
    assert all(a["status"] == "completed" for a in status["activities"])
    assert runner.outputs == {"n": 192}
    # The recorded Spec output was forced to the static value (option A).
    assert runner.values.get(("Tag",), "spec") == {"capacity": 96}
