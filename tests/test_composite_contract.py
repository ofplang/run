"""Contracts on the top-level composite `main` (spec §9; dev-notes design.md D32
Phase 1).

The entry composite's contracts are the whole-workflow envelope: `requires` is
checked at the run boundary before any work runs, and `ensures` at the end over
the run's inputs and produced outputs. A violation stops the run gracefully (D25):
a `requires` failure before the run starts (no activity runs), an `ensures` failure
at the end (activities stay completed, the run is marked failed). This reuses the
contract evaluator; only the check points and the boundary value source differ from
the atomic case (tests/test_contract.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
WF = str(FIXTURES / "composite_contract.workflow.yaml")
ENV = str(FIXTURES / "composite_contract.env.yaml")


def _boundary(raw):
    return {"boundary": {"inputs": {"raw": {"view": raw}}}}


def _processing(status):
    return {
        a.get("process"): a["status"]
        for a in status["activities"]
        if a.get("kind") == "processing"
    }


def test_entry_composite_contracts_are_parsed():
    # The top-level entry composite's contracts are picked up (unlike nested / other
    # composites), so they can be checked at the run boundary.
    runner = RollingRunner(WF, ENV, _boundary(72), random_seed=0)
    assert runner.jobs[0].entry_is_composite is True
    assert "main" in runner.jobs[0].contract_asts
    assert (
        "requires" in runner.jobs[0].contract_asts["main"]
        and "ensures" in runner.jobs[0].contract_asts["main"]
    )


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_main_contracts_hold(poll_interval):
    # raw = 72: requires (raw >= 0) holds and margin = 72 - 60 = 12 satisfies ensures.
    runner = RollingRunner(WF, ENV, _boundary(72), poll_interval=poll_interval, random_seed=0)
    status = runner.run()

    assert not runner.failed
    assert runner.outputs == {"margin": 12}
    assert _processing(status) == {"score": "completed"}


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_main_requires_violation_stops_before_the_run_starts(poll_interval):
    # raw = -5 violates `main.requires: inputs.raw.view >= 0`. The run must not
    # proceed: no activity is dispatched (activities are empty), and the run is failed.
    runner = RollingRunner(WF, ENV, _boundary(-5), poll_interval=poll_interval, random_seed=0)
    status = runner.run()

    assert runner.failed
    assert status["activities"] == []   # nothing ran (S2)
    assert runner.outputs == {}


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_main_ensures_violation_fails_at_end(poll_interval, tmp_path):
    # An inner script that adds instead of subtracting makes the returned margin
    # (raw + 60) break `main.ensures: outputs.margin.view == inputs.raw.view - 60`.
    # The workflow runs to completion, then the whole-workflow ensures fails: the
    # activity stays `completed` but the run is marked failed (S3).
    wf = tmp_path / "bad_main_ensures.workflow.yaml"
    wf.write_text(
        Path(WF).read_text(encoding="utf-8").replace(
            'return {"margin": raw - threshold}', 'return {"margin": raw + threshold}'
        ),
        encoding="utf-8",
    )
    runner = RollingRunner(str(wf), ENV, _boundary(72), poll_interval=poll_interval, random_seed=0)
    status = runner.run()

    assert runner.failed
    assert _processing(status) == {"score": "completed"}   # the activity itself succeeded
    assert runner.outputs == {"margin": 132}               # the out-of-contract value was produced
