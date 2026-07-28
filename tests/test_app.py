"""Tests for the library front door (`ofplang.run.app`): `front_door_check`,
`run_workflow`, `RunResult`, `FrontDoorError`.

These pin the shared seam the CLIs sit on -- the same validate + capability gate a
dialect wrapper (e.g. `lc`) reuses, and the `backend_factory` injection point -- at
the library level (the `ofp-run` CLI path is covered by `test_cli`). The scheduler is
required for a full run; those tests are skipped if it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ofplang.run.app import (
    FrontDoorError,
    RunResult,
    front_door_check,
    run_workflow,
)

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")

GENERIC_WF = """\
spec_version: "0.0"
processes:
  gen:
    kind: atomic
    type_params: {O: {domain: object}}
    inputs: {}
    outputs: {}
entry: gen
"""

IMPORT_WF = """\
spec_version: "0.0"
processes:
  $import: shared.yaml
entry: gen
"""


class FakeClock:
    """A clock where only `sleep` makes time pass, so a real-time backend can be
    driven in a test without spending real seconds."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


# -- front_door_check ----------------------------------------------------------


def test_front_door_accepts_valid_workflow():
    fd = front_door_check(SIMPLE_WF)
    assert fd.ok
    assert fd.diagnostics == []
    assert fd.unsupported is None


def test_front_door_rejects_generics(tmp_path):
    wf = tmp_path / "generic.workflow.yaml"
    wf.write_text(GENERIC_WF, encoding="utf-8")
    fd = front_door_check(str(wf), validate=False)  # isolate the capability gate
    assert not fd.ok
    assert fd.unsupported is not None
    assert "generic" in fd.unsupported


def test_front_door_rejects_import(tmp_path):
    wf = tmp_path / "import.workflow.yaml"
    wf.write_text(IMPORT_WF, encoding="utf-8")
    fd = front_door_check(str(wf), validate=False)
    assert not fd.ok
    assert "import" in fd.unsupported


def test_front_door_gate_runs_even_when_validate_skipped(tmp_path):
    # validate=False skips the ofplang-validate pass (no diagnostics) but the
    # capability gate still runs -- so an unsupported feature is still caught.
    wf = tmp_path / "generic.workflow.yaml"
    wf.write_text(GENERIC_WF, encoding="utf-8")
    fd = front_door_check(str(wf), validate=False)
    assert fd.diagnostics == []  # validate was skipped
    assert fd.unsupported is not None  # gate still fired


# -- run_workflow --------------------------------------------------------------


def test_run_workflow_returns_result():
    pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")
    result = run_workflow(SIMPLE_WF, SIMPLE_ENV, random_seed=0)
    assert isinstance(result, RunResult)
    assert not result.failed
    assert all(a["status"] == "completed" for a in result.status["activities"])


def test_run_workflow_validate_true_raises_front_door_error(tmp_path):
    wf = tmp_path / "generic.workflow.yaml"
    wf.write_text(GENERIC_WF, encoding="utf-8")
    with pytest.raises(FrontDoorError) as exc:
        run_workflow(str(wf), SIMPLE_ENV, validate=True)
    assert not exc.value.result.ok
    assert exc.value.result.unsupported is not None


def test_run_workflow_accepts_in_memory_document_with_validate_false():
    pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")
    import yaml

    doc = yaml.safe_load(Path(SIMPLE_WF).read_text(encoding="utf-8"))
    result = run_workflow(doc, SIMPLE_ENV, random_seed=0, validate=False)
    assert isinstance(result, RunResult)
    assert not result.failed
    assert all(a["status"] == "completed" for a in result.status["activities"])


def test_run_workflow_rejects_in_memory_document_with_validate_true():
    # The front door validates a file; an in-memory document must be validated by the
    # caller beforehand, so `validate=True` with a mapping is a usage error.
    import yaml

    doc = yaml.safe_load(Path(SIMPLE_WF).read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="validate=False"):
        run_workflow(doc, SIMPLE_ENV, validate=True)


def test_run_workflow_injects_backend_factory():
    pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")
    from ofplang.run.simulator import SubprocessBackend, subprocess_backend_factory

    clock = FakeClock()
    factory = subprocess_backend_factory(
        seconds_per_tick=1.0, monotonic=clock.monotonic, sleep=clock.sleep
    )
    seen: dict = {}
    orig = SubprocessBackend.close

    def spy_close(self):
        seen["closed"] = True
        return orig(self)

    SubprocessBackend.close = spy_close  # type: ignore[method-assign]
    try:
        result = run_workflow(
            SIMPLE_WF, SIMPLE_ENV, backend_factory=factory, random_seed=0, validate=False
        )
    finally:
        SubprocessBackend.close = orig  # type: ignore[method-assign]

    assert not result.failed
    assert all(a["status"] == "completed" for a in result.status["activities"])
    assert seen.get("closed") is True  # run_workflow closed the backend in its finally
