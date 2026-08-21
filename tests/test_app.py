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
    capability_gate,
    front_door_check,
    run_workflow,
)

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")
# Valid v0 the runner cannot execute (spec 4.1); shared with test_cli.
STRUCTURED_WF = str(FIXTURES / "structured_node.workflow.yaml")

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
    assert isinstance(fd.document, dict)  # the expanded document is returned


def test_front_door_rejects_generics(tmp_path):
    wf = tmp_path / "generic.workflow.yaml"
    wf.write_text(GENERIC_WF, encoding="utf-8")
    fd = front_door_check(str(wf), validate=False)  # isolate the capability gate
    assert not fd.ok
    assert fd.unsupported is not None
    assert "generic" in fd.unsupported


def test_front_door_rejects_structured_node():
    # A structured node is valid v0 this runner cannot execute (spec 4.1). Without the
    # gate it reached the scheduler and came back as a *failed run*, though nothing had
    # run; the gate answers before the run starts, naming the node and the feature.
    fd = front_door_check(STRUCTURED_WF, validate=False)  # isolate the capability gate
    assert not fd.ok
    assert fd.unsupported is not None
    assert "make_cups" in fd.unsupported
    assert "node_map" in fd.unsupported


@pytest.mark.parametrize(
    "document",
    [
        {"processes": ["not a mapping"]},
        {"processes": {"main": "not a mapping"}},
        {"processes": {"main": {"body": "not a mapping"}}},
        {"processes": {"main": {"body": {"nodes": "not a list"}}}},
        {"processes": {"main": {"body": {"nodes": ["not a mapping"]}}}},
        {"processes": {"main": {"body": {"nodes": [{"id": "n", "kind": {"a": 1}}]}}}},
    ],
)
def test_capability_gate_tolerates_a_malformed_document(document):
    # The gate is handed whatever the caller has -- including a document validate has
    # already found errors in, since `front_door_check` gates before weighing the
    # diagnostics -- so a shape it did not expect must not raise. Saying nothing leaves
    # the complaining to validate, which reports it with a position.
    assert capability_gate(document) is None


def test_front_door_expands_import(tmp_path):
    # A $import is no longer rejected: the front door resolves it and returns the
    # expanded document (no $import remains), which the gate then passes.
    (tmp_path / "shared.yaml").write_text(
        "gen:\n  kind: atomic\n  inputs: {}\n  outputs: {}\n", encoding="utf-8"
    )
    wf = tmp_path / "import.workflow.yaml"
    wf.write_text(IMPORT_WF, encoding="utf-8")
    fd = front_door_check(str(wf), validate=False)
    assert fd.ok
    assert fd.unsupported is None
    assert fd.document is not None
    assert "$import" not in fd.document["processes"]
    assert "gen" in fd.document["processes"]


def test_front_door_expansion_failure_is_a_diagnostic(tmp_path):
    # A structural $import failure (missing target) surfaces as a diagnostic with
    # no document -- not as an "unsupported" gate reason.
    wf = tmp_path / "import.workflow.yaml"
    wf.write_text(IMPORT_WF, encoding="utf-8")  # shared.yaml intentionally absent
    fd = front_door_check(str(wf), validate=False)
    assert not fd.ok
    assert fd.document is None
    assert fd.unsupported is None
    assert fd.diagnostics  # the unreadable-import failure is reported


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


def test_run_workflow_front_doors_an_in_memory_document():
    """`validate=True` with a mapping validates it, rather than refusing to."""
    pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")
    import yaml

    doc = yaml.safe_load(Path(SIMPLE_WF).read_text(encoding="utf-8"))
    result = run_workflow(doc, SIMPLE_ENV, random_seed=0, validate=True)
    assert not result.failed


def test_run_workflow_front_doors_a_malformed_in_memory_document():
    """And a rejection is the same `FrontDoorError` a file gets -- not a deep failure
    inside the runner, which is what happened while this route had no front door."""
    import yaml

    doc = yaml.safe_load(Path(SIMPLE_WF).read_text(encoding="utf-8"))
    doc["processes"]["main"]["body"]["nodes"][0]["process"] = "no_such_process"
    with pytest.raises(FrontDoorError) as exc:
        run_workflow(doc, SIMPLE_ENV, validate=True)
    assert not exc.value.result.ok
    assert exc.value.result.diagnostics


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
