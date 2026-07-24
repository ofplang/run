"""Python script processes (spec §22; dev-notes design.md D31).

Covers the three layers of `python_script_processes` support:

* the executor / device model in isolation -- `run_python_script` binds inputs and
  runs the code, and `script_device_model` verifies the result against the declared
  outputs (§22.2), raising `DeviceComputationError` on any mismatch;
* the simulator wiring -- a signed script operation completes with computed outputs,
  and a computation failure ends the operation `failed` (D25), not `completed`;
* the runner end to end -- a script workflow computes its whole-workflow outputs
  through the value layer, and a failing script gracefully stops the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ofplang.run.simulator import (
    DeviceComputationError,
    Simulator,
    default_device_model,
    run_python_script,
    script_device_model,
)

FIXTURES = Path(__file__).parent / "fixtures"
SCRIPT_WF = str(FIXTURES / "script.workflow.yaml")
SCRIPT_ENV = str(FIXTURES / "script.env.yaml")

# Value-shape descriptors (contracts.to_descriptor form) for the script's outputs.
INT = {"kind": "primitive", "name": "Int"}
STRING = {"kind": "primitive", "name": "String"}


# -- executor: run_python_script (§22.2) -------------------------------------


def test_run_python_script_binds_inputs_and_returns_mapping():
    # Each input port name is bound as a local; the code is the function body.
    result = run_python_script('return {"z": x + y}', {"x": 2, "y": 3})
    assert result == {"z": 5}


def test_run_python_script_wraps_script_exception():
    # An exception raised by the script becomes a graceful runtime failure (§22.2),
    # not a crash.
    with pytest.raises(DeviceComputationError):
        run_python_script('return {"z": 1 // 0}', {})


def test_run_python_script_wraps_syntax_error():
    # A code body that does not compile is a runtime verification failure too.
    with pytest.raises(DeviceComputationError):
        run_python_script("this is not python", {})


# -- device model: script_device_model verification (§22.2) ------------------

ADD_DEF = {
    "kind": "atomic",
    "inputs": {"x": {"type": "Int"}, "y": {"type": "Int"}},
    "outputs": {"z": {"type": "Int"}},
    "script": {"language": "python", "code": 'return {"z": x + y}'},
}


def test_script_device_model_computes_declared_outputs():
    out = script_device_model("add", "v0", {"x": 2, "y": 3}, {"z": INT}, ADD_DEF)
    assert out == {"z": 5}


def test_script_device_model_rejects_wrong_output_names():
    # The returned mapping must contain exactly the declared output names (§22.2).
    defn = {**ADD_DEF, "script": {"language": "python", "code": 'return {"wrong": x + y}'}}
    with pytest.raises(DeviceComputationError):
        script_device_model("add", "v0", {"x": 2, "y": 3}, {"z": INT}, defn)


def test_script_device_model_rejects_extra_output_name():
    defn = {**ADD_DEF, "script": {"language": "python", "code": 'return {"z": x + y, "extra": 1}'}}
    with pytest.raises(DeviceComputationError):
        script_device_model("add", "v0", {"x": 2, "y": 3}, {"z": INT}, defn)


def test_script_device_model_rejects_non_conformant_value():
    # A returned value that does not conform to the declared output type (§22.2).
    defn = {**ADD_DEF, "script": {"language": "python", "code": 'return {"z": "not-an-int"}'}}
    with pytest.raises(DeviceComputationError):
        script_device_model("add", "v0", {"x": 2, "y": 3}, {"z": INT}, defn)


def test_script_device_model_rejects_non_mapping_result():
    defn = {**ADD_DEF, "script": {"language": "python", "code": "return 5"}}
    with pytest.raises(DeviceComputationError):
        script_device_model("add", "v0", {"x": 2, "y": 3}, {"z": INT}, defn)


def test_script_device_model_rejects_unsupported_language():
    # Only `python` is a v0 script language; any other cannot be run as Python.
    defn = {**ADD_DEF, "script": {"language": "ruby", "code": "return {}"}}
    with pytest.raises(DeviceComputationError):
        script_device_model("add", "v0", {"x": 2, "y": 3}, {"z": INT}, defn)


def test_script_device_model_delegates_when_no_script():
    # A non-script process falls back to the type-default model, so a signed
    # non-script op behaves exactly as before this feature.
    defn = {"kind": "atomic", "inputs": {}, "outputs": {"z": {"type": "Int"}}}
    assert script_device_model("plain", "v0", {}, {"z": INT}, defn) == default_device_model(
        "plain", "v0", {}, {"z": INT}, defn
    )


# -- simulator wiring: computed outputs, and failure -> `failed` (D25) -------

# A device-less environment (no spots) for a single script processing op.
_SIM_ENV = {
    "time": {"unit": "second"},
    "devices": [{"id": "rack", "spots": ["slot"]}],
    "processes": {"add": {"modes": [{"id": "v0", "duration": 0}]}},
}


def test_simulator_reveals_computed_script_outputs_on_completion():
    # The built-in default (script_device_model) runs the script at completion; the
    # outputs are revealed only once the op is `completed` (D31).
    sim = Simulator(_SIM_ENV)
    uuid = sim.dispatch_processing(
        "add", "v0", output_schema={"z": INT}, inputs={"x": 4, "y": 5}, definition=ADD_DEF
    )
    sim.advance(0)  # duration 0 -> completes immediately
    assert sim.state(uuid) == {"status": "completed", "outputs": {"z": 9}}


def test_simulator_fails_operation_on_script_computation_error():
    # A script that fails runtime verification ends the op `failed` (no outputs), the
    # same graceful path as an injected capability failure (D25).
    bad = {**ADD_DEF, "script": {"language": "python", "code": 'return {"z": 1 // 0}'}}
    sim = Simulator(_SIM_ENV)
    uuid = sim.dispatch_processing(
        "add", "v0", output_schema={"z": INT}, inputs={"x": 4, "y": 5}, definition=bad
    )
    sim.advance(0)
    # The failed op also carries the model's (code, message) reason (D36).
    state = sim.state(uuid)
    assert state["status"] == "failed"
    assert state["reason"][0] == "script_error"


# -- runner end to end -------------------------------------------------------

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402


def _boundary(a, b):
    """A boundary supplying the Pure Data entry inputs `a` and `b` (no spots)."""
    return {"boundary": {"inputs": {"a": {"view": a}, "b": {"view": b}}}}


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_script_workflow_computes_whole_workflow_outputs(poll_interval):
    # The two chained script processes compute the whole-workflow outputs through the
    # value layer, with no injected device model (the built-in default runs them).
    runner = RollingRunner(SCRIPT_WF, SCRIPT_ENV, _boundary(2, 3), poll_interval=poll_interval, random_seed=0)
    status = runner.run()

    assert not runner.failed
    assert all(a["status"] == "completed" for a in status["activities"])
    # sum = 2 + 3 = 5; summary = "sum=5" (computed by `add` then `label`).
    assert runner.outputs == {"sum": 5, "summary": "sum=5"}
    # The result boundary echoes the computed output views (D28).
    assert runner.result_boundary["boundary"]["outputs"]["sum"]["view"] == 5


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_failing_script_stops_the_run_gracefully(poll_interval, tmp_path):
    # A script that returns the wrong output name fails runtime verification (§22.2).
    # The run stops gracefully (D25): the script op is `failed`, its downstream
    # consumer never starts and is `cancelled`, and `runner.failed` is set.
    wf = tmp_path / "bad_script.workflow.yaml"
    wf.write_text(
        SCRIPT_WORKFLOW_WITH_ADD_CODE.replace("__ADD_CODE__", 'return {"wrong": x + y}'),
        encoding="utf-8",
    )
    runner = RollingRunner(str(wf), SCRIPT_ENV, _boundary(2, 3), poll_interval=poll_interval, random_seed=0)
    status = runner.run()

    assert runner.failed
    statuses = {a.get("process"): a["status"] for a in status["activities"] if a.get("kind") == "processing"}
    assert statuses.get("add") == "failed"
    assert statuses.get("label") == "cancelled"


# A script workflow template whose `add` body is substituted per test. Mirrors
# script.workflow.yaml but with the `add` code as a placeholder, so a failure-mode
# variant can be written to a temp file without a committed fixture per case.
SCRIPT_WORKFLOW_WITH_ADD_CODE = """
spec_version: "0.0"
processes:
  add:
    kind: atomic
    inputs:
      x: { type: Int, phase: data }
      y: { type: Int, phase: data }
    outputs:
      z: { type: Int, phase: data }
    script:
      language: python
      code: |
        __ADD_CODE__
  label:
    kind: atomic
    inputs:
      n: { type: Int, phase: data }
    outputs:
      text: { type: String, phase: data }
    script:
      language: python
      code: |
        return {"text": "sum=" + str(n)}
  main:
    kind: composite
    inputs:
      a: { type: Int, phase: data }
      b: { type: Int, phase: data }
    outputs:
      sum:     { type: Int,    phase: data }
      summary: { type: String, phase: data }
    body:
      nodes:
        - id: Add
          process: add
          bind:
            x: { from: inputs.a }
            y: { from: inputs.b }
        - id: Label
          process: label
          bind:
            n: { from: Add.z }
      returns:
        sum:     { from: Add.z }
        summary: { from: Label.text }
entry: main
"""
