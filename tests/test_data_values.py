"""Integration smoke for the value layer (dev-notes design.md D26-3).

The rolling runner drives a workflow with a nested composite and a non-empty
`returns` to completion, dispatching each processing with its output-port
signature (the value seam) and recording the values the backend produces. This
test checks the run completes and the whole-workflow output is assembled from the
produced values, including one that crosses a nested composite boundary. The
producer -> consumer *wiring* correctness is pinned by the D26-1 unit tests
(tests/test_dataflow.py); here we confirm the pieces work together end to end.

The workflow (tests/fixtures/nested_returns.workflow.yaml) reuses the pure_data
environment/interface and returns `final_score` from an `analyze` wrapped in a
composite, so the final output is produced inside a nested boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner, RunnerError, load_document  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
WF = str(FIXTURES / "nested_returns.workflow.yaml")
ENV = str(FIXTURES / "pure_data.env.yaml")


def _boundary(sample=None, extra_inputs=None):
    """A boundary doc pinning `sample` on the loader (the pure_data interface). An
    optional `sample` view value is supplied (the job); `extra_inputs` adds raw port
    descriptors (used to exercise the unknown-port error)."""
    sample_desc = {"spot": "loader.stage"}
    if sample is not None:
        sample_desc["view"] = sample
    inputs = {"sample": sample_desc}
    if extra_inputs:
        inputs.update(extra_inputs)
    return {"boundary": {"inputs": inputs}}


def _count_boundary(value):
    """A boundary doc supplying the Pure Data `start` value for count_chain (no spot)."""
    return {"boundary": {"inputs": {"start": {"view": {"value": value}}}}}


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_value_layer_produces_whole_workflow_output(poll_interval):
    runner = RollingRunner(WF, ENV, _boundary(), poll_interval=poll_interval, random_seed=0)
    status = runner.run()

    # The run completes normally.
    assert all(a["status"] == "completed" for a in status["activities"])
    assert not runner.failed

    # The whole-workflow output is assembled from a produced value. `final_score`
    # is returned from `analyze`, which sits inside the `Analyzer` composite, so the
    # value crossed a nested boundary to reach the output.
    assert set(runner.outputs) == {"final_score"}
    produced = runner.values.get(("Az", "A"), "score")  # the nested producer's value
    assert runner.outputs["final_score"] == produced     # the return follows it out
    # `Score` declares no view, so its typed value is an empty record (F2 typed
    # defaults; view-ful types are exercised in test_typed_values_are_view_shaped).
    assert produced == {}

    # Every value-producing processing recorded its outputs in the store; `Finish`
    # produces nothing (empty signature), so it records nothing.
    snapshot = runner.values.snapshot()
    assert (("Measure",), "plate_out") in snapshot
    assert (("Measure",), "reading") in snapshot
    assert (("Az", "A"), "score") in snapshot


def test_typed_values_are_view_shaped():
    # With view-ful Pure Data types, the backend generates non-empty records shaped
    # by each type's view schema (F2 typed defaults). `final_score` is a `Score`
    # (view {value: Float, ok: Bool}) returned from the nested analyze.
    runner = RollingRunner(
        str(FIXTURES / "typed_returns.workflow.yaml"), ENV, _boundary(), random_seed=0,
    )
    status = runner.run()
    assert all(a["status"] == "completed" for a in status["activities"])
    assert runner.outputs == {"final_score": {"value": 0.0, "ok": False}}
    # The intermediate reading is a view record too.
    assert runner.values.get(("Measure",), "reading") == {"mean": 0.0, "n": 0}


TYPED_WF = str(FIXTURES / "typed_returns.workflow.yaml")


def test_boundary_supplies_and_defaults_entry_values():
    # A supplied boundary view seeds the entry input (contract-checked); an unsupplied
    # one falls back to a typed default. `sample` is a Plate with view {barcode}.
    with_view = RollingRunner(TYPED_WF, ENV, _boundary({"barcode": "ABC"}), random_seed=0)
    with_view.run()
    assert with_view.values.get((), "sample") == {"barcode": "ABC"}

    without_view = RollingRunner(TYPED_WF, ENV, _boundary(), random_seed=0)
    without_view.run()
    assert without_view.values.get((), "sample") == {"barcode": ""}  # typed default


def test_boundary_rejects_nonconformant_and_unknown_entries():
    # A non-conformant view value is rejected at run (the seed conformance check); an
    # unknown entry port is rejected up front, at boundary parse (construction).
    bad_value = RollingRunner(TYPED_WF, ENV, _boundary({"barcode": 1}), random_seed=0)
    with pytest.raises(RunnerError):
        bad_value.run()

    with pytest.raises(RunnerError):
        RollingRunner(TYPED_WF, ENV, _boundary(extra_inputs={"nope": {"view": {"barcode": "X"}}}), random_seed=0)


COUNT_WF = str(FIXTURES / "count_chain.workflow.yaml")
COUNT_ENV = str(FIXTURES / "count_chain.env.yaml")


def _echo_inc(process, mode, inputs, output_schema, definition):
    """A device model for `inc`: echo the input `x` to every output port."""
    return {port: inputs["x"] for port in output_schema}


def test_device_model_propagates_job_value_end_to_end():
    # With a device model, a supplied job value flows through the backend, down the
    # two-step chain, to the returned output (D27 F4b) -- end-to-end propagation.
    runner = RollingRunner(
        COUNT_WF, COUNT_ENV, _count_boundary(42), device_model=_echo_inc, random_seed=0,
    )
    status = runner.run()
    assert all(a["status"] == "completed" for a in status["activities"])
    assert runner.outputs == {"result": {"value": 42}}
    # It reached the output via each step, not by chance.
    assert runner.values.get(("S1",), "y") == {"value": 42}
    assert runner.values.get(("S2",), "y") == {"value": 42}


def test_device_model_receives_the_process_definition():
    # The runner passes each process's raw definition (kind/inputs/outputs/objects)
    # to the device model, so a model can act on the declared structure (D27 F4b).
    seen = {}

    def capture(process, mode, inputs, output_schema, definition):
        seen[process] = definition
        return {port: inputs["x"] for port in output_schema}  # echo the Count

    runner = RollingRunner(COUNT_WF, COUNT_ENV, _count_boundary(1), device_model=capture, random_seed=0)
    runner.run()
    assert "inc" in seen
    definition = seen["inc"]
    assert definition["kind"] == "atomic"
    assert set(definition["inputs"]) == {"x"} and set(definition["outputs"]) == {"y"}


LITERAL_WF = str(FIXTURES / "literal_chain.workflow.yaml")
LITERAL_ENV = str(FIXTURES / "literal_chain.env.yaml")


def test_static_literal_reaches_backend_and_output():
    # A static literal `bind: {seed: {value: 7}}` (§11 / D30) is assembled as the
    # input the backend receives (no producer, no boundary job), and -- with an echo
    # model -- flows to the whole-workflow output. This exercises the full path:
    # schedule flattener -> data_literals -> dataflow.literals -> assemble_inputs ->
    # dispatch, with node paths matching the plan at runtime.
    seen = {}

    def echo(process, mode, inputs, output_schema, definition):
        seen.update(inputs)
        return {port: inputs["seed"] for port in output_schema}

    runner = RollingRunner(LITERAL_WF, LITERAL_ENV, device_model=echo, random_seed=0)
    status = runner.run()
    assert all(a["status"] == "completed" for a in status["activities"])
    assert seen["seed"] == 7              # the literal was assembled and dispatched
    assert runner.outputs == {"result": 7}  # and reaches the whole-workflow output


def test_static_literal_supplied_even_without_device_model():
    # Without a device model the default computes a typed default output (the literal
    # does not propagate downstream), but it is still assembled and handed to the
    # backend -- the literal mechanism does not depend on a model.
    seen = {}

    def capture(process, mode, inputs, output_schema, definition):
        seen.update(inputs)
        from ofplang.run.simulator import default_device_model

        return default_device_model(process, mode, inputs, output_schema, definition)

    runner = RollingRunner(LITERAL_WF, LITERAL_ENV, device_model=capture, random_seed=0)
    runner.run()
    assert seen["seed"] == 7                 # literal reached the backend
    assert runner.outputs == {"result": 0}   # default model ignores it -> typed default


def test_without_device_model_computed_outputs_are_defaults():
    # No model -> the built-in default applies. `inc` declares no objects.map, so its
    # outputs are type defaults; the job value is not computed through (result is the
    # Count view default {value: 0}).
    runner = RollingRunner(COUNT_WF, COUNT_ENV, _count_boundary(42), random_seed=0)
    runner.run()
    assert runner.outputs == {"result": {"value": 0}}


def test_default_carries_mapped_object_without_a_device_model():
    # The built-in default carries a mapped Object output even with no device model:
    # measure's plate_out is `objects.map: {outputs.plate_out: inputs.plate}`, so the
    # supplied plate flows through to it (its view value, {barcode: "XY"}).
    runner = RollingRunner(TYPED_WF, ENV, _boundary({"barcode": "XY"}), random_seed=0)
    runner.run()
    assert runner.values.get(("Measure",), "plate_out") == {"barcode": "XY"}


def test_device_model_output_is_contract_checked():
    # A model that returns a non-conformant value is caught at poll (the F4a
    # conformance check, now live under a device model).
    def bad_model(process, mode, inputs, output_schema, definition):
        return {port: "not-an-int-record" for port in output_schema}

    runner = RollingRunner(COUNT_WF, COUNT_ENV, device_model=bad_model, random_seed=0)
    with pytest.raises(RunnerError):
        runner.run()


def test_cli_writes_result_boundary(tmp_path):
    # `--boundary-out` writes the result boundary document (D28): the same schema as
    # --boundary, with each produced output's `view` filled in; a run-local artifact,
    # separate from the status document. `final_score` is a Pure Data output (no spot).
    import yaml

    from ofplang.run.cli import main

    boundary_in = tmp_path / "boundary.yaml"
    boundary_in.write_text(
        yaml.safe_dump({"boundary": {"inputs": {"sample": {"spot": "loader.stage"}}}}),
        encoding="utf-8",
    )
    out = tmp_path / "boundary_out.yaml"
    code = main([
        "run", TYPED_WF, "--env", ENV,
        "--boundary", str(boundary_in),
        "--boundary-out", str(out),
    ])
    assert code == 0
    assert load_document(out) == {
        "boundary": {
            "inputs": {"sample": {"spot": "loader.stage"}},
            "outputs": {"final_score": {"view": {"value": 0.0, "ok": False}}},
        }
    }


def test_value_layer_is_deterministic():
    # Typed defaults are deterministic, so both poll modes agree on the output.
    a = RollingRunner(WF, ENV, _boundary(), poll_interval=None, random_seed=0)
    b = RollingRunner(WF, ENV, _boundary(), poll_interval=1, random_seed=0)
    a.run()
    b.run()
    assert a.outputs == b.outputs


# -- integration edge cases across the runner's existing behaviors -----------

SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")
REROUTE_ENV = str(FIXTURES / "reroute.env.yaml")
FAIL_WF = str(FIXTURES / "failure.workflow.yaml")
FAIL_ENV = str(FIXTURES / "failure.env.yaml")


def test_object_entry_and_object_return_end_to_end():
    # An Object entry input and an Object return: the final output follows the
    # return back to the producing atomic's recorded value.
    # The boundary pins the Object entry on the loader and the Object return on the
    # output rack (so the run-end delivery check, P3, applies to `result`).
    boundary = {
        "boundary": {
            "inputs": {"sample": {"spot": "loader.stage"}},
            "outputs": {"result": {"spot": "output.slot"}},
        }
    }
    runner = RollingRunner(
        str(FIXTURES / "interface_load.workflow.yaml"),
        str(FIXTURES / "interface_load.env.yaml"),
        boundary,
        random_seed=0,
    )
    status = runner.run()
    assert all(a["status"] == "completed" for a in status["activities"])
    assert set(runner.outputs) == {"result"}
    assert runner.outputs["result"] == runner.values.get(("Heat",), "out")
    # The entry input was seeded at the boundary as a typed default (Plate declares
    # no view -> empty record).
    assert runner.values.get((), "sample") == {}


def test_output_spot_delivery_check_flags_empty_spot():
    # P3 (D28): a pinned Object output that is not on its declared spot at run end is
    # an inconsistency, raised. A successful run with a pinned output always delivers
    # (the §6.8 interface_out node holds the spot), so this exercises the guard
    # directly: point the check at a valid but unoccupied spot and confirm it fires.
    import dataclasses

    runner = RollingRunner(
        str(FIXTURES / "interface_load.workflow.yaml"),
        str(FIXTURES / "interface_load.env.yaml"),
        random_seed=0,
    )
    # output.slot is empty before any delivery; claim `result` should be there.
    runner.boundary = dataclasses.replace(runner.boundary, output_spots={"result": "output.slot"})
    with pytest.raises(RunnerError, match="did not reach its declared spot"):
        runner._check_output_spots()


def test_create_workflow_records_producer_but_has_no_outputs():
    # simple.workflow: a CREATE source, no entry input, empty returns. The run
    # completes, the created value is recorded, and there is no whole-workflow output.
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0)
    status = runner.run()
    assert all(a["status"] == "completed" for a in status["activities"])
    assert runner.outputs == {}
    assert runner.values.has(("SampleSource",), "source_out")
    # The pure-consume target produced nothing.
    assert not runner.values.has(("SampleTarget",), "target_in")


def test_values_survive_reroute():
    # A device goes down and the run re-routes (the plan and its spots change), but
    # values are keyed by workflow node path, so the producer's value is unaffected.
    runner = RollingRunner(SIMPLE_WF, REROUTE_ENV, random_seed=0)
    runner.sim.schedule_device_down(3, "station_1")
    status = runner.run()
    assert status["now"] == 9  # re-routed makespan
    # The value keys are workflow node paths, unaffected by the re-route; `Sample`
    # declares no view, so the produced value is an empty record.
    assert runner.values.has(("SampleSource",), "source_out")
    assert runner.values.get(("SampleSource",), "source_out") == {}


def test_values_under_duration_variance():
    # With duration variance (a running-task margin is required), the run still
    # completes and records its produced values.
    runner = RollingRunner(
        SIMPLE_WF, SIMPLE_ENV, random_seed=0,
        poll_interval=1, running_task_margin=2, duration_model=lambda a, planned: planned + 1,
    )
    runner.run()
    assert runner.values.has(("SampleSource",), "source_out")


def test_values_on_failure_are_partial():
    # When an activity fails, the run stops: producers that completed before the
    # failure keep their values; the failed op produces none, and abandoned pending
    # work never records anything. (The slow chain drains, so its source completes.)
    runner = RollingRunner(FAIL_WF, FAIL_ENV, poll_interval=None, random_seed=0)
    runner.sim.schedule_process_failure("src_bad", "m0")
    runner.run()
    assert runner.failed
    assert not runner.values.has(("SrcBad",), "out")   # failed -> no value
    assert runner.values.has(("SrcSlow",), "out")      # completed while draining
    assert runner.outputs == {}                        # no returns in this workflow
