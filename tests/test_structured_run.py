"""Running a workflow with `map` / `fold` nodes over an Array of Objects (D59 R2).

The scheduler expands each structured node into its invocations before planning
(schedule D57), and the runner reads the same expanded graph: invocation `i` of a
node `N` is the activity at node path `(N, i, ...)`, and where its values come from is
the flattener's `Source` trees. An Array of Objects at the boundary is one Object per
element, each on a spot of its own: the boundary document binds the port to a list of
spots and, if it supplies them, a list of views in the same order.

The fixture is ofplang-schedule's `dispense_read` example, unchanged: a `fold` that
dispenses one shared reagent into each plate in turn, then a `map` that reads every
plate on one of two readers.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402
from ofplang.run.runner.runner import RunnerError  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
ENV = str(FIXTURES / "dispense_read.env.yaml")
SPOTS = ["hotel.a", "hotel.b", "hotel.c"]
RACK = ["rack.a", "rack.b", "rack.c"]


def _workflow(plate_view: bool = True) -> dict:
    """The example, with an `id` on each plate's view so a test can follow each one."""
    doc = yaml.safe_load((FIXTURES / "dispense_read.workflow.yaml").read_text(encoding="utf-8"))
    if plate_view:
        doc["types"]["Plate"]["view"] = {"id": {"type": "String"}}
    return doc


def _boundary(spots=SPOTS, views=None, rack=RACK, volume=2.5) -> dict:
    plates: dict = {"spot": spots}
    if views is not None:
        plates["view"] = views
    inputs: dict = {"reagent": {"spot": "shelf.r", "view": {}}, "plates": plates}
    if volume is not None:
        inputs["volume"] = {"view": volume}
    outputs: dict = {"plates": {"spot": rack}} if rack is not None else {}
    return {"boundary": {"inputs": inputs, "outputs": outputs}}


def _run(workflow, boundary, env=ENV):
    runner = RollingRunner(workflow, env, boundary, random_seed=0, observe=True)
    status = runner.run()
    return runner, status


def test_each_plate_is_followed_through_its_own_invocations():
    views = [{"id": "p0"}, {"id": "p1"}, {"id": "p2"}]
    runner, status = _run(_workflow(), _boundary(views=views))
    assert not runner.failed, runner.failure
    # The same makespan the scheduler's example plans for it.
    assert status["now"] == 59

    processed = {
        tuple(e["node"]): e for e in runner.observations if e.get("kind") == "processing"
    }
    assert set(processed) == {
        ("Dispense", 0), ("Dispense", 1), ("Dispense", 2), ("Read", 0), ("Read", 1), ("Read", 2)
    }
    # Invocation i gets element i: from the boundary for the fold, from the fold's
    # gathered output for the map.
    for i, view in enumerate(views):
        assert processed[("Dispense", i)]["inputs"]["plate"] == {"view": view}
        assert processed[("Read", i)]["inputs"]["plate"] == {"view": view}
    # The run-phase volume reaches every dispense.
    assert all(
        processed[("Dispense", i)]["inputs"]["volume"] == {"view": 2.5} for i in range(3)
    )

    # The gathered outputs come back in element order.
    assert runner.outputs["plates"] == views
    assert runner.outputs["ods"] == [0.0, 0.0, 0.0]

    # A transport that carries one plate is handed that plate's view, not the Array.
    plate_moves = [
        e["moved"]["view"] for e in runner.observations
        if e.get("kind") == "transport" and e["moved"]["view"] != {}  # {} is the reagent
    ]
    assert plate_moves and all(m in views for m in plate_moves)
    assert {m["id"] for m in plate_moves} == {"p0", "p1", "p2"}


def test_the_result_boundary_lists_one_view_per_spot():
    views = [{"id": "p0"}, {"id": "p1"}, {"id": "p2"}]
    runner, _ = _run(_workflow(), _boundary(views=views))
    out = runner.result_boundary["boundary"]["outputs"]
    assert out["plates"] == {"spot": RACK, "view": views}
    assert out["ods"] == {"view": [0.0, 0.0, 0.0]}
    # Fed back in, the inputs echo as given.
    assert runner.result_boundary["boundary"]["inputs"]["plates"] == {"spot": SPOTS, "view": views}


def test_a_shorter_array_is_planned_and_run_at_its_own_length():
    runner, status = _run(_workflow(), _boundary(
        spots=SPOTS[:2], views=[{"id": "a"}, {"id": "b"}], rack=RACK[:2]
    ))
    assert not runner.failed, runner.failure
    nodes = {tuple(a["node"]) for a in status["activities"] if a.get("kind") == "processing"}
    assert nodes == {("Dispense", 0), ("Dispense", 1), ("Read", 0), ("Read", 1)}
    assert runner.outputs["plates"] == [{"id": "a"}, {"id": "b"}]


def _read_only() -> dict:
    """The example without the dispensing fold: every plate is read, nothing else."""
    doc = _workflow()
    main = doc["processes"]["main"]
    main["body"]["nodes"] = [n for n in main["body"]["nodes"] if n["id"] == "Read"]
    main["body"]["nodes"][0]["each"]["plate"] = {"from": "inputs.plates"}
    del main["inputs"]["reagent"], main["inputs"]["volume"], main["outputs"]["reagent"]
    del main["body"]["returns"]["reagent"]
    return doc


def test_no_plates_is_no_invocation():
    # L = 0: nothing is invoked, and the gathered outputs are empty Arrays.
    boundary = {"boundary": {"inputs": {"plates": {"spot": [], "view": []}}}}
    runner, status = _run(_read_only(), boundary)
    assert not runner.failed, runner.failure
    assert [a for a in status["activities"] if a.get("kind") == "processing"] == []
    assert runner.outputs == {"plates": [], "ods": []}
    assert runner.warnings == []


def test_no_plates_through_a_fold_carrying_an_object_returns_it_untouched():
    # With no plates the fold hands its carried reagent straight through: the final
    # `reagent` is the entry Object returned as it came, which the scheduler plans as a
    # through arc (schedule D60 J2) -- it used to refuse the whole run for it. The
    # run reports it with the view it came in with, and nothing is warned about.
    boundary = _boundary(spots=[], views=[], rack=None)
    boundary["boundary"]["inputs"]["reagent"]["view"] = {}
    boundary["boundary"]["outputs"]["reagent"] = {"spot": "shelf.r"}
    runner, status = _run(_workflow(), boundary)
    assert not runner.failed, runner.failure
    assert runner.outputs == {"reagent": {}, "plates": [], "ods": []}
    assert runner.warnings == []
    assert runner.result_boundary["boundary"]["outputs"]["reagent"] == {
        "spot": "shelf.r", "view": {}
    }


def test_views_left_out_are_one_default_per_spot_and_said():
    runner, _ = _run(_workflow(), _boundary(views=None))
    assert not runner.failed, runner.failure
    assert runner.outputs["plates"] == [{"id": ""}] * 3
    defaulted = [w for w in runner.warnings if w.code == "entry_input_defaulted"]
    assert len(defaulted) == 1 and "'plates'" in defaulted[0].message


def test_an_entry_input_left_out_is_said():
    runner, _ = _run(_workflow(), _boundary(views=[{"id": "x"}] * 3, volume=None))
    assert not runner.failed, runner.failure
    assert [(w.code, "'volume'" in w.message) for w in runner.warnings] == [
        ("entry_input_defaulted", True)
    ]


def test_views_not_one_per_spot_are_refused():
    with pytest.raises(RunnerError, match="shape of its spots"):
        _run(_workflow(), _boundary(views=[{"id": "x"}, {"id": "y"}]))


# -- lengths only a value shows (schedule `LengthCheck`) ---------------------------


def _zipped(source: str) -> dict:
    """The example with a per-plate gain zipped into the reads: `gains` comes either
    from the boundary (`inputs.gains`, a run-phase value) or from an atomic `Plan`
    that produces it (a data-phase value). Either way its length is the value's, and
    the plan assumed it equals the number of plates."""
    doc = _workflow()
    doc["processes"]["read"]["inputs"]["gain"] = {"type": "Float", "phase": "data"}
    main = doc["processes"]["main"]
    main["inputs"]["gains"] = {"type": "Array<Float>", "phase": "run"}
    doc["processes"]["plan"] = {
        "kind": "atomic",
        "outputs": {"gains": {"type": "Array<Float>", "phase": "data"}},
    }
    nodes = main["body"]["nodes"]
    if source == "produced":
        nodes.insert(0, {"id": "Plan", "process": "plan"})
        del main["inputs"]["gains"]
    read = next(n for n in nodes if n["id"] == "Read")
    read["each"]["gain"] = {"from": "inputs.gains" if source == "boundary" else "Plan.gains"}
    return doc


def test_a_boundary_value_of_the_wrong_length_refuses_the_run_before_it_starts():
    # The run states the gains' length (`expansion`, schedule D62), so the scheduler
    # sees three plates zipped with two gains before anything is planned: the boundary
    # contradicts itself, and the run is refused before it starts (D62 H1). Until then
    # this stopped the job at its first preflight.
    boundary = _boundary(views=[{"id": "x"}] * 3)
    boundary["boundary"]["inputs"]["gains"] = {"view": [1.0, 2.0]}
    with pytest.raises(RunnerError, match="each_length_mismatch"):
        _run(_zipped("boundary"), boundary)


def test_a_boundary_value_of_the_right_length_runs():
    boundary = _boundary(views=[{"id": "x"}] * 3)
    boundary["boundary"]["inputs"]["gains"] = {"view": [1.0, 2.0, 3.0]}
    runner, _ = _run(_zipped("boundary"), boundary)
    assert not runner.failed, runner.failure
    reads = {
        tuple(e["node"]): e["inputs"]["gain"]["view"]
        for e in runner.observations
        if e.get("kind") == "processing" and e["node"][0] == "Read"
    }
    assert reads == {("Read", 0): 1.0, ("Read", 1): 2.0, ("Read", 2): 3.0}


def test_a_produced_value_of_the_wrong_length_stops_the_job_when_it_is_produced(tmp_path):
    # The simulator's `plan` returns its type's default, an empty Array: a data error
    # of the job, found when `Plan` completes and before any read is dispatched.
    env = yaml.safe_load(Path(ENV).read_text(encoding="utf-8"))
    env["devices"].append({"id": "planner", "spots": []})
    env["processes"]["plan"] = {"modes": [{"devices": ["planner"], "duration": 1}]}
    env_path = tmp_path / "env.yaml"
    env_path.write_text(yaml.safe_dump(env), encoding="utf-8")
    runner, status = _run(
        _zipped("produced"), _boundary(views=[{"id": "x"}] * 3), env=str(env_path)
    )
    assert runner.failed
    assert runner.failure.kind == "each_length_mismatch"
    assert "no elements" not in runner.failure.detail
    assert "0 elements" in runner.failure.detail
    started = {
        tuple(a["node"])
        for a in status["activities"]
        if a.get("kind") == "processing" and a.get("status") in ("completed", "running", "failed")
    }
    assert not any(node[0] == "Read" for node in started)


def test_the_workflow_document_is_not_changed_by_running_it():
    doc = _workflow()
    before = copy.deepcopy(doc)
    _run(doc, _boundary(views=[{"id": "x"}] * 3))
    assert doc == before


def test_the_elements_of_one_boundary_port_are_distinct_arcs():
    # Each element of an Array of Objects crosses the boundary on an arc of its own,
    # and on the boundary side those arcs differ only in their `index`. The runner
    # matches committed history against each new plan by this key, so two elements
    # keyed alike would have one's move mark the other's done.
    from ofplang.run.runner.provenance import activity_key

    def leg(i):
        return {
            "kind": "transport",
            "arc": {"from": {"node": [], "port": "plates", "index": [i]},
                    "to": {"node": ["Heat", i], "port": "plate"}},
        }

    def same_consumer(i):
        move = leg(i)
        move["arc"]["to"] = {"node": ["Heat"], "port": "plate"}
        return move

    assert activity_key(leg(0)) != activity_key(leg(1))
    assert activity_key(same_consumer(0)) != activity_key(same_consumer(1))


# -- an entry Object returned untouched (schedule D60 J2) --------------------------------

_PASS_WF = """\
spec_version: "0.5"
types: {Plate: {domain: object, view: {id: {type: String}}}}
processes:
  heat:
    kind: atomic
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}}
    objects: {map: {outputs.plate: inputs.plate}}
  main:
    kind: composite
    inputs: {a: {type: Plate, phase: data}, b: {type: Plate, phase: data}}
    outputs: {a: {type: Plate, phase: data}, b: {type: Plate, phase: data}}
    body:
      nodes:
        - {id: H, process: heat, state: {plate: {from: inputs.a}}}
      returns: {a: {from: H.plate}, b: {from: inputs.b}}
entry: main
"""

_PASS_ENV = """\
time: {unit: second}
devices:
  - {id: hotel, spots: [a, b]}
  - {id: oven, spots: [tray]}
  - {id: rack, spots: [a, b]}
transporters: [{id: arm}]
transports:
  - {transporter: arm, from: hotel.a, to: oven.tray, duration: 2}
  - {transporter: arm, from: oven.tray, to: rack.a, duration: 2}
  - {transporter: arm, from: hotel.b, to: rack.b, duration: 3}
processes:
  heat:
    modes:
      - {devices: [oven], duration: 5, input_spots: {plate: oven.tray},
         output_spots: {plate: oven.tray}}
"""


def test_an_object_returned_untouched_is_moved_and_reported(tmp_path):
    env = tmp_path / "env.yaml"
    env.write_text(_PASS_ENV, encoding="utf-8")
    boundary = {"boundary": {
        "inputs": {"a": {"spot": "hotel.a", "view": {"id": "A"}},
                   "b": {"spot": "hotel.b", "view": {"id": "B"}}},
        "outputs": {"a": {"spot": "rack.a"}, "b": {"spot": "rack.b"}},
    }}
    runner, _ = _run(yaml.safe_load(_PASS_WF), boundary, env=str(env))
    assert not runner.failed, runner.failure
    # `b` crossed the boundary in and out untouched: carried to its output spot (the
    # delivery check found it there), with the view it came in with.
    assert runner.outputs == {"a": {"id": "A"}, "b": {"id": "B"}}
    moved = [e["moved"]["view"] for e in runner.observations
             if e.get("kind") == "transport" and e["arc"]["to"] == {"node": [], "port": "b"}]
    assert moved == [{"id": "B"}]
