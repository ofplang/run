"""The run states the lengths of the lists it was given (schedule SPEC §6.13, D62).

A `map` / `fold` over a Pure Data entry input -- a list of labels -- is as many
invocations as the list has elements, and the scheduler never sees the list. The run
does: it counts every Pure Data Array entry input and states the counts in the
document's `expansion` (top-level for one workflow, per roster entry for several), and
flattens its own copy of the workflow with the same counts so the two graphs agree.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import JobRequest, RollingRunner, load_document  # noqa: E402
from ofplang.run.runner.boundary import parse_boundary  # noqa: E402
from ofplang.run.runner.contracts import Contracts  # noqa: E402
from ofplang.run.runner.job import build_job, expansion_of  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
# `main` maps `make` over `labels: Array<String>`, one Cup per label.
LABELS_WF = FIXTURES / "structured_node.workflow.yaml"

MAKER_ENV = {
    "time": {"unit": "second"},
    "devices": [{"id": "maker", "spots": ["out"]}, {"id": "shelf", "spots": ["a", "b", "c"]}],
    "transporters": [{"id": "arm"}],
    "transports": [
        {"transporter": "arm", "from": "maker.out", "to": f"shelf.{s}", "duration": 1}
        for s in ("a", "b", "c")
    ],
    "processes": {
        "make": {"modes": [{"devices": ["maker"], "duration": 2,
                            "output_spots": {"cup": "maker.out"}}]},
    },
}

# Every kind of entry input, to say which ones get a length.
MIXED_WF = """\
spec_version: "0.0"
types:
  Plate: {domain: object}
processes:
  use:
    kind: atomic
    inputs:
      labels: {type: Array<String>, phase: data}
      volumes: {type: Array<Float>, phase: data}
      rows: {type: Array<Array<String>>, phase: data}
      plates: {type: Array<Plate>, phase: data}
      t: {type: Float, phase: data}
    outputs:
      plates: {type: Array<Plate>, phase: data}
    objects: {map: {outputs.plates: inputs.plates}}
entry: use
"""


def _labels(*values) -> dict:
    return {"boundary": {"inputs": {"labels": {"view": list(values)}}}}


def _made(status, job=None) -> list:
    return sorted(
        a["node"] for a in status["activities"]
        if a["kind"] == "processing" and a.get("status") == "completed"
        and (job is None or a.get("job") == job)
    )


# -- what is stated ------------------------------------------------------------


def test_every_pure_data_array_entry_input_gets_its_length():
    workflow = yaml.safe_load(MIXED_WF)
    contracts = Contracts.from_workflow(workflow)
    boundary = parse_boundary(
        {"boundary": {"inputs": {"labels": {"view": ["a", "b"]},
                                 "rows": {"view": [["x"], ["y", "z"], []]},
                                 "plates": {"spot": "maker.out"}}}},
        contracts,
    )
    # In declaration order. `volumes` was not supplied, so it runs on its default --
    # an empty list -- and that is its length. `rows` gets its outermost length only.
    # Nothing for the Array of Objects (its length is its spots') or the scalar.
    assert expansion_of(contracts, boundary) == {"lengths": [
        {"node": [], "port": "labels", "length": 2},
        {"node": [], "port": "volumes", "length": 0},
        {"node": [], "port": "rows", "length": 3},
    ]}


def test_a_value_that_is_not_a_list_has_no_length_to_state():
    contracts = Contracts.from_workflow(yaml.safe_load(MIXED_WF))
    boundary = parse_boundary({"boundary": {"inputs": {"labels": {"view": "abc"}}}}, contracts)
    lengths = expansion_of(contracts, boundary)["lengths"]
    assert [e["port"] for e in lengths] == ["volumes", "rows"]


def test_a_workflow_with_no_pure_data_array_states_nothing():
    contracts = Contracts.from_workflow(load_document(FIXTURES / "simple.workflow.yaml"))
    assert expansion_of(contracts, parse_boundary(None, contracts)) is None


def test_the_job_carries_it_and_its_roster_entry_says_it():
    job = build_job(load_document(LABELS_WF), _labels("a", "b"), id="j")
    assert job.expansion == {"lengths": [{"node": [], "port": "labels", "length": 2}]}
    assert job.roster_entry()["expansion"] == job.expansion
    # The run's own graph is the one the plan will name, and the run still checks the
    # length it stated against the value.
    assert sorted(job.dataflow.process_of) == [("make_cups", 0), ("make_cups", 1)]
    assert [c.length for c in job.dataflow.length_checks] == [2]


# -- runs ---------------------------------------------------------------------


def test_a_single_workflow_runs_once_per_label_and_reports_the_lengths():
    runner = RollingRunner(load_document(LABELS_WF), copy.deepcopy(MAKER_ENV),
                           _labels("a", "b", "c"), poll_interval=None, random_seed=0)
    status = runner.run()
    assert not runner.failed, runner.failure
    assert _made(status) == [["make_cups", i] for i in range(3)]
    assert status["expansion"] == {"lengths": [{"node": [], "port": "labels", "length": 3}]}


def test_each_job_expands_by_its_own_list():
    workflow = load_document(LABELS_WF)
    runner = RollingRunner(
        [JobRequest("two", copy.deepcopy(workflow), _labels("a", "b")),
         JobRequest("one", copy.deepcopy(workflow), _labels("c"))],
        copy.deepcopy(MAKER_ENV), poll_interval=None, random_seed=0,
    )
    status = runner.run()
    assert not runner.failed, runner.failure
    assert _made(status, "two") == [["make_cups", 0], ["make_cups", 1]]
    assert _made(status, "one") == [["make_cups", 0]]
    assert [e["expansion"]["lengths"][0]["length"] for e in status["jobs"]] == [2, 1]
    assert "expansion" not in status


def test_an_arriving_job_brings_its_lengths():
    workflow = load_document(LABELS_WF)
    runner = RollingRunner(
        [JobRequest("first", copy.deepcopy(workflow), _labels("a"))],
        copy.deepcopy(MAKER_ENV), poll_interval=None, random_seed=0,
    )
    original = runner.sim.advance
    admitted: list[int] = []

    def advance(until):
        if not admitted and runner.now >= 1:
            admitted.append(runner.now)
            runner.admit(JobRequest("late", copy.deepcopy(workflow), _labels("b", "c")))
        return original(until)

    runner.sim.advance = advance  # type: ignore[method-assign]
    status = runner.run()
    assert admitted and not runner.failed, runner.failure
    assert _made(status, "late") == [["make_cups", 0], ["make_cups", 1]]
    late = next(e for e in status["jobs"] if e["id"] == "late")
    assert late["expansion"] == {"lengths": [{"node": [], "port": "labels", "length": 2}]}
