"""A `map` / `fold` whose length is a value made during the run (schedule design.md D70).

An inspection says which wells to redo, and a traversal redoes each. The plan is made on
one invocation, assumed; the run counts the list once the inspection has made it,
states the count in `expansion.lengths`, and replans before anything of the traversal
is dispatched -- so a count of none runs nothing that was assumed, and a count of three
runs three.
"""

from __future__ import annotations

import copy

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import JobRequest, RollingRunner  # noqa: E402
from ofplang.run.simulator import default_device_model  # noqa: E402

_STEPS = """\
  inspect:
    kind: atomic
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}, redo: {type: "Array<Int>", phase: data}}
    objects: {map: {outputs.plate: inputs.plate}}
  prep:
    kind: atomic
    inputs: {}
    outputs: {ready: {type: Bool, phase: data}}
  fix:
    kind: atomic
    inputs: {well: {type: Int, phase: data}, ready: {type: Bool, phase: data}}
    outputs: {ok: {type: Bool, phase: data}}
  fix_well:
    kind: composite
    inputs: {well: {type: Int, phase: data}}
    outputs: {ok: {type: Bool, phase: data}}
    body:
      nodes:
        - {id: P, process: prep}
        - {id: F, process: fix, bind: {well: {from: inputs.well}, ready: {from: P.ready}}}
      returns: {ok: {from: F.ok}}
  wash:
    kind: atomic
    inputs: {plate: {type: Plate, phase: data}, well: {type: Int, phase: data}}
    outputs: {plate: {type: Plate, phase: data}}
    objects: {map: {outputs.plate: inputs.plate}}
  wash_well:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}, well: {type: Int, phase: data}}
    outputs: {plate: {type: Plate, phase: data}}
    body:
      nodes:
        - {id: W, process: wash, state: {plate: {from: inputs.plate}},
           bind: {well: {from: inputs.well}}}
      returns: {plate: {from: W.plate}}
  store:
    kind: atomic
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}}
    objects: {map: {outputs.plate: inputs.plate}}
"""


def _workflow(main: str) -> dict:
    return yaml.safe_load(
        'spec_version: "0.5"\ntypes:\n  Plate: {domain: object}\nprocesses:\n'
        + _STEPS + main + "entry: main\n"
    )


REDO = _workflow("""\
  main:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}, oks: {type: "Array<Bool>", phase: data}}
    body:
      nodes:
        - {id: I, process: inspect, state: {plate: {from: inputs.plate}}}
        - {id: R, kind: map, process: fix_well, each: {well: {from: I.redo}}}
        - {id: S, process: store, state: {plate: {from: I.plate}}}
      returns: {plate: {from: S.plate}, oks: {from: R.ok}}
""")

WASH = _workflow("""\
  main:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}}
    body:
      nodes:
        - {id: I, process: inspect, state: {plate: {from: inputs.plate}}}
        - {id: W, kind: fold, process: wash_well, carry: {plate: {from: I.plate}},
           each: {well: {from: I.redo}}}
        - {id: S, process: store, state: {plate: {from: W.plate}}}
      returns: {plate: {from: S.plate}}
""")

_SPOTS = ["rack.in", "rack.out", "reader.slot", "sink.basin"]
ENV = {
    "time": {"unit": "second"},
    "devices": [{"id": "rack", "spots": ["in", "out"]}, {"id": "reader", "spots": ["slot"]},
                {"id": "sink", "spots": ["basin"]}, {"id": "bench", "spots": []}],
    "transporters": [{"id": "arm"}],
    "transports": [{"transporter": "arm", "from": a, "to": b, "duration": 1}
                   for a in _SPOTS for b in _SPOTS if a != b],
    "processes": {
        "inspect": {"modes": [{"devices": ["reader"], "duration": 3,
                               "input_spots": {"plate": "reader.slot"},
                               "output_spots": {"plate": "reader.slot"}}]},
        "prep": {"modes": [{"devices": ["bench"], "duration": 2}]},
        "fix": {"modes": [{"devices": ["bench"], "duration": 4}]},
        "wash": {"modes": [{"devices": ["sink"], "duration": 5,
                            "input_spots": {"plate": "sink.basin"},
                            "output_spots": {"plate": "sink.basin"}}]},
        "store": {"modes": [{"devices": ["rack"], "duration": 1,
                             "input_spots": {"plate": "rack.out"},
                             "output_spots": {"plate": "rack.out"}}]},
    },
}
PLATE = {"boundary": {"inputs": {"plate": {"spot": "rack.in"}},
                      "outputs": {"plate": {"spot": "rack.out"}}}}


def _redo(n: int):
    """A device model whose inspection finds `n` wells to redo."""

    def model(process, mode, inputs, output_schema, definition):
        outputs = default_device_model(process, mode, inputs, output_schema, definition)
        if process == "inspect":
            outputs["redo"] = list(range(n))
        return outputs

    return model


def _run(workflow, n: int):
    runner = RollingRunner(copy.deepcopy(workflow), copy.deepcopy(ENV), copy.deepcopy(PLATE),
                           poll_interval=None, random_seed=0, device_model=_redo(n))
    return runner, runner.run()


def _done(status, process):
    return [a for a in status["activities"]
            if a["kind"] == "processing" and a["process"] == process
            and a.get("status") == "completed"]


@pytest.mark.parametrize("n", [0, 1, 3])
def test_a_map_runs_as_many_invocations_as_the_inspection_lists(n):
    runner, status = _run(REDO, n)
    assert not runner.failed, runner.failure
    assert status["expansion"]["lengths"] == [{"node": ["I"], "port": "redo", "length": n}]
    fixes, preps = _done(status, "fix"), _done(status, "prep")
    assert len(fixes) == len(preps) == n
    (inspect,) = _done(status, "inspect")
    # Prep reads nothing of the list, and still did not start before it was counted.
    assert all(p["start"] >= inspect["end"] for p in preps)
    assert len(runner.outputs["oks"]) == n


@pytest.mark.parametrize("n", [0, 2])
def test_a_fold_carries_the_plate_through_as_many_washes(n):
    runner, status = _run(WASH, n)
    assert not runner.failed, runner.failure
    assert len(_done(status, "wash")) == n
    (store,) = _done(status, "store")
    assert store["input_spots"]["plate"] == "rack.out"


def test_two_jobs_count_their_own_lists():
    # One plate has two wells to redo, the other none: each job's plan is remade for its
    # own count, and its promise and fingerprint with it.
    env = copy.deepcopy(ENV)
    env["devices"][0]["spots"] = ["in", "out", "in2", "out2"]
    spots = [*_SPOTS, "rack.in2", "rack.out2"]
    env["transports"] = [{"transporter": "arm", "from": a, "to": b, "duration": 1}
                         for a in spots for b in spots if a != b]
    env["processes"]["store"]["modes"].append(
        {"devices": ["rack"], "duration": 1, "input_spots": {"plate": "rack.out2"},
         "output_spots": {"plate": "rack.out2"}})
    counts = iter([2, 0])

    def model(process, mode, inputs, output_schema, definition):
        outputs = default_device_model(process, mode, inputs, output_schema, definition)
        if process == "inspect":
            outputs["redo"] = list(range(next(counts)))
        return outputs

    second = {"boundary": {"inputs": {"plate": {"spot": "rack.in2"}},
                           "outputs": {"plate": {"spot": "rack.out2"}}}}
    runner = RollingRunner(
        [JobRequest("a", copy.deepcopy(REDO), copy.deepcopy(PLATE)),
         JobRequest("b", copy.deepcopy(REDO), second)],
        env, poll_interval=None, random_seed=0, device_model=model,
    )
    status = runner.run()
    assert not runner.failed, runner.failure
    stated = sorted(e["expansion"]["lengths"][0]["length"] for e in status["jobs"])
    assert stated == [0, 2]
    assert len(_done(status, "fix")) == 2
