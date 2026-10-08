"""Running a `branch` whose arm the boundary decides (schedule D63, stage 1).

The scheduler expands a branch with one arm, so the arm has to be known before the
plan. Where the condition is an entry input -- a flag, or one flag per element inside
a `map` -- the runner holds the value: it decides the arm, states it in the
document's `expansion.arms`, and flattens its own copy of the workflow with the same
arms, so the activities it runs are the ones the plan names. A branch inside an arm
appears only once that arm is decided, so the runner asks again until nothing is
left. A condition produced during the run is the scheduler's to refuse.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

import copy

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.app import front_door_check  # noqa: E402
from ofplang.run.runner import JobRequest, RollingRunner, RunnerError  # noqa: E402
from ofplang.run.runner.job import build_job  # noqa: E402

CUP_PROCESSES = """\
  soak:
    kind: atomic
    inputs: {cup: {type: Cup, phase: data}}
    outputs: {cup: {type: Cup, phase: data}}
    objects: {map: {outputs.cup: inputs.cup}}
  wash:
    kind: atomic
    inputs: {cup: {type: Cup, phase: data}}
    outputs: {cup: {type: Cup, phase: data}}
    objects: {map: {outputs.cup: inputs.cup}}
  polish:
    kind: atomic
    inputs: {cup: {type: Cup, phase: data}}
    outputs: {cup: {type: Cup, phase: data}}
    objects: {map: {outputs.cup: inputs.cup}}
"""

WASH_OR_POLISH = """\
spec_version: "0.5"
types:
  Cup: {domain: object}
processes:
""" + CUP_PROCESSES + """\
  main:
    kind: composite
    inputs:
      cup: {type: Cup, phase: data}
      dirty: {type: Bool, phase: run}
    outputs: {cup: {type: Cup, phase: data}}
    body:
      nodes:
        - id: H
          kind: branch
          condition: {from: inputs.dirty}
          args: {cup: {from: inputs.cup}}
          then: {process: wash}
          else: {process: polish}
      returns: {cup: {from: H.cup}}
entry: main
"""

PER_CUP = """\
spec_version: "0.5"
types:
  Cup: {domain: object}
processes:
""" + CUP_PROCESSES + """\
  handle:
    kind: composite
    inputs:
      cup: {type: Cup, phase: data}
      dirty: {type: Bool, phase: run}
    outputs: {cup: {type: Cup, phase: data}}
    body:
      nodes:
        - id: H
          kind: branch
          condition: {from: inputs.dirty}
          args: {cup: {from: inputs.cup}}
          then: {process: wash}
          else: {process: polish}
      returns: {cup: {from: H.cup}}
  main:
    kind: composite
    inputs:
      cups: {type: "Array<Cup>", phase: data}
      dirty: {type: "Array<Bool>", phase: run}
    outputs: {cups: {type: "Array<Cup>", phase: data}}
    body:
      nodes:
        - id: M
          kind: map
          process: handle
          each: {cup: {from: inputs.cups}, dirty: {from: inputs.dirty}}
      returns: {cups: {from: M.cup}}
entry: main
"""

NESTED = """\
spec_version: "0.5"
types:
  Cup: {domain: object}
processes:
""" + CUP_PROCESSES + """\
  clean:
    kind: composite
    inputs:
      cup: {type: Cup, phase: data}
      stained: {type: Bool, phase: run}
    outputs: {cup: {type: Cup, phase: data}}
    body:
      nodes:
        - id: S
          kind: branch
          condition: {from: inputs.stained}
          args: {cup: {from: inputs.cup}}
          then: {process: soak}
        - {id: W, process: wash, state: {cup: {from: S.cup}}}
      returns: {cup: {from: W.cup}}
  finish:
    kind: composite
    inputs:
      cup: {type: Cup, phase: data}
      stained: {type: Bool, phase: run}
    outputs: {cup: {type: Cup, phase: data}}
    body:
      nodes:
        - {id: P, process: polish, state: {cup: {from: inputs.cup}}}
      returns: {cup: {from: P.cup}}
  main:
    kind: composite
    inputs:
      cup: {type: Cup, phase: data}
      dirty: {type: Bool, phase: run}
      stained: {type: Bool, phase: run}
    outputs: {cup: {type: Cup, phase: data}}
    body:
      nodes:
        - id: H
          kind: branch
          condition: {from: inputs.dirty}
          args: {cup: {from: inputs.cup}, stained: {from: inputs.stained}}
          then: {process: clean}
          else: {process: finish}
      returns: {cup: {from: H.cup}}
entry: main
"""

MEASURED = """\
spec_version: "0.5"
types:
  Cup: {domain: object}
processes:
""" + CUP_PROCESSES + """\
  inspect:
    kind: atomic
    inputs: {cup: {type: Cup, phase: data}}
    outputs: {cup: {type: Cup, phase: data}, dirty: {type: Bool, phase: data}}
    objects: {map: {outputs.cup: inputs.cup}}
  main:
    kind: composite
    inputs: {cup: {type: Cup, phase: data}}
    outputs: {cup: {type: Cup, phase: data}}
    body:
      nodes:
        - {id: I, process: inspect, state: {cup: {from: inputs.cup}}}
        - id: H
          kind: branch
          condition: {from: I.dirty}
          args: {cup: {from: I.cup}}
          then: {process: wash}
          else: {process: polish}
      returns: {cup: {from: H.cup}}
entry: main
"""

ENV = yaml.safe_load("""
time: {unit: second}
devices:
  - {id: tray, spots: [a, b]}
  - {id: sink, spots: [basin]}
  - {id: buffer, spots: [pad]}
  - {id: tub, spots: [bath]}
  - {id: rack, spots: [a, b]}
transporters: [{id: arm}]
transports:
  - {transporter: arm, from: tray.a, to: sink.basin, duration: 1}
  - {transporter: arm, from: tray.b, to: sink.basin, duration: 1}
  - {transporter: arm, from: tray.a, to: buffer.pad, duration: 1}
  - {transporter: arm, from: tray.b, to: buffer.pad, duration: 1}
  - {transporter: arm, from: tray.a, to: tub.bath, duration: 1}
  - {transporter: arm, from: tub.bath, to: sink.basin, duration: 1}
  - {transporter: arm, from: sink.basin, to: rack.a, duration: 1}
  - {transporter: arm, from: sink.basin, to: rack.b, duration: 1}
  - {transporter: arm, from: buffer.pad, to: rack.a, duration: 1}
  - {transporter: arm, from: buffer.pad, to: rack.b, duration: 1}
processes:
  wash:
    modes:
      - {devices: [sink], duration: 5,
         input_spots: {cup: sink.basin}, output_spots: {cup: sink.basin}}
  polish:
    modes:
      - {devices: [buffer], duration: 3,
         input_spots: {cup: buffer.pad}, output_spots: {cup: buffer.pad}}
  soak:
    modes:
      - {devices: [tub], duration: 8,
         input_spots: {cup: tub.bath}, output_spots: {cup: tub.bath}}
  inspect:
    modes:
      - {devices: [sink], duration: 1,
         input_spots: {cup: sink.basin}, output_spots: {cup: sink.basin}}
""")


def one_cup(**flags) -> dict:
    inputs: dict = {"cup": {"spot": "tray.a"}}
    inputs.update({name: {"view": value} for name, value in flags.items()})
    return {"boundary": {"inputs": inputs, "outputs": {"cup": {"spot": "rack.a"}}}}


def _run(text: str, boundary: dict) -> tuple[RollingRunner, dict]:
    runner = RollingRunner(yaml.safe_load(text), copy.deepcopy(ENV), boundary,
                           poll_interval=None, random_seed=0)
    return runner, runner.run()


def _done(status: dict, job: str | None = None) -> list[tuple[list, str]]:
    return sorted(
        (a["node"], a["process"]) for a in status["activities"]
        if a["kind"] == "processing" and a.get("status") == "completed"
        and (job is None or a.get("job") == job)
    )


def test_the_gate_lets_a_branch_through():
    assert front_door_check(yaml.safe_load(WASH_OR_POLISH), validate=False).ok


@pytest.mark.parametrize("dirty, process, arm", [(True, "wash", "then"), (False, "polish", "else")])
def test_the_flag_decides_the_arm_that_runs(dirty, process, arm):
    runner, status = _run(WASH_OR_POLISH, one_cup(dirty=dirty))
    assert not runner.failed, runner.failure
    assert _done(status) == [(["H"], process)]
    assert status["expansion"] == {"arms": [{"node": ["H"], "arm": arm}]}


def test_a_flag_left_out_is_its_default_and_says_so():
    runner, status = _run(WASH_OR_POLISH, one_cup())
    assert not runner.failed, runner.failure
    assert _done(status) == [(["H"], "polish")]
    # The flag is reported as any entry input left out is (the cup's view is too).
    assert any(w.code == "entry_input_defaulted" and "'dirty'" in w.message
               for w in runner.warnings), runner.warnings


def test_each_cup_takes_the_arm_its_own_flag_says():
    boundary = {"boundary": {
        "inputs": {"cups": {"spot": ["tray.a", "tray.b"]}, "dirty": {"view": [False, True]}},
        "outputs": {"cups": {"spot": ["rack.a", "rack.b"]}},
    }}
    runner, status = _run(PER_CUP, boundary)
    assert not runner.failed, runner.failure
    assert _done(status) == [(["M", 0, "H"], "polish"), (["M", 1, "H"], "wash")]
    assert status["expansion"]["arms"] == [
        {"node": ["M", 0, "H"], "arm": "else"}, {"node": ["M", 1, "H"], "arm": "then"},
    ]


def test_a_branch_inside_an_arm_is_decided_once_the_arm_is():
    runner, status = _run(NESTED, one_cup(dirty=True, stained=True))
    assert not runner.failed, runner.failure
    assert _done(status) == [(["H", "S"], "soak"), (["H", "W"], "wash")]
    assert status["expansion"]["arms"] == [
        {"node": ["H"], "arm": "then"}, {"node": ["H", "S"], "arm": "then"},
    ]
    # Not dirty: the inner branch is in the arm not taken, and nothing states it.
    runner, status = _run(NESTED, one_cup(dirty=False, stained=True))
    assert _done(status) == [(["H", "P"], "polish")]
    assert status["expansion"]["arms"] == [{"node": ["H"], "arm": "else"}]


def test_each_job_takes_its_own_arm():
    workflow = yaml.safe_load(WASH_OR_POLISH)
    jobs = [
        JobRequest("a", copy.deepcopy(workflow), {"boundary": {
            "inputs": {"cup": {"spot": "tray.a"}, "dirty": {"view": True}},
            "outputs": {"cup": {"spot": "rack.a"}}}}),
        JobRequest("b", copy.deepcopy(workflow), {"boundary": {
            "inputs": {"cup": {"spot": "tray.b"}, "dirty": {"view": False}},
            "outputs": {"cup": {"spot": "rack.b"}}}}),
    ]
    runner = RollingRunner(jobs, copy.deepcopy(ENV), poll_interval=None, random_seed=0)
    status = runner.run()
    assert not runner.failed, runner.failure
    assert _done(status, "a") == [(["H"], "wash")] and _done(status, "b") == [(["H"], "polish")]
    assert [e["expansion"]["arms"][0]["arm"] for e in status["jobs"]] == ["then", "else"]


def test_a_condition_produced_during_the_run_is_refused_before_it_starts():
    with pytest.raises(RunnerError, match="branch_arm_unknown"):
        build_job(yaml.safe_load(MEASURED), one_cup())


def test_a_flag_that_is_not_a_boolean_is_refused():
    with pytest.raises(RunnerError, match="does not conform"):
        build_job(yaml.safe_load(WASH_OR_POLISH), one_cup(dirty="yes"))
