"""Running a `branch` whose condition is produced during the run (schedule D64).

The scheduler plans such a branch on an assumed arm -- `then` -- and holds all of it
back until the condition exists. The runner decides it as soon as the value is
recorded: it states the arm in the document's `expansion.arms` (dropping the job's
promise and fingerprint in a joint plan, both made for the arm assumed), reads its
own copy of the workflow again with it, and replans. Checks that read what the branch
has not settled wait until then.

The simulator's default device model makes every Bool `false`, so a dry run always
takes `else`: the arm planned is never the arm run, which is the case worth testing.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

import copy

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import (  # noqa: E402
    JobRequest,
    RollingRunner,
    RunnerError,
    rolling,  # noqa: E402
)
from ofplang.run.runner.echo import Echo  # noqa: E402
from ofplang.run.runner.job import build_job, settle_arms  # noqa: E402
from ofplang.run.simulator import default_device_model  # noqa: E402

PROCESSES = """\
  inspect:
    kind: atomic
    inputs: {cup: {type: Cup, phase: data}}
    outputs: {cup: {type: Cup, phase: data}, dirty: {type: Bool, phase: data}}
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
  soak:
    kind: atomic
    inputs: {cup: {type: Cup, phase: data}}
    outputs: {cup: {type: Cup, phase: data}}
    objects: {map: {outputs.cup: inputs.cup}}
"""

# A cup is inspected, then washed if dirty and polished if not.
MEASURED = """\
spec_version: "0.5"
types:
  Cup: {domain: object}
processes:
""" + PROCESSES + """\
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

# As above, but a clean cup is finished, and soaked first if the boundary says it is
# stained: a branch on an entry input that exists only on the `else` arm.
NESTED = """\
spec_version: "0.5"
types:
  Cup: {domain: object}
processes:
""" + PROCESSES + """\
  just_wash:
    kind: composite
    inputs: {cup: {type: Cup, phase: data}, stained: {type: Bool, phase: run}}
    outputs: {cup: {type: Cup, phase: data}}
    body:
      nodes:
        - {id: W, process: wash, state: {cup: {from: inputs.cup}}}
      returns: {cup: {from: W.cup}}
  finish:
    kind: composite
    inputs: {cup: {type: Cup, phase: data}, stained: {type: Bool, phase: run}}
    outputs: {cup: {type: Cup, phase: data}}
    body:
      nodes:
        - id: S
          kind: branch
          condition: {from: inputs.stained}
          args: {cup: {from: inputs.cup}}
          then: {process: soak}
        - {id: P, process: polish, state: {cup: {from: S.cup}}}
      returns: {cup: {from: P.cup}}
  main:
    kind: composite
    inputs: {cup: {type: Cup, phase: data}, stained: {type: Bool, phase: run}}
    outputs: {cup: {type: Cup, phase: data}}
    body:
      nodes:
        - {id: I, process: inspect, state: {cup: {from: inputs.cup}}}
        - id: H
          kind: branch
          condition: {from: I.dirty}
          args: {cup: {from: I.cup}, stained: {from: inputs.stained}}
          then: {process: just_wash}
          else: {process: finish}
      returns: {cup: {from: H.cup}}
entry: main
"""

_PLACES = ["tray.a", "tray.b", "scope.stage", "sink.basin", "buffer.pad", "tub.bath",
           "rack.a", "rack.b"]
ENV = {
    "time": {"unit": "second"},
    "devices": [
        {"id": "tray", "spots": ["a", "b"]}, {"id": "scope", "spots": ["stage"]},
        {"id": "sink", "spots": ["basin"]}, {"id": "buffer", "spots": ["pad"]},
        {"id": "tub", "spots": ["bath"]}, {"id": "rack", "spots": ["a", "b"]},
    ],
    "transporters": [{"id": "arm"}],
    "transports": [
        {"transporter": "arm", "from": a, "to": b, "duration": 1}
        for a in _PLACES for b in _PLACES if a != b
    ],
    "processes": {
        name: {"modes": [{
            "devices": [device], "duration": duration,
            "input_spots": {"cup": f"{device}.{spot}"},
            "output_spots": {"cup": f"{device}.{spot}"},
        }]}
        for name, (device, spot, duration) in {
            "inspect": ("scope", "stage", 2), "wash": ("sink", "basin", 5),
            "polish": ("buffer", "pad", 3), "soak": ("tub", "bath", 4),
        }.items()
    },
}


def one_cup(**flags) -> dict:
    inputs: dict = {"cup": {"spot": "tray.a"}}
    inputs.update({name: {"view": value} for name, value in flags.items()})
    return {"boundary": {"inputs": inputs, "outputs": {"cup": {"spot": "rack.a"}}}}


def found_dirty(process, mode, inputs, output_schema, definition):
    """The simulator's device model, but every inspection finds the cup dirty."""
    outputs = default_device_model(process, mode, inputs, output_schema, definition)
    if process == "inspect":
        outputs["dirty"] = True
    return outputs


def _run(text, boundary, env=ENV, **kwargs):
    runner = RollingRunner(yaml.safe_load(text), copy.deepcopy(env), boundary,
                           poll_interval=None, random_seed=0, **kwargs)
    return runner, runner.run()


def _done(status, job=None):
    return sorted(
        (a["node"], a["process"]) for a in status["activities"]
        if a["kind"] == "processing" and a.get("status") == "completed"
        and (job is None or a.get("job") == job)
    )


def _processing(status, node):
    return next(a for a in status["activities"]
                if a["kind"] == "processing" and a["node"] == node)


# --- running one -----------------------------------------------------------------------


def test_the_arm_run_is_the_one_the_value_says_not_the_one_assumed():
    runner, status = _run(MEASURED, one_cup())  # the dry run finds every cup clean
    assert not runner.failed, runner.failure
    assert _done(status) == [(["H"], "polish"), (["I"], "inspect")]
    assert status["expansion"] == {"arms": [{"node": ["H"], "arm": "else"}]}
    assert _processing(status, ["H"])["start"] >= _processing(status, ["I"])["end"]
    # Nothing is waiting any more, so the final document marks no decision.
    assert not [a for a in status["activities"] if a["kind"] == "decision"]


def test_a_value_that_agrees_with_the_assumption_runs_that_arm():
    runner, status = _run(MEASURED, one_cup(), device_model=found_dirty)
    assert not runner.failed, runner.failure
    assert _done(status) == [(["H"], "wash"), (["I"], "inspect")]
    assert status["expansion"] == {"arms": [{"node": ["H"], "arm": "then"}]}


def test_a_branch_the_arm_taken_reveals_is_decided_from_the_boundary():
    runner, status = _run(NESTED, one_cup(stained=True))
    assert not runner.failed, runner.failure
    assert _done(status) == [(["H", "P"], "polish"), (["H", "S"], "soak"), (["I"], "inspect")]
    assert status["expansion"]["arms"] == [
        {"node": ["H"], "arm": "else"}, {"node": ["H", "S"], "arm": "then"},
    ]


def test_each_job_is_decided_and_promised_again():
    workflow = yaml.safe_load(MEASURED)
    jobs = [
        JobRequest("a", copy.deepcopy(workflow), {"boundary": {
            "inputs": {"cup": {"spot": "tray.a"}}, "outputs": {"cup": {"spot": "rack.a"}}}}),
        JobRequest("b", copy.deepcopy(workflow), {"boundary": {
            "inputs": {"cup": {"spot": "tray.b"}}, "outputs": {"cup": {"spot": "rack.b"}}}}),
    ]
    runner = RollingRunner(jobs, copy.deepcopy(ENV), poll_interval=None, random_seed=0)
    status = runner.run()
    assert not runner.failed, runner.failure
    assert _done(status, "a") == _done(status, "b") == [(["H"], "polish"), (["I"], "inspect")]
    for entry in status["jobs"]:
        assert entry["expansion"]["arms"] == [{"node": ["H"], "arm": "else"}]
        # Dropped when the arm was stated, and written again by the next plan.
        assert isinstance(entry["bound"], int) and isinstance(entry["fingerprint"], str)


def test_an_arm_that_could_not_be_planned_stops_the_run_before_it_starts():
    blocked = copy.deepcopy(ENV)
    blocked["transports"] = [t for t in blocked["transports"] if t["to"] != "buffer.pad"]
    with pytest.raises(RunnerError, match="arm_unplannable"):
        _run(MEASURED, one_cup(), env=blocked)


def test_the_arms_are_checked_only_when_the_answer_can_have_changed(monkeypatch):
    asked = []
    original = rolling.replan

    def counting(*args, **kwargs):
        asked.append(kwargs["check_arms"])
        return original(*args, **kwargs)

    monkeypatch.setattr(rolling, "replan", counting)
    runner, _status = _run(MEASURED, one_cup())
    assert not runner.failed, runner.failure
    assert asked[0] is True and not any(asked[1:]) and len(asked) > 2


# --- what waits ------------------------------------------------------------------------


def test_a_check_inside_the_branch_waits_until_it_is_decided():
    job = build_job(yaml.safe_load(MEASURED), one_cup())
    assert job.waits(("H",)) and not job.waits(("I",))
    assert job.waits(("X",), frozenset({("I",)}))  # waits for I, which has not produced
    assert settle_arms(job) == []
    job.values.put(("I",), "cup", {})
    job.values.put(("I",), "dirty", False)
    assert not job.waits(("X",), frozenset({("I",)}))
    assert settle_arms(job) == [{"node": ["H"], "arm": "else"}]


def test_a_condition_that_is_not_a_boolean_decides_nothing():
    job = build_job(yaml.safe_load(MEASURED), one_cup())
    job.values.put(("I",), "dirty", "yes")
    with pytest.raises(RunnerError, match="not a Boolean"):
        settle_arms(job)


# --- the document ----------------------------------------------------------------------


def test_stating_an_arm_writes_it_where_the_scheduler_reads_it():
    single = Echo(expansion={"lengths": []})
    assert single.state_arm("", ("H",), "else") == {
        "lengths": [], "arms": [{"node": ["H"], "arm": "else"}]
    }
    joint = Echo(jobs=[
        {"id": "a", "release": 0, "bound": 9, "fingerprint": "f"},
        {"id": "b", "release": 0, "bound": 12, "fingerprint": "g"},
    ])
    joint.state_arm("a", ("H",), "then")
    a, b = joint.roster
    assert a == {"id": "a", "release": 0, "expansion": {"arms": [{"node": ["H"], "arm": "then"}]}}
    assert b["bound"] == 12 and b["fingerprint"] == "g"
    with pytest.raises(KeyError):
        joint.state_arm("c", ("H",), "then")


def test_a_decision_is_neither_dispatched_nor_stamped():
    echo = Echo()
    echo.adopt({"activities": [
        {"kind": "decision", "start": 2, "end": 2, "node": ["H"], "arm": "then",
         "assumed": True, "condition": {"node": ["I"], "port": "dirty"}},
        {"kind": "processing", "start": 0, "end": 2, "node": ["I"], "process": "inspect",
         "mode": "m"},
    ]})
    assert [a["kind"] for a in echo.pending()] == ["processing"]
    echo.stamp([], 1, {""})  # the only job has stopped: its pending work is cancelled
    decision, inspect = echo.document["activities"]
    assert "status" not in decision and inspect["status"] == "cancelled"


# --- replay ----------------------------------------------------------------------------


def _plan(expansion=None):
    from ofplang.schedule import schedule

    document = {"interface": {"inputs": {"cup": "tray.a"}, "outputs": {"cup": "rack.a"}},
                "activities": []}
    if expansion:
        document["expansion"] = expansion
    report = schedule(yaml.safe_load(MEASURED), copy.deepcopy(ENV), document_path=document)
    assert report.ok, report.diagnostics
    return report.plan


def test_replay_refuses_a_plan_made_on_an_assumed_arm():
    from ofplang.run.runner.runner import Runner

    with pytest.raises(RunnerError, match="assumes the arm of branch"):
        Runner(_plan(), copy.deepcopy(ENV)).run()


def test_replay_runs_a_plan_whose_arm_was_stated():
    from ofplang.run.runner.runner import Runner

    plan = _plan({"arms": [{"node": ["H"], "arm": "else"}]})
    assert [a["assumed"] for a in plan["activities"] if a["kind"] == "decision"] == [False]
    status = Runner(plan, copy.deepcopy(ENV)).run()
    assert _done(status) == [(["H"], "polish"), (["I"], "inspect")]
    assert not [a for a in status["activities"] if a["kind"] == "decision"]


# --- a check the branch has not settled -------------------------------------------------

# A sample is inspected; the cup is washed if it was dirty -- by a composite whose
# `requires` says the cup must be stained -- and polished if not. Every input of the
# wash is at the boundary, so its `requires` could be evaluated before the sample is.
REQUIRES_ON_THEN = """\
spec_version: "0.5"
types:
  Cup: {domain: object}
processes:
""" + PROCESSES + """\
  stained_wash:
    kind: composite
    inputs: {cup: {type: Cup, phase: data}, stained: {type: Bool, phase: run}}
    outputs: {cup: {type: Cup, phase: data}}
    contracts:
      requires:
        - expr: "inputs.stained.view == true"
    body:
      nodes:
        - {id: W, process: wash, state: {cup: {from: inputs.cup}}}
      returns: {cup: {from: W.cup}}
  plain_polish:
    kind: composite
    inputs: {cup: {type: Cup, phase: data}, stained: {type: Bool, phase: run}}
    outputs: {cup: {type: Cup, phase: data}}
    body:
      nodes:
        - {id: P, process: polish, state: {cup: {from: inputs.cup}}}
      returns: {cup: {from: P.cup}}
  main:
    kind: composite
    inputs:
      cup: {type: Cup, phase: data}
      sample: {type: Cup, phase: data}
      stained: {type: Bool, phase: run}
    outputs: {cup: {type: Cup, phase: data}, sample: {type: Cup, phase: data}}
    body:
      nodes:
        - {id: I, process: inspect, state: {cup: {from: inputs.sample}}}
        - id: H
          kind: branch
          condition: {from: I.dirty}
          args: {cup: {from: inputs.cup}, stained: {from: inputs.stained}}
          then: {process: stained_wash}
          else: {process: plain_polish}
      returns: {cup: {from: H.cup}, sample: {from: I.cup}}
entry: main
"""


def _cup_and_sample(stained):
    return {"boundary": {
        "inputs": {"cup": {"spot": "tray.a"}, "sample": {"spot": "tray.b"},
                   "stained": {"view": stained}},
        "outputs": {"cup": {"spot": "rack.a"}, "sample": {"spot": "rack.b"}},
    }}


def test_the_contract_of_an_arm_assumed_is_not_held_against_the_arm_taken():
    # Clean sample, so `else`: the wash's `requires` -- false here -- is never the
    # run's to check. Checked on the arm assumed, it would have stopped the job.
    runner, status = _run(REQUIRES_ON_THEN, _cup_and_sample(False))
    assert not runner.failed, runner.failure
    assert _done(status) == [(["H", "P"], "polish"), (["I"], "inspect")]


def test_the_contract_of_the_arm_taken_is_checked_once_it_is_taken():
    runner, _status = _run(REQUIRES_ON_THEN, _cup_and_sample(False), device_model=found_dirty)
    assert runner.failed and runner.failure.kind == "contract_requires"
