"""Running a `do_while` (schedule design.md D65).

The scheduler plans a loop whose condition is read during the run to end after the
invocation it is in, and holds whatever follows back until that reading. The runner
decides as each reading is recorded: false ends the loop, true adds one invocation --
stated ended where that one is the last `max_iterations` allows -- and replans. An
ended loop's `exhausted` is recorded at the loop's own path once its last condition is
settled, and a loop the limit stopped is reported (`loop_exhausted`).

The simulator's default device model makes every Bool `false`, so a dry run ends a
loop after one invocation; `reading` below makes a reader find a plate dirty a number
of times first.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

import copy

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.validate import validate  # noqa: E402

from ofplang.run.runner import JobRequest, RollingRunner, RunnerError  # noqa: E402
from ofplang.run.runner.job import build_job  # noqa: E402
from ofplang.run.simulator import default_device_model  # noqa: E402

PROCESSES = """\
  rinse:
    kind: atomic
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}}
    objects: {map: {outputs.plate: inputs.plate}}
  read:
    kind: atomic
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}, dirty: {type: Bool, phase: data}}
    objects: {map: {outputs.plate: inputs.plate}}
  dry:
    kind: atomic
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}}
    objects: {map: {outputs.plate: inputs.plate}}
  rinse_and_read:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}, dirty: {type: Bool, phase: data}}
    body:
      nodes:
        - {id: Rinse, process: rinse, state: {plate: {from: inputs.plate}}}
        - {id: Read, process: read, state: {plate: {from: Rinse.plate}}}
      returns: {plate: {from: Read.plate}, dirty: {from: Read.dirty}}
"""

# A plate rinsed and read until the reading says it is clean, at most `LIMIT` times,
# then dried; `gave_up` is the loop's `exhausted`.
RINSE = """\
spec_version: "0.5"
types:
  Plate: {domain: object}
processes:
""" + PROCESSES + """\
  main:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}, gave_up: {type: Bool, phase: data}}
    body:
      nodes:
        - id: Clean
          kind: do_while
          process: rinse_and_read
          carry: {plate: {from: inputs.plate}}
          condition: {output: dirty}
          max_iterations: LIMIT
        - {id: Dry, process: dry, state: {plate: {from: Clean.plate}}}
      returns: {plate: {from: Dry.plate}, gave_up: {from: Clean.exhausted}}
entry: main
"""


def rinse(limit="{value: 5}") -> dict:
    return yaml.safe_load(RINSE.replace("LIMIT", limit))


# The same, but each invocation hands back the flag it was given -- from the boundary
# (`soiled`) or a literal: the loop is settled before anything runs.
PASSED_ON = """\
spec_version: "0.5"
types:
  Plate: {domain: object}
processes:
""" + PROCESSES + """\
  rinse_only:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}, dirty: {type: Bool, phase: run}}
    outputs: {plate: {type: Plate, phase: data}, dirty: {type: Bool, phase: run}}
    body:
      nodes:
        - {id: Rinse, process: rinse, state: {plate: {from: inputs.plate}}}
      returns: {plate: {from: Rinse.plate}, dirty: {from: inputs.dirty}}
  main:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}, soiled: {type: Bool, phase: run}}
    outputs: {plate: {type: Plate, phase: data}, gave_up: {type: Bool, phase: data}}
    body:
      nodes:
        - id: Clean
          kind: do_while
          process: rinse_only
          carry: {plate: {from: inputs.plate}, dirty: FLAG}
          condition: {output: dirty}
          max_iterations: {value: 3}
        - {id: Dry, process: dry, state: {plate: {from: Clean.plate}}}
      returns: {plate: {from: Dry.plate}, gave_up: {from: Clean.exhausted}}
entry: main
"""

_SPOTS = ["rack.in", "rack.out", "rack.in2", "rack.out2", "sink.basin", "reader.slot",
          "oven.shelf"]
ENV = {
    "time": {"unit": "second"},
    "devices": [{"id": "rack", "spots": ["in", "out", "in2", "out2"]},
                {"id": "sink", "spots": ["basin"]},
                {"id": "reader", "spots": ["slot"]}, {"id": "oven", "spots": ["shelf"]}],
    "transporters": [{"id": "arm"}],
    "transports": [{"transporter": "arm", "from": a, "to": b, "duration": 1}
                   for a in _SPOTS for b in _SPOTS if a != b],
    "processes": {
        name: {"modes": [{"devices": [device], "duration": duration,
                          "input_spots": {"plate": f"{device}.{spot}"},
                          "output_spots": {"plate": f"{device}.{spot}"}}]}
        for name, (device, spot, duration) in {
            "rinse": ("sink", "basin", 3), "read": ("reader", "slot", 1),
            "dry": ("oven", "shelf", 4),
        }.items()
    },
}

PLATE = {"boundary": {"inputs": {"plate": {"spot": "rack.in"}},
                      "outputs": {"plate": {"spot": "rack.out"}}}}


def reading(dirty_times: int):
    """A device model whose reader finds the plate dirty the first `dirty_times`
    times, and clean after."""
    seen: list = []

    def model(process, mode, inputs, output_schema, definition):
        outputs = default_device_model(process, mode, inputs, output_schema, definition)
        if process == "read":
            seen.append(1)
            outputs["dirty"] = len(seen) <= dirty_times
        return outputs

    return model


def _run(workflow, boundary=PLATE, **kwargs):
    runner = RollingRunner(copy.deepcopy(workflow), copy.deepcopy(ENV), copy.deepcopy(boundary),
                           poll_interval=None, random_seed=0, **kwargs)
    return runner, runner.run()


def _done(status, process=None):
    return [a for a in status["activities"]
            if a["kind"] == "processing" and a.get("status") == "completed"
            and (process is None or a["process"] == process)]


def _codes(runner):
    return [w.code for w in runner.warnings if w.code != "entry_input_defaulted"]


def test_the_fixtures_are_valid_v0():
    for workflow in (rinse(), yaml.safe_load(PASSED_ON.replace("FLAG", "{from: inputs.soiled}"))):
        result = validate(workflow)
        assert result.ok, [(d.code, d.message) for d in result.diagnostics]


# --- running one --------------------------------------------------------------------------


def test_a_clean_first_reading_ends_the_loop_at_once():
    runner, status = _run(rinse())  # the dry run reads every plate clean
    assert not runner.failed, runner.failure
    assert len(_done(status, "rinse")) == 1 and len(_done(status, "dry")) == 1
    assert status["expansion"] == {"loops": [{"node": ["Clean"], "count": 1, "ended": True}]}
    assert runner.outputs["gave_up"] is False and _codes(runner) == []
    assert not [a for a in status["activities"] if a["kind"] == "decision"]


def test_the_loop_goes_on_as_long_as_the_reading_says_and_nothing_runs_early():
    runner, status = _run(rinse(), device_model=reading(2))
    assert not runner.failed, runner.failure
    assert status["expansion"] == {"loops": [{"node": ["Clean"], "count": 3, "ended": True}]}
    reads = {a["node"][1]: a for a in _done(status, "read")}
    rinses = {a["node"][1]: a for a in _done(status, "rinse")}
    assert sorted(reads) == sorted(rinses) == [0, 1, 2]
    for i in (1, 2):
        assert rinses[i]["start"] >= reads[i - 1]["end"]
    (dry,) = _done(status, "dry")
    assert dry["start"] >= reads[2]["end"]
    assert runner.outputs["gave_up"] is False


def test_at_the_limit_it_stops_and_says_so():
    runner, status = _run(rinse("{value: 3}"), device_model=reading(99))
    assert not runner.failed, runner.failure
    assert status["expansion"] == {"loops": [{"node": ["Clean"], "count": 3, "ended": True}]}
    assert runner.outputs["gave_up"] is True
    assert _codes(runner) == ["loop_exhausted"]


def test_a_last_clean_reading_at_the_limit_is_not_giving_up():
    runner, _status = _run(rinse("{value: 3}"), device_model=reading(2))
    assert runner.outputs["gave_up"] is False and _codes(runner) == []


def test_a_limit_from_the_boundary():
    workflow = rinse("{from: inputs.rounds}")
    workflow["processes"]["main"]["inputs"]["rounds"] = {"type": "Int", "phase": "run"}
    boundary = copy.deepcopy(PLATE)
    boundary["boundary"]["inputs"]["rounds"] = {"view": 2}
    runner, status = _run(workflow, boundary, device_model=reading(99))
    assert not runner.failed, runner.failure
    assert status["expansion"]["loops"] == [{"node": ["Clean"], "count": 2, "ended": True}]
    assert runner.outputs["gave_up"] is True
    boundary["boundary"]["inputs"]["rounds"] = {"view": 0}
    with pytest.raises(RunnerError, match="max_iterations"):
        _run(workflow, boundary, device_model=reading(99))


# --- settled before the run ---------------------------------------------------------------


@pytest.mark.parametrize("soiled, count, gave_up", [(True, 3, True), (False, 1, False)])
def test_a_condition_from_the_boundary_is_decided_before_the_run(soiled, count, gave_up):
    workflow = yaml.safe_load(PASSED_ON.replace("FLAG", "{from: inputs.soiled}"))
    boundary = copy.deepcopy(PLATE)
    boundary["boundary"]["inputs"]["soiled"] = {"view": soiled}
    job = build_job(copy.deepcopy(workflow), copy.deepcopy(boundary))
    assert job.expansion == {"loops": [{"node": ["Clean"], "count": count, "ended": True}]}
    runner, status = _run(workflow, boundary)
    assert not runner.failed, runner.failure
    assert len(_done(status, "rinse")) == count
    assert runner.outputs["gave_up"] is gave_up
    assert _codes(runner) == (["loop_exhausted"] if gave_up else [])


def test_a_literal_loop_records_its_exhausted_too():
    workflow = yaml.safe_load(PASSED_ON.replace("FLAG", "{value: true}"))
    runner, status = _run(workflow)
    assert not runner.failed, runner.failure
    assert len(_done(status, "rinse")) == 3
    assert runner.outputs["gave_up"] is True and _codes(runner) == ["loop_exhausted"]
    assert "expansion" not in status  # the scheduler decided it; nothing to state


# --- shapes ------------------------------------------------------------------------------


def _two_of_each() -> dict:
    """ENV with a second sink and reader, either of which may rinse or read.

    With one of each and nowhere to wait, two jobs can lock each other: planned on the
    first plate's loop ending, the second plate is moved into the sink, and when the
    first one goes on instead it needs the sink back while the second needs the reader
    the first is waiting on. That is what planning on an assumption risks (design.md
    D65): the replan fails, as a device going down can make it fail."""
    env = copy.deepcopy(ENV)
    env["devices"] += [{"id": "sink2", "spots": ["basin"]}, {"id": "reader2", "spots": ["slot"]}]
    spots = _SPOTS + ["sink2.basin", "reader2.slot"]
    env["transports"] = [{"transporter": "arm", "from": a, "to": b, "duration": 1}
                         for a in spots for b in spots if a != b]
    for name, device in (("rinse", "sink2"), ("read", "reader2")):
        mode = copy.deepcopy(env["processes"][name]["modes"][0])
        spot = next(iter(mode["input_spots"].values())).split(".")[1]
        mode.update({"devices": [device], "input_spots": {"plate": f"{device}.{spot}"},
                     "output_spots": {"plate": f"{device}.{spot}"}})
        env["processes"][name]["modes"].append(mode)
    return env


def test_each_job_is_decided_and_promised_again():
    workflow = rinse()
    jobs = [JobRequest("a", copy.deepcopy(workflow), copy.deepcopy(PLATE)),
            JobRequest("b", copy.deepcopy(workflow), {"boundary": {
                "inputs": {"plate": {"spot": "rack.in2"}},
                "outputs": {"plate": {"spot": "rack.out2"}}}})]
    runner = RollingRunner(jobs, _two_of_each(), poll_interval=None, random_seed=0,
                           device_model=reading(1))
    status = runner.run()
    assert not runner.failed, runner.failure
    counts = sorted(e["expansion"]["loops"][0]["count"] for e in status["jobs"])
    assert counts == [1, 2]  # the first reading of the run is dirty, the rest clean
    for entry in status["jobs"]:
        assert isinstance(entry["bound"], int) and isinstance(entry["fingerprint"], str)


# A branch on a flag the boundary gives, handed on by a branch the sample decides --
# kept on `then`, read again on `else`. The second branch is not decided before the first.
FLAG_THROUGH = """\
spec_version: "0.5"
types:
  Plate: {domain: object}
processes:
""" + PROCESSES + """\
  keep_flag:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}, flag: {type: Bool, phase: run}}
    outputs: {plate: {type: Plate, phase: data}, flag: {type: Bool, phase: run}}
    body:
      nodes:
        - {id: W, process: rinse, state: {plate: {from: inputs.plate}}}
      returns: {plate: {from: W.plate}, flag: {from: inputs.flag}}
  new_flag:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}, flag: {type: Bool, phase: run}}
    outputs: {plate: {type: Plate, phase: data}, flag: {type: Bool, phase: run}}
    body:
      nodes:
        - {id: L, process: read, state: {plate: {from: inputs.plate}}}
      returns: {plate: {from: L.plate}, flag: {from: L.dirty}}
  main:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}, flag: {type: Bool, phase: run}}
    outputs: {plate: {type: Plate, phase: data}}
    body:
      nodes:
        - {id: I, process: read, state: {plate: {from: inputs.plate}}}
        - id: H
          kind: branch
          condition: {from: I.dirty}
          args: {plate: {from: I.plate}, flag: {from: inputs.flag}}
          then: {process: keep_flag}
          else: {process: new_flag}
          outputs: {plate: {mode: common}, flag: {mode: common}}
        - id: K
          kind: branch
          condition: {from: H.flag}
          args: {plate: {from: H.plate}}
          then: {process: dry}
      returns: {plate: {from: K.plate}}
entry: main
"""


def test_a_flag_a_pending_branch_hands_on_is_not_decided_early():
    # The boundary's flag is true, but the first reading is clean: H takes `else` and
    # reads the flag again -- false. Deciding K from the boundary before H, as 0.16 did,
    # would have dried the plate.
    workflow = yaml.safe_load(FLAG_THROUGH)
    assert validate(copy.deepcopy(workflow)).ok
    boundary = copy.deepcopy(PLATE)
    boundary["boundary"]["inputs"]["flag"] = {"view": True}
    assert build_job(copy.deepcopy(workflow), copy.deepcopy(boundary)).expansion is None
    runner, status = _run(workflow, boundary)
    assert not runner.failed, runner.failure
    assert status["expansion"]["arms"] == [{"node": ["H"], "arm": "else"},
                                           {"node": ["K"], "arm": "else"}]
    assert _done(status, "dry") == []


# --- replay ------------------------------------------------------------------------------


def test_replay_refuses_a_plan_that_assumes_a_loop_ends():
    from ofplang.schedule import schedule

    from ofplang.run.runner.runner import Runner

    plan = schedule(rinse(), copy.deepcopy(ENV),
                    document_path={"interface": {"inputs": {"plate": "rack.in"},
                                                 "outputs": {"plate": "rack.out"}},
                                   "activities": []}).plan
    with pytest.raises(RunnerError, match="assumes the arm"):
        Runner(plan, copy.deepcopy(ENV)).run()


def test_planning_on_an_assumption_can_lock_two_jobs():
    # The case `_two_of_each` avoids, pinned as the behaviour it is: with one sink and
    # one reader and nowhere to wait, the second plate is planned into the sink while
    # the first waits on the reader for its reading; the reading says go on, and
    # neither can move. The replan has no answer, and every job stops (as for a
    # device going down, design.md D64).
    workflow = rinse()
    jobs = [JobRequest("a", copy.deepcopy(workflow), copy.deepcopy(PLATE)),
            JobRequest("b", copy.deepcopy(workflow), {"boundary": {
                "inputs": {"plate": {"spot": "rack.in2"}},
                "outputs": {"plate": {"spot": "rack.out2"}}}})]
    runner = RollingRunner(jobs, copy.deepcopy(ENV), poll_interval=None, random_seed=0,
                           device_model=reading(1))
    runner.run()
    assert runner.failed and runner.failure.kind == "replan_infeasible"


# Each invocation soaks the plate when its reading says dirty: a branch per invocation,
# each on its own reading.
SOAKED = """\
spec_version: "0.5"
types:
  Plate: {domain: object}
processes:
""" + PROCESSES + """\
  soak:
    kind: atomic
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}}
    objects: {map: {outputs.plate: inputs.plate}}
  rinse_read_soak:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}, dirty: {type: Bool, phase: data}}
    body:
      nodes:
        - {id: Rinse, process: rinse, state: {plate: {from: inputs.plate}}}
        - {id: Read, process: read, state: {plate: {from: Rinse.plate}}}
        - id: Soak
          kind: branch
          condition: {from: Read.dirty}
          args: {plate: {from: Read.plate}}
          then: {process: soak}
      returns: {plate: {from: Soak.plate}, dirty: {from: Read.dirty}}
  main:
    kind: composite
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}, readings: {type: "Array<Bool>", phase: data}}
    body:
      nodes:
        - id: Clean
          kind: do_while
          process: rinse_read_soak
          carry: {plate: {from: inputs.plate}}
          condition: {output: dirty}
          max_iterations: {value: 4}
          outputs: {plate: {mode: carry}, dirty: {mode: collect}}
        - {id: Log, kind: map, process: note, each: {reading: {from: Clean.dirty}}}
        - {id: Dry, process: dry, state: {plate: {from: Clean.plate}}}
      returns: {plate: {from: Dry.plate}, readings: {from: Log.line}}
  note:
    kind: atomic
    inputs: {reading: {type: Bool, phase: data}}
    outputs: {line: {type: Bool, phase: data}}
entry: main
"""


def test_a_branch_per_invocation_and_a_map_over_what_the_loop_collected():
    workflow = yaml.safe_load(SOAKED)
    assert validate(copy.deepcopy(workflow)).ok
    env = copy.deepcopy(ENV)
    env["devices"] += [{"id": "tub", "spots": ["bath"]}, {"id": "logger", "spots": []}]
    spots = _SPOTS + ["tub.bath"]
    env["transports"] = [{"transporter": "arm", "from": a, "to": b, "duration": 1}
                         for a in spots for b in spots if a != b]
    env["processes"]["soak"] = {"modes": [{"devices": ["tub"], "duration": 2,
                                           "input_spots": {"plate": "tub.bath"},
                                           "output_spots": {"plate": "tub.bath"}}]}
    env["processes"]["note"] = {"modes": [{"devices": ["logger"], "duration": 1}]}
    runner = RollingRunner(copy.deepcopy(workflow), env, copy.deepcopy(PLATE),
                           poll_interval=None, random_seed=0, device_model=reading(2))
    status = runner.run()
    assert not runner.failed, runner.failure
    # Dirty, dirty, clean: soaked after the first two readings only.
    assert sorted(a["node"][1] for a in _done(status, "soak")) == [0, 1]
    assert [e["arm"] for e in status["expansion"]["arms"]] == ["then", "then", "else"]
    assert status["expansion"]["loops"] == [{"node": ["Clean"], "count": 3, "ended": True}]
    # The map logs every reading, as many as there were -- after the last one.
    assert runner.outputs["readings"] == [False, False, False]  # the logger's own default
    last_read = max(a["end"] for a in _done(status, "read"))
    assert all(a["start"] >= last_read for a in _done(status, "note"))
    assert len(_done(status, "note")) == 3
