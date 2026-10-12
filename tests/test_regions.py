"""Runs whose branches and loops keep their spots while undecided (schedule design.md
D66, stage 5).

A branch or loop whose condition is read during the run is planned on an assumed
outcome, and keeps the spots any outcome may need for as long as it is open. So the
run never finds another job standing where the outcome it reads has to go: it only
waits longer when the outcome is not the one assumed, and replans whenever work takes
longer than planned. These are end-to-end runs of that.
"""

from __future__ import annotations

import copy

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import JobRequest, RollingRunner  # noqa: E402
from tests.test_branch_measured import ENV as CUPS  # noqa: E402
from tests.test_branch_measured import MEASURED, PROCESSES  # noqa: E402
from tests.test_loops import ENV as RINSE_ENV  # noqa: E402
from tests.test_loops import PLATE, reading, rinse  # noqa: E402

# A cup polished first, then inspected: it needs the buffer and then the scope -- the
# reverse of a clean cup of `MEASURED`, which is inspected and then polished.
POLISH_THEN_INSPECT = """\
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
        - {id: P, process: polish, state: {cup: {from: inputs.cup}}}
        - {id: L, process: inspect, state: {cup: {from: P.cup}}}
      returns: {cup: {from: L.cup}}
entry: main
"""


def _cup(job: str, workflow: str, tray: str, rack: str) -> JobRequest:
    return JobRequest(job, yaml.safe_load(workflow), {"boundary": {
        "inputs": {"cup": {"spot": f"tray.{tray}"}},
        "outputs": {"cup": {"spot": f"rack.{rack}"}}}})


def _on(status, job, spot):
    """Every stretch of time `job` has something on `spot`, from the history."""
    spans = []
    for a in status["activities"]:
        if a.get("job") != job or a.get("status") != "completed":
            continue
        if a["kind"] == "processing":
            spots = {*a.get("input_spots", {}).values(), *a.get("output_spots", {}).values()}
            if spot in spots:
                spans.append((a["start"], a["end"]))
        elif a["kind"] == "transport" and a["from_spot"] != a["to_spot"]:
            if spot in (a["from_spot"], a["to_spot"]):
                spans.append((a["start"], a["end"]))
    return spans


def test_a_branch_taking_the_arm_not_assumed_finds_its_place_free():
    # Cup a is planned to be washed (the arm assumed) but turns out clean and is
    # polished; cup b goes through the polishing buffer too, then the scope a stands
    # on while it is decided. Had b been planned onto the buffer around a's decision,
    # each would wait for the other. The buffer is a's from its decision to its exit.
    jobs = [_cup("a", MEASURED, "a", "a"), _cup("b", POLISH_THEN_INSPECT, "b", "b")]
    runner = RollingRunner(jobs, copy.deepcopy(CUPS), poll_interval=None, random_seed=0)
    status = runner.run()
    assert not runner.failed, runner.failure
    arms = next(e for e in status["jobs"] if e["id"] == "a")["expansion"]["arms"]
    assert arms == [{"node": ["H"], "arm": "else"}]
    inspected = next(
        a for a in status["activities"]
        if a.get("job") == "a" and a["kind"] == "processing" and a["node"] == ["I"]
    )
    polished = next(
        a for a in status["activities"]
        if a.get("job") == "a" and a["kind"] == "processing" and a["node"] == ["H"]
    )
    # Nothing of b on the buffer between a's decision and a's polishing being done.
    for start, end in _on(status, "b", "buffer.pad"):
        assert end <= inspected["end"] or start >= polished["end"], (start, end)


def _overrun_first_rinse(by: int):
    """Every first rinse takes `by` longer than planned; everything else as planned."""
    def model(activity, planned):
        node = activity.get("node") or []
        late = activity["kind"] == "processing" and node[-1:] == ["Rinse"] and node[1:2] == [0]
        return planned + by if late else planned

    return model


@pytest.mark.parametrize("by", [1, 4])
def test_two_loops_on_one_sink_still_take_turns_when_work_runs_late(by):
    # The lock the reservations remove, with the work taking longer than planned: the
    # replans only push things later, and the two loops still take turns at the sink.
    workflow = rinse()
    jobs = [JobRequest("a", copy.deepcopy(workflow), copy.deepcopy(PLATE)),
            JobRequest("b", copy.deepcopy(workflow), {"boundary": {
                "inputs": {"plate": {"spot": "rack.in2"}},
                "outputs": {"plate": {"spot": "rack.out2"}}}})]
    runner = RollingRunner(
        jobs, copy.deepcopy(RINSE_ENV), poll_interval=1, running_task_margin=1,
        random_seed=0, device_model=reading(1), duration_model=_overrun_first_rinse(by),
    )
    status = runner.run()
    assert not runner.failed, runner.failure
    counts = sorted(e["expansion"]["loops"][0]["count"] for e in status["jobs"])
    assert counts == [1, 2]
    a_sink, b_sink = _on(status, "a", "sink.basin"), _on(status, "b", "sink.basin")
    a_span = (min(s for s, _e in a_sink), max(e for _s, e in a_sink))
    b_span = (min(s for s, _e in b_sink), max(e for _s, e in b_sink))
    assert a_span[1] <= b_span[0] or b_span[1] <= a_span[0], (a_span, b_span)
