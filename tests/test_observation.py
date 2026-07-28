"""Tests for the observation document (D38): the run-produced record of completed
activities' I/O view values (spec `docs/OBSERVATION.md`).

Two layers are covered: the pure document builders (`observation.py`) and the
runner wiring (`RollingRunner(observe=...)` / `observation_out=...`). The runner
tests drive real v0 workflows through `ofplang.schedule`, so they are skipped when
the scheduler is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ofplang.run.runner import observation
from ofplang.run.runner.provenance import Committed

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
COUNT_WF = str(FIXTURES / "count_chain.workflow.yaml")
COUNT_ENV = str(FIXTURES / "count_chain.env.yaml")
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")
FAIL_WF = str(FIXTURES / "failure.workflow.yaml")
FAIL_ENV = str(FIXTURES / "failure.env.yaml")

COUNT_BOUNDARY = {"boundary": {"inputs": {"start": {"view": {"value": 42}}}}}


def _inc_model(process, mode, inputs, output_schema, definition):
    """`inc`: every output Count is the input Count plus one (mirrors the example)."""
    n = inputs["x"]["value"]
    return {port: {"value": n + 1} for port in output_schema}


def _by_node(entries, node):
    return next(e for e in entries if e.get("node") == list(node))


# -- pure builders --------------------------------------------------------------


def test_entry_doc_echoes_structure_and_wraps_values():
    rec = Committed(
        activity={
            "kind": "processing",
            "process": "inc",
            "mode": "default",
            "node": ["S1"],
            "devices": ["d0"],
        },
        kind="processing",
        status="completed",
        start=0,
        end=5,
    )
    entry = observation.entry_doc(rec, inputs={"x": {"value": 42}}, outputs={"y": {"value": 43}})
    assert entry["kind"] == "processing"
    assert entry["process"] == "inc"
    assert entry["mode"] == "default"
    assert entry["node"] == ["S1"]
    assert entry["devices"] == ["d0"]
    assert entry["start"] == 0 and entry["end"] == 5
    assert entry["inputs"] == {"x": {"view": {"value": 42}}}
    assert entry["outputs"] == {"y": {"view": {"value": 43}}}
    assert "status" not in entry  # every entry is completed by the scope rule


def test_entry_doc_deep_copies_values():
    # A later in-place mutation of a shared value object must not rewrite the record.
    rec = Committed(activity={"kind": "processing", "node": ["S1"]}, kind="processing",
                    status="completed", start=0, end=1)
    shared = {"value": 42}
    entry = observation.entry_doc(rec, inputs={"x": shared}, outputs={})
    shared["value"] = 999
    assert entry["inputs"]["x"]["view"] == {"value": 42}


def test_entry_doc_transport_uses_moved():
    rec = Committed(
        activity={
            "kind": "transport",
            "from_spot": "a.slot",
            "to_spot": "b.slot",
            "transporter": "arm",
            "arc": {"from": {"node": ["S1"], "port": "y"}, "to": {"node": ["S2"], "port": "x"}},
        },
        kind="transport",
        status="completed",
        start=1,
        end=2,
    )
    entry = observation.entry_doc(rec, moved={"barcode": "ABC"})
    assert entry["from_spot"] == "a.slot" and entry["to_spot"] == "b.slot"
    assert entry["arc"]["from"]["node"] == ["S1"]
    assert entry["moved"] == {"view": {"barcode": "ABC"}}
    assert "inputs" not in entry and "outputs" not in entry


def test_write_stream_round_trips(tmp_path):
    entries = [
        observation.entry_doc(
            Committed({"kind": "processing", "node": ["S1"]}, "processing", "completed", 0, 1),
            inputs={"x": {"value": 1}},
            outputs={"y": {"value": 2}},
        )
    ]
    path = tmp_path / "obs.yaml"
    observation.write_stream(
        str(path), entries, time_section={"unit": "second"}, now=1, outcome="completed"
    )
    docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    assert docs[0]["schema"] == observation.SCHEMA
    assert docs[0]["time"] == {"unit": "second"}
    assert docs[1]["node"] == ["S1"]
    assert docs[-1] == {"final": True, "now": 1, "outcome": "completed"}


def test_format_text_renders_value_flow():
    entries = [
        observation.entry_doc(
            Committed({"kind": "processing", "process": "inc", "node": ["S1"]},
                      "processing", "completed", 0, 1),
            inputs={"x": {"value": 42}},
            outputs={"y": {"value": 43}},
        )
    ]
    text = observation.format_text(entries)
    assert "S1 [inc]" in text
    assert "42" in text and "43" in text


# -- runner wiring --------------------------------------------------------------


def test_observation_off_by_default():
    runner = RollingRunner(COUNT_WF, COUNT_ENV, COUNT_BOUNDARY, device_model=_inc_model,
                           random_seed=0, poll_interval=None)
    runner.run()
    assert runner.observations == []


def test_observe_accumulates_value_flow():
    runner = RollingRunner(COUNT_WF, COUNT_ENV, COUNT_BOUNDARY, device_model=_inc_model,
                           random_seed=0, poll_interval=None, observe=True)
    runner.run()
    entries = runner.observations
    procs = [e for e in entries if e["kind"] == "processing"]
    assert len(procs) == 2
    s1 = _by_node(entries, ("S1",))
    s2 = _by_node(entries, ("S2",))
    # inc: 42 -> 43 -> 44, faithful per-activity inputs and outputs.
    assert s1["inputs"]["x"]["view"] == {"value": 42}
    assert s1["outputs"]["y"]["view"] == {"value": 43}
    assert s2["inputs"]["x"]["view"] == {"value": 43}
    assert s2["outputs"]["y"]["view"] == {"value": 44}


def test_observation_stream_file(tmp_path):
    path = tmp_path / "run.observation.yaml"
    runner = RollingRunner(COUNT_WF, COUNT_ENV, COUNT_BOUNDARY, device_model=_inc_model,
                           random_seed=0, poll_interval=None, observation_out=str(path))
    runner.run()
    docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    header, *body, trailer = docs
    assert header["schema"] == observation.SCHEMA
    assert header["time"] == {"unit": "second"}
    assert "now" not in header  # unknown at run start
    assert trailer == {"final": True, "now": runner.now, "outcome": "completed"}
    # The streamed body matches the in-memory accumulation exactly.
    assert body == runner.observations


def test_object_empty_view_and_transport():
    # simple: SampleSource (creates an Object with no view schema) -> transport -> target.
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0, poll_interval=None, observe=True)
    runner.run()
    entries = runner.observations
    src = _by_node(entries, ("SampleSource",))
    tgt = _by_node(entries, ("SampleTarget",))
    # An Object-bearing port with no view fields is recorded as an empty view record.
    assert src["outputs"]["source_out"] == {"view": {}}
    assert tgt["inputs"]["target_in"] == {"view": {}}
    transports = [e for e in entries if e["kind"] == "transport"]
    assert transports and all("moved" in t and "arc" in t for t in transports)


def test_failed_run_trailer_outcome(tmp_path):
    # Inject a process failure (as test_failure does): SrcBad fails, the parallel
    # SrcSlow completes. Only completed activities are recorded; the trailer says
    # the run failed.
    path = tmp_path / "obs.yaml"
    runner = RollingRunner(FAIL_WF, FAIL_ENV, poll_interval=None, random_seed=0,
                           observation_out=str(path))
    runner.sim.schedule_process_failure("src_bad", "m0")
    runner.run()
    assert runner.failed
    docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    assert docs[-1] == {"final": True, "now": runner.now, "outcome": "failed"}
    recorded_nodes = {tuple(d["node"]) for d in docs if d.get("kind") == "processing"}
    assert ("SrcSlow",) in recorded_nodes  # completed -> recorded
    assert ("SrcBad",) not in recorded_nodes  # failed -> not recorded
    assert ("SinkBad",) not in recorded_nodes  # cancelled -> never ran -> not recorded
