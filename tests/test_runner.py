"""Tests for the runner (milestone 2a: replay a plan on the simulator).

These drive `Runner` with self-contained §6 plans against an inline environment,
checking that a feasible plan runs to completion and yields a well-formed status
(§6/§7), that boundary inputs and bookkeeping (relays / same-spot transports) are
handled, and that a physically inconsistent plan is caught by the backend oracle.
A final pair drives the CLI end to end through files.
"""

from __future__ import annotations

import textwrap

import pytest

from ofplang.run.cli import EXIT_FAILED, EXIT_OK, main
from ofplang.run.runner import Runner, RunnerError, load_document, serialize_document
from ofplang.run.simulator import MissingObject

# A source -> transport -> target environment (§5), matching the simulator tests.
ENV = {
    "time": {"unit": "second"},
    "devices": [
        {"id": "station_0", "spots": ["core"]},
        {"id": "station_1", "spots": ["core"]},
    ],
    "transporters": [{"id": "transport"}],
    "transports": [
        {"transporter": "transport", "from": "station_0.core", "to": "station_1.core", "duration": 1},
    ],
    "processes": {
        "source": {
            "modes": [
                {"devices": ["station_0"], "duration": 2, "output_spots": {"source_out": "station_0.core"}},
            ]
        },
        "target": {
            "modes": [
                {"devices": ["station_1"], "duration": 2, "input_spots": {"target_in": "station_1.core"}},
            ]
        },
    },
}


# Plan A: source produces an Object, a transport carries it, target consumes it.
# No interface (the source creates the entry Object). makespan 5.
PLAN_A = {
    "time": {"unit": "second"},
    "outcome": "optimal",
    "objective": {"kind": "makespan", "value": 5},
    "activities": [
        {"kind": "processing", "start": 0, "end": 2, "process": "source", "mode": "0", "node": ["Source"]},
        {
            "kind": "transport",
            "start": 2,
            "end": 3,
            "from_spot": "station_0.core",
            "to_spot": "station_1.core",
            "transporter": "transport",
            "arc": {"from": {"node": ["Source"], "port": "source_out"}, "to": {"node": ["Target"], "port": "target_in"}},
        },
        {"kind": "processing", "start": 3, "end": 5, "process": "target", "mode": "0", "node": ["Target"]},
    ],
}


# Plan B: a boundary input `sample` sits at station_0.core (interface), a transport
# carries it, target consumes it. makespan 3.
PLAN_B = {
    "time": {"unit": "second"},
    "interface": {"inputs": {"sample": "station_0.core"}},
    "activities": [
        {
            "kind": "transport",
            "start": 0,
            "end": 1,
            "from_spot": "station_0.core",
            "to_spot": "station_1.core",
            "transporter": "transport",
            "arc": {"from": {"node": [], "port": "sample"}, "to": {"node": ["Target"], "port": "target_in"}},
        },
        {"kind": "processing", "start": 1, "end": 3, "process": "target", "mode": "0", "node": ["Target"]},
    ],
}


def test_plan_a_runs_to_completion():
    status = Runner(PLAN_A, ENV).run()
    # Every activity is reported completed at its planned times.
    assert all(a["status"] == "completed" for a in status["activities"])
    assert [a["end"] for a in status["activities"]] == [2, 3, 5]
    # `now` sits at the makespan; provenance is preserved.
    assert status["now"] == 5
    assert status["activities"][0]["node"] == ["Source"]


def test_plan_a_leaves_nothing_resting():
    runner = Runner(PLAN_A, ENV)
    runner.run()
    # target consumed the Object; both spots are empty at the end.
    assert runner.sim.spot_state() == {}


def test_plan_b_seeds_interface_input_and_delivers():
    runner = Runner(PLAN_B, ENV)
    status = runner.run()
    assert all(a["status"] == "completed" for a in status["activities"])
    assert status["now"] == 3
    # The interface constraint is carried through unchanged (§6.8).
    assert status["interface"] == {"inputs": {"sample": "station_0.core"}}


def test_status_omits_solver_only_fields():
    # A status is not a solver output: it carries no `outcome` (§6.1).
    status = Runner(PLAN_A, ENV).run()
    assert "outcome" not in status
    assert status["time"] == {"unit": "second"}


def test_same_spot_transport_is_bookkeeping():
    # A zero-distance boundary input: the sample is already where target reads it,
    # so the transport is a same-spot no-op the runner never dispatches.
    plan = {
        "interface": {"inputs": {"sample": "station_1.core"}},
        "activities": [
            {
                "kind": "transport",
                "start": 0,
                "end": 0,
                "from_spot": "station_1.core",
                "to_spot": "station_1.core",
                "arc": {"from": {"node": [], "port": "sample"}, "to": {"node": ["Target"], "port": "target_in"}},
            },
            {"kind": "processing", "start": 0, "end": 2, "process": "target", "mode": "0", "node": ["Target"]},
        ],
    }
    runner = Runner(plan, ENV)
    status = runner.run()
    assert all(a["status"] == "completed" for a in status["activities"])
    # The same-spot transport never became a backend operation.
    assert len(runner.sim.observe()) == 1  # only the target processing


def test_relay_is_bookkeeping():
    # A relay carries no physical operation; it is echoed completed without a
    # dispatch. (Two real legs move station_0 -> station_1 -> ... would need more
    # devices; here we only assert the relay itself is not dispatched.)
    plan = {
        "interface": {"inputs": {"sample": "station_0.core"}},
        "activities": [
            {
                "kind": "transport",
                "start": 0,
                "end": 1,
                "seq": 0,
                "from_spot": "station_0.core",
                "to_spot": "station_1.core",
                "transporter": "transport",
                "arc": {"from": {"node": [], "port": "sample"}, "to": {"node": ["Target"], "port": "target_in"}},
            },
            {
                "kind": "relay",
                "start": 1,
                "end": 1,
                "seq": 1,
                "spot": "station_1.core",
                "arc": {"from": {"node": [], "port": "sample"}, "to": {"node": ["Target"], "port": "target_in"}},
            },
            {"kind": "processing", "start": 1, "end": 3, "process": "target", "mode": "0", "node": ["Target"]},
        ],
    }
    runner = Runner(plan, ENV)
    status = runner.run()
    assert all(a["status"] == "completed" for a in status["activities"])
    # One transport + one processing dispatched; the relay is not an operation.
    assert len(runner.sim.observe()) == 2


def test_inconsistent_plan_is_caught_by_backend():
    # A transport whose source spot was never filled (no producer, no interface):
    # the backend oracle rejects the dispatch (D16), surfacing as a failure.
    plan = {
        "activities": [
            {
                "kind": "transport",
                "start": 0,
                "end": 1,
                "from_spot": "station_0.core",
                "to_spot": "station_1.core",
                "transporter": "transport",
                "arc": {"from": {"node": ["A"], "port": "p"}, "to": {"node": ["B"], "port": "q"}},
            },
        ],
    }
    with pytest.raises(MissingObject):
        Runner(plan, ENV).run()


def test_empty_plan_produces_empty_status():
    status = Runner({"activities": []}, ENV).run()
    assert status["activities"] == []
    assert status["now"] == 0


def test_history_confirms_actual_times_match_plan():
    # Genuine actual-vs-planned check: read the backend's completion history (real
    # event times, via _history) and confirm each dispatched activity finished when
    # the plan said it would -- evidence the run matched the plan, not just that the
    # runner echoed the plan's times into the status. The runner's own main loop
    # never touches these times (it only calls advance).
    runner = Runner(PLAN_A, ENV)
    runner.run()
    actual = {e.uuid: e.time for e in runner.sim._history()}
    dispatched = [r for r in runner._records if r.dispatched]
    assert len(dispatched) == 3  # source, transport, target (no bookkeeping in PLAN_A)
    for rec in dispatched:
        assert actual[rec.uuid] == rec.end  # finished exactly when planned


def test_status_round_trips_through_yaml():
    status = Runner(PLAN_A, ENV).run()
    text = serialize_document(status)
    assert "status: completed" in text
    # It parses back to an equivalent document.
    import yaml

    assert yaml.safe_load(text) == status


# -- CLI end to end --------------------------------------------------------

def _write_env(tmp_path):
    path = tmp_path / "env.yaml"
    path.write_text(
        textwrap.dedent(
            """
            time:
              unit: second
            devices:
              - id: station_0
                spots: [core]
              - id: station_1
                spots: [core]
            transporters:
              - id: transport
            transports:
              - { transporter: transport, from: station_0.core, to: station_1.core, duration: 1 }
            processes:
              source:
                modes:
                  - devices: [station_0]
                    duration: 2
                    output_spots: { source_out: station_0.core }
              target:
                modes:
                  - devices: [station_1]
                    duration: 2
                    input_spots: { target_in: station_1.core }
            """
        ),
        encoding="utf-8",
    )
    return path


def test_cli_runs_plan_to_file(tmp_path, capsys):
    env_path = _write_env(tmp_path)
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(serialize_document(PLAN_A), encoding="utf-8")
    out_path = tmp_path / "status.yaml"

    code = main(["run", str(plan_path), "--env", str(env_path), "-o", str(out_path)])
    assert code == EXIT_OK

    status = load_document(out_path)
    assert status["now"] == 5
    assert all(a["status"] == "completed" for a in status["activities"])


def test_cli_reports_failure_on_inconsistent_plan(tmp_path, capsys):
    env_path = _write_env(tmp_path)
    plan_path = tmp_path / "bad.yaml"
    # A transport from an empty source spot: the backend rejects it.
    bad = {
        "activities": [
            {
                "kind": "transport",
                "start": 0,
                "end": 1,
                "from_spot": "station_0.core",
                "to_spot": "station_1.core",
                "transporter": "transport",
                "arc": {"from": {"node": ["A"], "port": "p"}, "to": {"node": ["B"], "port": "q"}},
            }
        ]
    }
    plan_path.write_text(serialize_document(bad), encoding="utf-8")

    code = main(["run", str(plan_path), "--env", str(env_path)])
    assert code == EXIT_FAILED
    assert "execution failed" in capsys.readouterr().err
