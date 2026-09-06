"""Running a move that no transporter carries (schedule SPEC §4.6 / §5.4).

An environment may declare a route with `transporter: null`: a device shifting
material between its own spots, a chute. The scheduler plans it like any other
transport -- it takes time and holds its endpoint devices -- and reports it with a
null `transporter`. This is the runner executing one, end to end: it plans, it
dispatches, the material moves, and no transporter is held, there being none.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402
from ofplang.run.simulator import UnknownReference, VirtualTimeSimulator  # noqa: E402

WORKFLOW = {
    "spec_version": "0.0",
    "types": {"Plate": {"domain": "object"}},
    "processes": {
        "dispense": {
            "kind": "atomic",
            "outputs": {"plate": {"type": "Plate", "phase": "data"}},
            "objects": {"create": ["outputs.plate"]},
        },
        "thermal_cycle": {
            "kind": "atomic",
            "inputs": {"plate": {"type": "Plate", "phase": "data"}},
            "objects": {"consume": ["inputs.plate"]},
        },
        "main": {
            "kind": "composite",
            "body": {
                "nodes": [
                    {"id": "Fill", "process": "dispense"},
                    {"id": "Cyc", "process": "thermal_cycle",
                     "state": {"plate": {"from": "Fill.plate"}}},
                ]
            },
        },
    },
    "entry": "main",
}

# The cycler is reachable by nothing: it loads its own block from its own door.
# There is no `transporters` section at all, which is the point -- an environment
# that needs none does not have to invent one.
ENV = {
    "time": {"unit": "second"},
    "devices": [{"id": "cycler", "spots": ["door", "block"]}],
    "transports": [
        {"transporter": None, "from": "cycler.door", "to": "cycler.block", "duration": 5}
    ],
    "processes": {
        "dispense": {"modes": [{"devices": ["cycler"], "duration": 30,
                                "output_spots": {"plate": "cycler.door"}}]},
        "thermal_cycle": {"modes": [{"devices": ["cycler"], "duration": 300,
                                     "input_spots": {"plate": "cycler.block"}}]},
    },
}


def test_a_transporter_less_move_is_planned_and_run():
    run = RollingRunner(copy.deepcopy(WORKFLOW), copy.deepcopy(ENV),
                        poll_interval=None, random_seed=0)
    status = run.run()
    assert not run.failed

    moves = [a for a in status["activities"] if a["kind"] == "transport"]
    assert len(moves) == 1
    move = moves[0]
    assert move["status"] == "completed"
    assert (move["from_spot"], move["to_spot"]) == ("cycler.door", "cycler.block")
    assert move["end"] - move["start"] == 5
    # Written, and written as null: the field is what says nothing carried it.
    assert "transporter" in move and move["transporter"] is None
    # 30 + 5 + 300, serial -- the move holds the cycler like any other transport
    # holds its endpoint devices, so nothing overlaps it.
    assert max(a["end"] for a in status["activities"]) == 335


def test_the_move_needs_a_declared_route_like_any_other():
    """Reverse of the declared one, which the table does not have."""
    sim = VirtualTimeSimulator(copy.deepcopy(ENV))
    sim.place("cycler.block")
    with pytest.raises(UnknownReference):
        sim.dispatch_transport(None, "cycler.block", "cycler.door")
