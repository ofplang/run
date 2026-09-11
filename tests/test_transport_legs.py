"""Carrying one arc in several transport legs from a run: `--max-transport-legs`.

How many transport activities one Object-bearing arc may be moved in is the
scheduler's to offer and not part of the document (schedule SPEC §6.4.1). The
scheduler has taken the option since 0.6.0; until now there was no way to ask for it
from here, because `schedule_client.replan` names the keywords it passes and this one
was not among them -- so a laboratory that needs a hand-off station could be described
but not run.

What is pinned is that raising it changes the answer, not merely that the keyword
arrives: a route that needs a relay is unreachable at one leg and planned at two.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import JobRequest, RollingRunner, load_document  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

# loader -> pad -> heater -> pad -> output. The arm reaches each machine only from the
# pad, so every arc of the workflow needs two legs and a relay on the pad.
HANDOFF_ENV = {
    "time": {"unit": "second"},
    "devices": [
        {"id": "loader", "spots": ["stage"]},
        {"id": "pad", "spots": ["core"]},
        {"id": "heater", "spots": ["stage"]},
        {"id": "output", "spots": ["slot"]},
    ],
    "transporters": [{"id": "arm"}],
    "transports": [
        {"transporter": "arm", "from": "loader.stage", "to": "pad.core", "duration": 1},
        {"transporter": "arm", "from": "pad.core", "to": "heater.stage", "duration": 1},
        {"transporter": "arm", "from": "heater.stage", "to": "pad.core", "duration": 1},
        {"transporter": "arm", "from": "pad.core", "to": "output.slot", "duration": 1},
    ],
    "processes": {
        "heat": {
            "modes": [
                {
                    "devices": ["heater"],
                    "duration": 5,
                    "input_spots": {"plate": "heater.stage"},
                    "output_spots": {"out": "heater.stage"},
                }
            ]
        },
    },
}

BOUNDARY = {
    "boundary": {
        "inputs": {"sample": {"spot": "loader.stage"}},
        "outputs": {"result": {"spot": "output.slot"}},
    }
}


def _run(tmp_path, legs):
    env = tmp_path / "handoff.env.yaml"
    env.write_text(yaml.safe_dump(HANDOFF_ENV, sort_keys=False), encoding="utf-8")
    runner = RollingRunner(
        [
            JobRequest(
                id="job1",
                workflow=load_document(FIXTURES / "interface_load.workflow.yaml"),
                boundary=copy.deepcopy(BOUNDARY),
            )
        ],
        str(env),
        poll_interval=None,
        random_seed=0,
        max_transport_legs=legs,
    )
    return runner.run()


def test_one_leg_cannot_reach_across_the_hand_off_station(tmp_path):
    """Nothing is planned and nothing runs: no single move joins the two machines."""
    status = _run(tmp_path, 1)
    assert not [a for a in status["activities"] if a["kind"] == "transport"]


def test_two_legs_route_through_the_pad_and_the_run_completes(tmp_path):
    """The same environment, planned: each arc is carried in two moves joined by a
    relay on the pad, and the plate gets all the way to the output rack."""
    status = _run(tmp_path, 2)
    moves = [
        (a["from_spot"], a["to_spot"])
        for a in sorted(status["activities"], key=lambda x: x["start"])
        if a["kind"] == "transport"
    ]
    assert moves == [
        ("loader.stage", "pad.core"),
        ("pad.core", "heater.stage"),
        ("heater.stage", "pad.core"),
        ("pad.core", "output.slot"),
    ]
    assert len([a for a in status["activities"] if a["kind"] == "relay"]) == 2
    assert all(a["status"] == "completed" for a in status["activities"])
