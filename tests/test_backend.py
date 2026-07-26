"""Tests for the injectable execution backend seam (the `Backend` protocol).

`RollingRunner` drives whatever `backend_factory(environment)` returns, through the
`Backend` protocol alone -- not the concrete `Simulator`. These tests pin that
seam: a custom backend (here a thin recording wrapper over a `Simulator`) drives a
real workflow to completion, the runner touches only the protocol surface, the
factory is handed the mode-id-normalized environment, and the time `advance`
returns is adopted as `now` (the contract a real, wall-clock backend relies on).

The scheduler is a required dependency for a full run; if it is not installed
these tests are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner, RunnerError  # noqa: E402
from ofplang.run.simulator import Simulator  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")

# The methods that make up the Backend protocol -- the only surface the runner is
# allowed to touch on an injected backend.
_PROTOCOL_METHODS = frozenset(
    {
        "advance",
        "down_devices",
        "place",
        "dispatch_processing",
        "dispatch_transport",
        "state",
        "spot_state",
    }
)


class RecordingBackend:
    """A `Backend` that delegates to an inner `Simulator` while recording which
    protocol methods the runner calls and every value `advance` returns. It is not
    a `Simulator` instance, so a completed run proves the runner drives it purely
    through the protocol."""

    def __init__(self, environment: dict):
        self.environment = environment
        self._sim = Simulator(environment)
        self.calls: list[str] = []
        self.advance_returns: list[int] = []

    def advance(self, until: int) -> int:
        self.calls.append("advance")
        reached = self._sim.advance(until)
        self.advance_returns.append(reached)
        return reached

    def down_devices(self) -> list[str]:
        self.calls.append("down_devices")
        return self._sim.down_devices()

    def place(self, spot: str, obj_id: str | None = None) -> str:
        self.calls.append("place")
        return self._sim.place(spot, obj_id)

    def dispatch_processing(self, process, mode, duration=None, output_schema=None,
                            inputs=None, definition=None) -> str:
        self.calls.append("dispatch_processing")
        return self._sim.dispatch_processing(
            process, mode, duration=duration, output_schema=output_schema,
            inputs=inputs, definition=definition,
        )

    def dispatch_transport(self, transporter, from_spot, to_spot, duration=None) -> str:
        self.calls.append("dispatch_transport")
        return self._sim.dispatch_transport(transporter, from_spot, to_spot, duration=duration)

    def state(self, uuid: str) -> dict:
        self.calls.append("state")
        return self._sim.state(uuid)

    def spot_state(self, spot: str | None = None):
        self.calls.append("spot_state")
        return self._sim.spot_state(spot)


def test_injected_backend_drives_run_through_protocol():
    captured: dict = {}

    def factory(env: dict) -> RecordingBackend:
        backend = RecordingBackend(env)
        captured["backend"] = backend
        return backend

    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, backend_factory=factory, random_seed=0)
    status = runner.run()

    backend = captured["backend"]
    # The run completed, driving the injected (non-Simulator) backend.
    assert runner.sim is backend
    assert not isinstance(runner.sim, Simulator)
    assert all(a["status"] == "completed" for a in status["activities"])
    assert status["now"] == 5  # same makespan as the default-Simulator run
    # The runner touched only the protocol surface.
    assert set(backend.calls) <= _PROTOCOL_METHODS


def test_injected_backend_matches_default_simulator_result():
    default_status = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0).run()
    injected_status = RollingRunner(
        SIMPLE_WF, SIMPLE_ENV, backend_factory=lambda env: RecordingBackend(env), random_seed=0
    ).run()
    assert injected_status["now"] == default_status["now"]
    assert (
        {a["status"] for a in injected_status["activities"]}
        == {a["status"] for a in default_status["activities"]}
    )


def test_factory_receives_mode_id_normalized_environment():
    seen: dict = {}

    def factory(env: dict) -> RecordingBackend:
        seen["env"] = env
        return RecordingBackend(env)

    RollingRunner(SIMPLE_WF, SIMPLE_ENV, backend_factory=factory, random_seed=0).run()

    # The factory is handed the runner's normalized environment: every process mode
    # carries an explicit id (pinned up front so ids stay stable across reduction).
    for process in (seen["env"].get("processes") or {}).values():
        for mode in process.get("modes") or []:
            assert mode.get("id") is not None


def test_advance_return_is_adopted_as_now():
    backend = {}

    def factory(env: dict) -> RecordingBackend:
        backend["b"] = RecordingBackend(env)
        return backend["b"]

    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, backend_factory=factory, random_seed=0)
    runner.run()

    # `self.now` is only ever set from what `advance` returns (besides the initial
    # 0), so after the run it equals the backend's last advance return.
    returns = backend["b"].advance_returns
    assert returns
    assert runner.now == returns[-1]


def test_backend_factory_with_device_model_raises():
    with pytest.raises(RunnerError):
        RollingRunner(
            SIMPLE_WF,
            SIMPLE_ENV,
            backend_factory=lambda env: RecordingBackend(env),
            device_model=lambda *a, **k: {},
            random_seed=0,
        )
