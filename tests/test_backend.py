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
from ofplang.run.simulator import Simulator, VirtualTimeSimulator  # noqa: E402

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
        self._sim = VirtualTimeSimulator(environment)
        self.calls: list[str] = []
        self.advance_returns: list[int] = []
        self.transport_views: list = []

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

    def dispatch_transport(self, transporter, from_spot, to_spot, duration=None, view=None) -> str:
        self.calls.append("dispatch_transport")
        self.transport_views.append(view)
        return self._sim.dispatch_transport(
            transporter, from_spot, to_spot, duration=duration, view=view
        )

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


def test_transport_view_is_resolved_and_passed():
    # The runner resolves the moved Object's view (the producing arc endpoint's stored
    # output, D26) and passes it to dispatch_transport. simple.workflow has one real
    # transport (SampleSource.source_out -> SampleTarget), so the backend sees exactly
    # that view.
    captured: dict = {}

    def factory(env: dict) -> RecordingBackend:
        backend = RecordingBackend(env)
        captured["backend"] = backend
        return backend

    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, backend_factory=factory, random_seed=0)
    runner.run()

    backend = captured["backend"]
    assert runner.values.has(("SampleSource",), "source_out")
    expected = runner.values.get(("SampleSource",), "source_out")
    non_none = [v for v in backend.transport_views if v is not None]
    assert non_none, "the transport leg should have carried a resolved view"
    assert all(v == expected for v in non_none)


def test_simulator_dispatch_transport_records_view():
    # The simulator accepts and records the view (for a transport-running backend / a
    # test), and stays backward compatible when called without it.
    sim = VirtualTimeSimulator(SIMPLE_ENV)
    sim.place("station_0.core")
    uid = sim.dispatch_transport("transport", "station_0.core", "station_1.core", view={"k": 1})
    assert sim._ops[uid].view == {"k": 1}

    sim2 = VirtualTimeSimulator(SIMPLE_ENV)
    sim2.place("station_0.core")
    uid2 = sim2.dispatch_transport("transport", "station_0.core", "station_1.core")
    assert sim2._ops[uid2].view is None


def test_backend_factory_with_device_model_raises():
    with pytest.raises(RunnerError):
        RollingRunner(
            SIMPLE_WF,
            SIMPLE_ENV,
            backend_factory=lambda env: RecordingBackend(env),
            device_model=lambda *a, **k: {},
            random_seed=0,
        )
