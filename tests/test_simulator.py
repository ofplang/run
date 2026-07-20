"""Tests for the simulated execution backend (`ofplang.run.simulator`).

These exercise the physical model and the dispatch contract (dev-notes design.md
D10-D18): resource occupation and release, spot / object effects, the validating-
oracle preconditions (D16), status-only observation (D18), the two-stage clock
advance (D11), and a happy-path end-to-end run (D17). The environments are built
inline so the tests are self-contained.
"""

from __future__ import annotations

import textwrap

import pytest

from ofplang.run.simulator import (
    ClockError,
    Environment,
    MissingObject,
    RelayNotSupported,
    ResourceBusy,
    Simulator,
    SpotConflict,
    UnknownReference,
    environment_from_dict,
    load_environment,
)

# A minimal source -> transport -> target environment, mirroring the shape of
# ofplang-schedule's `simple` example (§5): two devices with one spot each, a
# single transporter, and one route.
SIMPLE_ENV = {
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
        # `source` has no id -> mode "0"; produces an Object at station_0.core.
        "source": {
            "modes": [
                {"devices": ["station_0"], "duration": 2, "output_spots": {"source_out": "station_0.core"}},
            ]
        },
        # `target` consumes an Object at station_1.core; no output.
        "target": {
            "modes": [
                {"devices": ["station_1"], "duration": 2, "input_spots": {"target_in": "station_1.core"}},
            ]
        },
        # In-place transform: reads and writes the same spot.
        "cook": {
            "modes": [
                {
                    "id": "fast",
                    "devices": ["station_0"],
                    "duration": 3,
                    "input_spots": {"in": "station_0.core"},
                    "output_spots": {"out": "station_0.core"},
                },
            ]
        },
        # Pure Data: duration only, no device / spot (D12).
        "compute": {"modes": [{"id": "v1", "duration": 5}]},
    },
}


def make_sim(device_model=None) -> Simulator:
    return Simulator(SIMPLE_ENV, device_model=device_model)


# -- environment loading ---------------------------------------------------

def test_environment_from_dict_shape():
    env = environment_from_dict(SIMPLE_ENV)
    assert env.time_unit == "second"
    assert env.devices["station_0"] == ("core",)
    assert env.spots == {"station_0.core", "station_1.core"}
    assert env.transporters == {"transport"}
    assert env.transports[("transport", "station_0.core", "station_1.core")] == 1
    # Auto-assigned vs explicit mode ids (§5.5).
    assert set(env.processes["source"].modes) == {"0"}
    assert set(env.processes["cook"].modes) == {"fast"}
    assert env.processes["compute"].modes["v1"].devices == ()


def test_load_environment_from_file(tmp_path):
    path = tmp_path / "env.yaml"
    path.write_text(
        textwrap.dedent(
            """
            time:
              unit: minute
            devices:
              - id: dev_0
                spots: [a, b]
            processes:
              noop:
                modes:
                  - duration: 4
            """
        ),
        encoding="utf-8",
    )
    env = load_environment(path)
    assert isinstance(env, Environment)
    assert env.time_unit == "minute"
    assert env.spots == {"dev_0.a", "dev_0.b"}
    assert env.processes["noop"].modes["0"].duration == 4


# -- processing ------------------------------------------------------------

def test_processing_happy_path_produces_object():
    sim = make_sim()
    uid = sim.dispatch_processing("source", "0")
    assert sim.state(uid) == {"status": "running"}
    assert sim.spot_state("station_0.core") is None  # not produced until completion

    sim.advance(2)
    assert sim.now == 2
    assert sim.state(uid) == {"status": "completed"}
    # A fresh opaque object now rests in the output spot.
    assert sim.spot_state("station_0.core") is not None


def test_processing_duration_override():
    # The mode's own duration is 2; an override injects a different one (D13).
    sim = make_sim()
    uid = sim.dispatch_processing("source", "0", duration=10)
    sim.advance(50)
    (event,) = sim._history()
    assert event.time == 10 and event.uuid == uid


# Value-shape descriptors (D27 F2), the runner/backend seam contract.
_INT = {"kind": "primitive", "name": "Int"}
_FLOAT = {"kind": "primitive", "name": "Float"}
_BOOL = {"kind": "primitive", "name": "Bool"}
_STRING = {"kind": "primitive", "name": "String"}


def test_value_seam_reveals_typed_outputs_only_when_signed_and_completed():
    # The value seam (D26/D27): dispatched with an `output_schema`, a completed
    # operation reveals a typed value per output port (walked from the descriptor);
    # while running it stays status-only, and a dispatch with no signature is
    # unaffected (backward compat).
    sim = make_sim()
    uid = sim.dispatch_processing("source", "0", output_schema={"source_out": _INT})
    assert sim.state(uid) == {"status": "running"}  # no outputs before completion

    sim.advance(2)
    st = sim.state(uid)
    assert st == {"status": "completed", "outputs": {"source_out": 0}}  # Int default
    assert sim.observe()[uid] == st

    # A legacy dispatch (no signature) never grows an `outputs` key. `cook` is an
    # in-place transform on station_0.core, which `source` just filled.
    other = sim.dispatch_processing("cook", "fast")  # no output_schema
    assert sim.state(other) == {"status": "running"}


def test_value_seam_no_outputs_on_failure():
    # A failed operation applies no material effect (D25) and produces no outputs,
    # even when dispatched with a signature.
    sim = make_sim()
    sim.schedule_process_failure("source", "0")
    uid = sim.dispatch_processing("source", "0", output_schema={"source_out": _INT})
    sim.advance(2)
    assert sim.state(uid) == {"status": "failed"}  # no `outputs` key


def test_value_seam_empty_signature_yields_empty_outputs():
    # A pure-consume processing (no output ports) still carries a signature, so its
    # completed view has an explicit empty `outputs` (distinct from an unsigned op).
    sim = make_sim()
    sim.place("station_1.core")
    uid = sim.dispatch_processing("target", "0", output_schema={})
    sim.advance(2)
    assert sim.state(uid) == {"status": "completed", "outputs": {}}


def test_value_seam_device_less_pure_data_op_generates_outputs():
    # A device-less Pure Data op (no spot, no device) still generates a typed value
    # -- the seam is independent of physical occupancy (D26).
    sim = make_sim()
    uid = sim.dispatch_processing("compute", "v1", output_schema={"score": _FLOAT})
    sim.advance(5)
    assert sim.state(uid) == {"status": "completed", "outputs": {"score": 0.0}}


def test_value_seam_zero_duration_op_reveals_outputs_after_settle():
    # A zero-duration signed op produces its outputs once the clock settles past it.
    sim = make_sim()
    uid = sim.dispatch_processing("compute", "v1", duration=0, output_schema={"x": _BOOL})
    assert sim.state(uid) == {"status": "running"}  # not settled yet
    sim.advance(0)
    assert sim.state(uid) == {"status": "completed", "outputs": {"x": False}}


def test_value_seam_generates_typed_defaults_per_shape():
    # Each descriptor shape yields its typed default: primitives, an empty array,
    # and a record of view-field defaults (a non-empty view).
    sim = make_sim()
    schema = {
        "n": _INT,
        "label": _STRING,
        "flags": {"kind": "array", "element": _BOOL},
        "reading": {"kind": "record", "fields": {"mean": _FLOAT, "count": _INT}},
    }
    uid = sim.dispatch_processing("compute", "v1", output_schema=schema)
    sim.advance(5)
    assert sim.state(uid)["outputs"] == {
        "n": 0,
        "label": "",
        "flags": [],
        "reading": {"mean": 0.0, "count": 0},
    }


def test_observe_mixes_signed_and_unsigned_and_running_ops():
    # observe() reports every op: a signed completed op carries outputs, an unsigned
    # completed op does not, and a running op is status-only regardless.
    sim = make_sim()
    signed = sim.dispatch_processing("source", "0", output_schema={"source_out": _INT})  # dur 2
    unsigned = sim.dispatch_processing("compute", "v1")  # dur 5, no signature
    sim.advance(2)  # source completes; compute still running
    obs = sim.observe()
    assert obs[signed] == {"status": "completed", "outputs": {"source_out": 0}}
    assert obs[unsigned] == {"status": "running"}


def test_device_model_computes_outputs_from_inputs():
    # An installed device model computes a signed op's outputs from its inputs
    # (D27 F4b), instead of the input-independent typed defaults. It is called with
    # the capability (process, mode), the inputs, and the output schema.
    seen = []

    def model(process, mode, inputs, output_schema, definition):
        seen.append((process, mode, dict(inputs), sorted(output_schema), definition))
        return {port: inputs.get("seed", 0) + 1 for port in output_schema}

    sim = make_sim(model)
    uid = sim.dispatch_processing(
        "source", "0", output_schema={"source_out": _INT}, inputs={"seed": 41}, definition={"kind": "atomic"}
    )
    sim.advance(2)
    assert sim.state(uid)["outputs"] == {"source_out": 42}  # computed, not the default 0
    # The model is handed the capability, inputs, output schema, and the raw process
    # definition, which the simulator passes through unchanged.
    assert seen == [("source", "0", {"seed": 41}, ["source_out"], {"kind": "atomic"})]


def test_device_model_left_default_when_absent():
    # Without a model the backend uses the built-in default: a typed default for an
    # unmapped output (source_out is not in any objects.map here).
    sim = make_sim()  # no device model
    uid = sim.dispatch_processing("source", "0", output_schema={"source_out": _INT}, inputs={"seed": 41})
    sim.advance(2)
    assert sim.state(uid)["outputs"] == {"source_out": 0}


def test_default_device_model_defaults_and_carries_mapped_objects():
    # The built-in default (D27): type defaults for every output, plus carrying each
    # Object output declared in the process's objects.map through from its input.
    from ofplang.run.simulator import default_device_model

    output_schema = {"y": _INT, "plate": {"kind": "record", "fields": {"barcode": _STRING}}}
    definition = {"objects": {"map": {"outputs.plate": "inputs.plate"}}}
    inputs = {"x": 5, "plate": {"barcode": "ABC"}}
    # y is unmapped -> its type default (0); plate is mapped -> carried from inputs.
    assert default_device_model("step", "0", inputs, output_schema, definition) == {
        "y": 0, "plate": {"barcode": "ABC"},
    }
    # No objects.map (or no definition) -> pure type defaults.
    assert default_device_model("p", "0", {}, {"y": _INT}, {"kind": "atomic"}) == {"y": 0}
    assert default_device_model("p", "0", {}, {"y": _INT}, None) == {"y": 0}


def test_processing_in_place_transform_keeps_spot():
    sim = make_sim()
    obj = sim.place("station_0.core")  # seed material
    uid = sim.dispatch_processing("cook", "fast")  # in-place: input == output spot
    sim.advance(3)
    assert sim.state(uid) == {"status": "completed"}
    held = sim.spot_state("station_0.core")
    # Still occupied, but with a regenerated id (identity is not tracked, D15).
    assert held is not None and held != obj


def test_processing_missing_input_is_error():
    sim = make_sim()
    with pytest.raises(MissingObject):
        sim.dispatch_processing("cook", "fast")  # station_0.core is empty


def test_processing_output_occupied_is_error():
    sim = make_sim()
    sim.place("station_0.core")  # occupy the output spot of `source`
    with pytest.raises(SpotConflict):
        sim.dispatch_processing("source", "0")


def test_processing_device_busy_is_error():
    sim = make_sim()
    sim.dispatch_processing("source", "0")  # occupies station_0
    with pytest.raises(ResourceBusy):
        # `cook` also needs station_0; it is busy until `source` completes.
        sim.dispatch_processing("cook", "fast")


def test_pure_data_processing_occupies_nothing():
    sim = make_sim()
    uid = sim.dispatch_processing("compute", "v1")
    assert sim.spot_state() == {}  # no spot touched
    sim.advance(5)
    assert sim.state(uid) == {"status": "completed"}


def test_pure_data_zero_duration_processing_settles_at_dispatch():
    # A device-less Pure-Data process may take zero time (ofplang-schedule now
    # allows a device-less mode duration of 0, §5.5). It occupies nothing and
    # completes as soon as the clock is advanced to its dispatch time -- like a
    # same-spot transport, no clock movement is needed to settle it.
    sim = make_sim()
    uid = sim.dispatch_processing("compute", "v1", duration=0)
    assert sim.state(uid) == {"status": "running"}  # not yet settled
    assert sim.spot_state() == {}  # no spot touched
    sim.advance(0)  # settle at the current time, without advancing the clock
    assert sim.state(uid) == {"status": "completed"}


def test_unknown_process_and_mode():
    sim = make_sim()
    with pytest.raises(UnknownReference):
        sim.dispatch_processing("nope", "0")
    with pytest.raises(UnknownReference):
        sim.dispatch_processing("source", "9")


# -- transport -------------------------------------------------------------

def test_transport_happy_path_moves_object():
    sim = make_sim()
    obj = sim.place("station_0.core")
    uid = sim.dispatch_transport("transport", "station_0.core", "station_1.core")
    # Both endpoint devices and the transporter are occupied during the move.
    with pytest.raises(ResourceBusy):
        sim.dispatch_transport("transport", "station_0.core", "station_1.core")

    sim.advance(1)  # table duration is 1
    assert sim.state(uid) == {"status": "completed"}
    assert sim.spot_state("station_0.core") is None  # source freed
    assert sim.spot_state("station_1.core") == obj  # same id carried over


def test_transport_duration_from_table():
    sim = make_sim()
    sim.place("station_0.core")
    uid = sim.dispatch_transport("transport", "station_0.core", "station_1.core")
    sim.advance(100)
    (event,) = sim._history()
    assert event.time == 1  # from the transport table, not overridden


def test_transport_missing_source_is_error():
    sim = make_sim()
    with pytest.raises(MissingObject):
        sim.dispatch_transport("transport", "station_0.core", "station_1.core")


def test_transport_destination_occupied_is_error():
    sim = make_sim()
    sim.place("station_0.core")
    sim.place("station_1.core")
    with pytest.raises(SpotConflict):
        sim.dispatch_transport("transport", "station_0.core", "station_1.core")


def test_transport_unknown_route_is_error():
    sim = make_sim()
    sim.place("station_1.core")
    # station_1.core -> station_0.core has no table entry: that move is impossible.
    with pytest.raises(UnknownReference):
        sim.dispatch_transport("transport", "station_1.core", "station_0.core")


def test_transport_unknown_spot_is_error():
    sim = make_sim()
    with pytest.raises(UnknownReference):
        sim.dispatch_transport("transport", "station_0.core", "nowhere.core")


def test_same_spot_transport_is_noop():
    sim = make_sim()
    obj = sim.place("station_0.core")
    # A same-spot move needs no transporter and has duration 0 (§5.4 / §6.4).
    uid = sim.dispatch_transport(None, "station_0.core", "station_0.core")
    sim.advance(0)
    assert sim.state(uid) == {"status": "completed"}
    assert sim.spot_state("station_0.core") == obj  # unchanged


def test_real_transport_requires_transporter():
    sim = make_sim()
    sim.place("station_0.core")
    with pytest.raises(ValueError):
        sim.dispatch_transport(None, "station_0.core", "station_1.core")


def test_transporter_busy_across_independent_devices():
    # A dedicated env: two disjoint device pairs sharing one transporter, so a
    # second move conflicts on the transporter alone (not on any device).
    env = {
        "time": {"unit": "second"},
        "devices": [
            {"id": "a", "spots": ["s"]},
            {"id": "b", "spots": ["s"]},
            {"id": "c", "spots": ["s"]},
            {"id": "d", "spots": ["s"]},
        ],
        "transporters": [{"id": "arm"}],
        "transports": [
            {"transporter": "arm", "from": "a.s", "to": "b.s", "duration": 2},
            {"transporter": "arm", "from": "c.s", "to": "d.s", "duration": 2},
        ],
        "processes": {},
    }
    sim = Simulator(env)
    sim.place("a.s")
    sim.place("c.s")
    sim.dispatch_transport("arm", "a.s", "b.s")  # holds the arm
    with pytest.raises(ResourceBusy):
        sim.dispatch_transport("arm", "c.s", "d.s")  # devices free, arm busy


# -- relay -----------------------------------------------------------------

def test_relay_is_rejected():
    sim = make_sim()
    with pytest.raises(RelayNotSupported):
        sim.dispatch_relay("station_0.core")


# -- clock advance ---------------------------------------------------------

def test_advance_reaches_until_without_events():
    sim = make_sim()
    sim.advance(7)  # nothing dispatched
    assert sim._history() == []
    assert sim.now == 7


def test_advance_does_not_return_early_on_event():
    # A completion at t=2 does not stop advance short of `until` (D11).
    sim = make_sim()
    sim.dispatch_processing("source", "0")  # completes at 2
    sim.advance(10)
    assert sim.now == 10


def test_advance_backwards_is_error():
    sim = make_sim()
    sim.advance(5)
    with pytest.raises(ClockError):
        sim.advance(4)


def test_advance_orders_events_by_completion_time():
    # Two ops on different devices, different durations; events come out in
    # completion order.
    sim = make_sim()
    slow = sim.dispatch_processing("source", "0", duration=8)  # station_0, ends 8
    # A second op on station_1 via a same-spot no-op transport (needs material).
    sim.place("station_1.core")
    quick = sim.dispatch_transport(None, "station_1.core", "station_1.core", duration=3)
    sim.advance(20)
    events = sim._history()
    assert [e.uuid for e in events] == [quick, slow]
    assert [e.time for e in events] == [3, 8]


def test_advance_ties_are_deterministic_in_dispatch_order():
    # Two independent same-spot no-ops ending at the same time complete in the
    # order they were dispatched.
    sim = make_sim()
    sim.place("station_0.core")
    sim.place("station_1.core")
    first = sim.dispatch_transport(None, "station_0.core", "station_0.core", duration=4)
    second = sim.dispatch_transport(None, "station_1.core", "station_1.core", duration=4)
    sim.advance(4)
    assert [e.uuid for e in sim._history()] == [first, second]


def test_history_accumulates_across_advances():
    # _history is cumulative: every advance appends its completion events, so the
    # main loop only ever calls advance while a test reads the full run log.
    sim = make_sim()
    a = sim.dispatch_processing("source", "0")  # ends 2
    sim.place("station_1.core")
    b = sim.dispatch_transport(None, "station_1.core", "station_1.core", duration=5)  # ends 5
    sim.advance(2)
    assert [e.uuid for e in sim._history()] == [a]
    sim.advance(5)
    assert [(e.uuid, e.time) for e in sim._history()] == [(a, 2), (b, 5)]


# -- observation & placement ----------------------------------------------

def test_observe_returns_all_statuses():
    sim = make_sim()
    a = sim.dispatch_processing("source", "0")  # ends 2
    sim.place("station_1.core")
    b = sim.dispatch_transport(None, "station_1.core", "station_1.core", duration=5)
    sim.advance(2)
    obs = sim.observe()
    assert obs[a] == {"status": "completed"}
    assert obs[b] == {"status": "running"}


def test_state_unknown_operation_is_error():
    sim = make_sim()
    with pytest.raises(UnknownReference):
        sim.state("op-999")


def test_place_onto_occupied_is_error():
    sim = make_sim()
    sim.place("station_0.core")
    with pytest.raises(SpotConflict):
        sim.place("station_0.core")


def test_place_unknown_spot_is_error():
    sim = make_sim()
    with pytest.raises(UnknownReference):
        sim.place("ghost.core")


def test_remove_empty_spot_is_error():
    sim = make_sim()
    with pytest.raises(MissingObject):
        sim.remove("station_0.core")


def test_place_with_explicit_id_then_remove():
    sim = make_sim()
    sim.place("station_0.core", obj_id="plate_42")
    assert sim.spot_state("station_0.core") == "plate_42"
    assert sim.remove("station_0.core") == "plate_42"
    assert sim.spot_state("station_0.core") is None


# -- end-to-end (happy path, D17) -----------------------------------------

def test_end_to_end_source_transport_target():
    """Drive the full `simple` chain: source produces an Object, a transport
    carries it, target consumes it -- the happy-path integration the first
    simulator milestone exists to support (D17)."""
    sim = make_sim()

    # source runs on station_0 and leaves an Object at station_0.core.
    src = sim.dispatch_processing("source", "0")
    sim.advance(2)
    assert sim.state(src) == {"status": "completed"}
    obj = sim.spot_state("station_0.core")
    assert obj is not None

    # transport carries it to station_1.core.
    mv = sim.dispatch_transport("transport", "station_0.core", "station_1.core")
    sim.advance(3)  # dispatched at t=2, table duration 1 -> done by t=3
    assert sim.state(mv) == {"status": "completed"}
    assert sim.spot_state("station_0.core") is None
    assert sim.spot_state("station_1.core") == obj

    # target consumes it; the spot is emptied on completion.
    tgt = sim.dispatch_processing("target", "0")
    sim.advance(5)
    assert sim.state(tgt) == {"status": "completed"}
    assert sim.spot_state() == {}  # nothing left resting anywhere


# -- injected failure (D25) ------------------------------------------------

def test_processing_failure_frees_resources_and_skips_effect():
    # A failing (process, mode) is declared up front (before dispatch). The op runs
    # its duration, then ends `failed` -- resources freed, but no material effect.
    sim = make_sim()
    sim.schedule_process_failure("source", "0")
    uid = sim.dispatch_processing("source", "0")
    assert sim.state(uid) == {"status": "running"}
    sim.advance(2)  # source duration 2
    assert sim.state(uid) == {"status": "failed"}
    assert sim.spot_state() == {}  # output NOT produced (no material effect)
    # The failure event is visible in the history channel with its terminal status.
    assert sim._history()[-1].status == "failed"
    # Resources released: station_0 is free, so a fresh op dispatches without error.
    uid2 = sim.dispatch_processing("source", "0")
    assert sim.state(uid2) == {"status": "running"}


def test_transport_failure_leaves_object_at_source():
    # A failing transport does not move its object: the source keeps it and the
    # destination stays empty, and the transporter / devices are freed.
    sim = make_sim()
    obj = sim.place("station_0.core")
    sim.schedule_transport_failure("transport", "station_0.core", "station_1.core")
    uid = sim.dispatch_transport("transport", "station_0.core", "station_1.core")
    sim.advance(1)  # route duration 1
    assert sim.state(uid) == {"status": "failed"}
    assert sim.spot_state("station_0.core") == obj  # not moved
    assert sim.spot_state("station_1.core") is None  # never arrived


def test_only_the_failing_capability_fails():
    # A failing capability does not taint others: source fails, but transport of a
    # separately-placed object still completes normally.
    sim = make_sim()
    sim.schedule_process_failure("source", "0")
    src = sim.dispatch_processing("source", "0")
    obj = sim.place("station_1.core")  # unrelated material to move back
    sim.advance(2)
    assert sim.state(src) == {"status": "failed"}
    # A non-failing transport completes and moves its object.
    mv = sim.dispatch_transport("transport", "station_1.core", "station_1.core")  # same-spot no-op
    sim.advance(2)
    assert sim.state(mv) == {"status": "completed"}
    assert sim.spot_state("station_1.core") == obj


def test_schedule_failure_unknown_targets_error():
    sim = make_sim()
    with pytest.raises(UnknownReference):
        sim.schedule_process_failure("nope", "0")
    with pytest.raises(UnknownReference):
        sim.schedule_process_failure("source", "9")
    with pytest.raises(UnknownReference):
        sim.schedule_transport_failure("nope", "station_0.core", "station_1.core")
    with pytest.raises(UnknownReference):
        sim.schedule_transport_failure("transport", "bad.spot", "station_1.core")
