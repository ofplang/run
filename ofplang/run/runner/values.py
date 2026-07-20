"""View-value bookkeeping and routing primitives (dev-notes design.md D26).

The runner is the home of the value layer (D26): it holds each produced port
value and routes it to the consumer input it feeds. This module is the pure core
of that -- a value store keyed by `(node, port)`, plus the separated functions
that seed the boundary, assemble a node's inputs from upstream, and record a
node's outputs. Keeping these as standalone functions (rather than folding them
into the runner loop) makes the value plumbing testable on its own and gives a
clear seam to grow in v0-full (typed generation, contract checks); the runner
loop just calls them.

v0-lite scope (D26): values are opaque, identifiable markers, not typed view
values. Generation of a *produced* value is the backend's job (the simulator, at
completion); this module only generates the *boundary / unconnected* dummies the
runner is responsible for, and routes whatever the backend produced.
"""

from __future__ import annotations

from typing import Any


class ValueStore:
    """The runner's view values, keyed by `(node_path, port)`.

    A producer's output is recorded here when the backend reports it complete; a
    boundary entry input is seeded here at run start. Keys use the scheduler's
    node-path convention (the boundary is the empty path `()`), so they line up
    with the dataflow adapter's sources and the plan's `node` paths. Node paths are
    normalised to tuples on the way in, so callers may pass a list (as the plan's
    activity dicts carry) or a tuple.
    """

    def __init__(self) -> None:
        self._values: dict[tuple, Any] = {}

    def put(self, node, port: str, value: Any) -> None:
        self._values[(tuple(node), port)] = value

    def get(self, node, port: str) -> Any:
        return self._values[(tuple(node), port)]

    def has(self, node, port: str) -> bool:
        return (tuple(node), port) in self._values

    def snapshot(self) -> dict:
        """A copy of the whole store (debug / inspection; the runner exposes the
        final outputs through this rather than the §6/§7 document in v0-lite)."""
        return dict(self._values)


def dummy_value(node, port: str) -> str:
    """An opaque but identifiable placeholder value (v0-lite).

    Deterministic and unique per `(node, port)` so a test can tell one origin from
    another, but carrying no real data (no type, no view schema -- that is v0-full).
    The boundary node `()` renders as `@entry`.
    """
    where = ".".join(node) if node else "@entry"
    return f"dummy:{where}/{port}"


def seed_entry(dataflow, store: ValueStore) -> None:
    """Seed every `main`-level entry input with a boundary dummy at `((), port)`.

    v0-lite supplies dummies for the whole-workflow inputs (D26); v0-full will take
    real values from a job document and honour static `value:` sources instead.
    """
    for port in dataflow.entry_ports:
        store.put((), port, dummy_value((), port))


def assemble_inputs(dataflow, store: ValueStore, node) -> dict:
    """Build a node's input values by following each input port back to its source.

    For each input port of `node`: if the dataflow gives it a source that has a
    value, use that (this is the producer -> consumer routing); otherwise it is
    unconnected (a literal `value:` or unbound input) and gets a dummy. This is the
    routing primitive the D26-1 unit tests exercise; the v0-lite rolling loop does
    not pass inputs to the backend (seam is output-only, D26), so it is wired into
    dispatch only in v0-full.
    """
    node = tuple(node)
    result: dict[str, Any] = {}
    for port in dataflow.in_ports.get(node, ()):
        source = dataflow.input_source.get((node, port))
        if source is not None and store.has(source[0], source[1]):
            result[port] = store.get(source[0], source[1])
        else:
            result[port] = dummy_value(node, port)
    return result


def record_outputs(store: ValueStore, node, outputs: dict) -> None:
    """Record a completed node's produced outputs (`{port: value}`) into the store,
    keyed by `(node, port)`, so downstream consumers and the final returns can read
    them."""
    for port, value in outputs.items():
        store.put(node, port, value)


def collect_outputs(dataflow, store: ValueStore) -> dict:
    """Assemble the whole-workflow outputs from the store, following each `main`
    output port back to its producing `(node, port)`. A return whose producer has
    not been recorded is omitted (it never ran)."""
    result: dict[str, Any] = {}
    for name, (node, port) in dataflow.returns.items():
        if store.has(node, port):
            result[name] = store.get(node, port)
    return result
