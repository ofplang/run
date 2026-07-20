"""View-value bookkeeping and routing primitives (dev-notes design.md D26/D27).

The runner is the home of the value layer (D26): it holds each produced port
value and routes it to the consumer input it feeds. This module is the pure core
of that -- a value store keyed by `(node, port)`, plus the separated functions
that seed the boundary, assemble a node's inputs from upstream, and record a
node's outputs. Keeping these as standalone functions (rather than folding them
into the runner loop) makes the value plumbing testable on its own; the runner
loop just calls them.

Values are typed view values (D27): a supplied job's entry values (contract-
checked, F4) or the backend's generated outputs (F2), routed along the workflow's
arcs. Where the runner must synthesise a value it is responsible for -- an entry
input the job did not supply, or an unconnected input -- it uses a typed default
(`contracts.default_value`), so every value conforms.
"""

from __future__ import annotations

from typing import Any

from .contracts import conforms, default_value
from .runner import RunnerError


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


def seed_entry(dataflow, contracts, store: ValueStore, job: dict | None = None) -> None:
    """Seed every `main`-level entry input at `((), port)` with a typed view value.

    A value supplied by `job` is used (and contract-checked against the entry
    input's type); an entry input the job omits gets a typed default (F4). A job key
    that is not an entry input is an error (a typo / wrong port). Object entries are
    seeded here too (their value is a view record); their physical placement on an
    interface spot is separate (§6.8)."""
    job = job or {}
    entry_inputs = contracts.processes[contracts.entry].inputs if contracts.entry else {}
    for port in job:
        if port not in entry_inputs:
            raise RunnerError(f"job supplies unknown entry input {port!r}")
    for port in dataflow.entry_ports:
        resolved = entry_inputs.get(port)
        if port in job:
            value = job[port]
            if resolved is not None and not conforms(value, resolved):
                raise RunnerError(f"job value for entry input {port!r} does not conform to its type")
        else:
            value = default_value(resolved) if resolved is not None else {}
        store.put((), port, value)


def assemble_inputs(dataflow, contracts, store: ValueStore, node) -> dict:
    """Build a node's input values by following each input port back to its source.

    For each input port of `node`: if the dataflow gives it a source that has a
    value, use that (the producer -> consumer routing); otherwise it is unconnected
    (a literal `value:` or unbound input) and gets a typed default of the port's
    type, so the assembled value always conforms. This is the routing primitive the
    dataflow unit tests exercise and the rolling loop passes to the backend (F4)."""
    node = tuple(node)
    process = dataflow.process_of.get(node)
    result: dict[str, Any] = {}
    for port in dataflow.in_ports.get(node, ()):
        source = dataflow.input_source.get((node, port))
        if source is not None and store.has(source[0], source[1]):
            result[port] = store.get(source[0], source[1])
        else:
            resolved = contracts.input_type(process, port) if process is not None else None
            result[port] = default_value(resolved) if resolved is not None else {}
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
