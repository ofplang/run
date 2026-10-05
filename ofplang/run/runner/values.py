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
`Source` trees. The one value the runner synthesises is an entry input the
boundary did not supply: a typed default (`contracts.default_value`), which the
run reports (D59). Nothing inside the workflow is ever defaulted -- every input
has a source (`dataflow.from_workflow` refuses one that does not), and an input
whose source has no value yet is not assembled at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .contracts import ArrayType, conforms, default_value, with_static_views
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

    def discard(self, node, port: str) -> None:
        """Remove a recorded value if present (a no-op otherwise). Used to withdraw
        an output that was recorded for a contract check but must not be surfaced --
        e.g. an invocation whose `ensures` then failed."""
        self._values.pop((tuple(node), port), None)

    def snapshot(self) -> dict:
        """A copy of the whole store (debug / inspection; the runner exposes the
        final outputs through this rather than the §6/§7 document in v0-lite)."""
        return dict(self._values)


# -- Source trees -------------------------------------------------------------
#
# A `Source` (schedule D57) says where one port's value comes from: a recorded value
# or one element of it (`SourceRef`), a literal, or an Array assembled element by
# element (`SourceSeq`). Matched by class name rather than imported, so importing
# the runner package does not pull the scheduler in before a run needs it.


def source_refs(source) -> Iterator:
    """Every `SourceRef` a source reads, in element order."""
    kind = type(source).__name__
    if kind == "SourceRef":
        yield source
    elif kind == "SourceSeq":
        for item in source.items:
            yield from source_refs(item)


def is_available(source, store: ValueStore) -> bool:
    """Whether every value `source` reads has been recorded (a literal always has)."""
    return all(store.has(ref.node, ref.port) for ref in source_refs(source))


def fixed_at_start(source) -> bool:
    """Whether `source` is known before anything runs: it reads only the boundary
    (seeded at run start) and literals."""
    return all(tuple(ref.node) == () for ref in source_refs(source))


def resolve(source, store: ValueStore) -> Any:
    """The value `source` describes, read from `store`.

    The caller asks `is_available` first; reading an unrecorded value is a runner
    bug and raises. So does an element index past the end of its Array: the plan
    was built for that many elements, and the length checks (`LengthCheck`) stop a
    job whose values disagree before any consumer of the missing element runs."""
    kind = type(source).__name__
    if kind == "SourceLiteral":
        return source.value
    if kind == "SourceSeq":
        return [resolve(item, store) for item in source.items]
    if kind != "SourceRef":
        raise RunnerError(f"unknown value source {source!r}")
    if not store.has(source.node, source.port):
        raise RunnerError(f"no value recorded for {describe(source)}")
    value = store.get(source.node, source.port)
    for depth, i in enumerate(source.index):
        if not isinstance(value, list) or not 0 <= i < len(value):
            whole = type(source)(source.node, source.port)
            raise RunnerError(
                f"{describe(whole)} has no element at "
                f"{''.join(f'[{j}]' for j in source.index[: depth + 1])}"
            )
        value = value[i]
    return value


def describe(source) -> str:
    """A source as a message names it: `Dispense/2.plate`, `main.plates[1]`, ..."""
    from ofplang.schedule.core.identifiers import format_element, format_node_path

    kind = type(source).__name__
    if kind == "SourceRef":
        node = format_node_path(source.node) if source.node else "main"
        return f"{node}.{format_element(source.port, tuple(source.index))}"
    if kind == "SourceLiteral":
        return f"the literal {source.value!r}"
    return "[" + ", ".join(describe(item) for item in source.items) + "]"


# -- the boundary ---------------------------------------------------------------


def _spot_shaped(spots, resolved, port: str, make):
    """Walk an Array port's spot binding (a spot, or lists of them as deep as the
    port nests Arrays) and build a value of the same shape, `make(element_type)`
    at each spot. The binding is what fixes how many Objects there are, so a
    default for an Array of Objects has one element per spot -- not `[]`, which
    would say there are none."""
    if isinstance(spots, list):
        if not isinstance(resolved, ArrayType):
            raise RunnerError(f"boundary input {port!r} binds a list of spots to a non-Array")
        return [_spot_shaped(item, resolved.element, port, make) for item in spots]
    return make(resolved)


def _matches_spots(value, spots) -> bool:
    """Whether a supplied view value has exactly the shape of its spot binding: a
    list of the same length wherever the binding has a list, at every depth."""
    if isinstance(spots, list):
        return (
            isinstance(value, list)
            and len(value) == len(spots)
            and all(_matches_spots(v, s) for v, s in zip(value, spots, strict=True))
        )
    return True


def seed_entry(
    dataflow, contracts, store: ValueStore, job: dict | None = None, spots: dict | None = None
) -> None:
    """Seed every `main`-level entry input at `((), port)` with a typed view value.

    A value supplied by `job` is used (and contract-checked against the entry
    input's type); an entry input the job omits gets a typed default (F4) -- the
    one place the runner makes a value up, and reported by the run (`Job.warnings`).
    A job key that is not an entry input is an error (a typo / wrong port).

    Object entries are seeded here too (their value is a view record); their
    physical placement on an interface spot is separate (§6.8). `spots` is that
    placement (`{port: spot | [spot, ...]}`): an Array of Objects has one element
    per spot, so its default has that many elements, and a supplied view must have
    exactly that shape."""
    job = job or {}
    spots = spots or {}
    entry_inputs = contracts.processes[contracts.entry].inputs if contracts.entry else {}
    for port in job:
        if port not in entry_inputs:
            raise RunnerError(f"job supplies unknown entry input {port!r}")
    for port in dataflow.entry_ports:
        resolved = entry_inputs.get(port)
        if resolved is None:
            # The flattener and the contract resolver read the same entry process, so
            # this is a runner bug, not a document problem -- and a value made up for
            # a port with no type could only be a guess.
            raise RunnerError(f"entry input {port!r} has no resolved type")
        bound = spots.get(port)
        if port in job:
            value = job[port]
            if not conforms(value, resolved):
                raise RunnerError(
                    f"job value for entry input {port!r} does not conform to its type"
                )
            if bound is not None and not _matches_spots(value, bound):
                raise RunnerError(
                    f"boundary input {port!r}: its view does not have the shape of its "
                    f"spots (one view per spot, in the same order)"
                )
        elif bound is not None:
            value = _spot_shaped(bound, resolved, port, default_value)
        else:
            value = default_value(resolved)
        # Project any type-level static view values onto the seeded value (D35), so a
        # supplied job value with a stale static field is corrected and a default
        # already carries them. The value store then always holds the static value.
        store.put((), port, with_static_views(value, resolved))


# -- routing ---------------------------------------------------------------------


def assemble_inputs(dataflow, contracts, store: ValueStore, node, ports=None) -> dict:
    """Build a node's input values by resolving each input port's `Source`.

    Every port has a source (`dataflow.from_workflow` refuses a workflow where one
    does not), and every source must already be available: a caller dispatching an
    activity asks `unproduced_inputs` first and refuses to proceed. `ports` limits
    the result to those ports -- the run-start preflight (D37) assembles only the
    inputs its phase-hoisted `requires` read, which are the ones fixed at run start.

    Every assembled value is checked against its port's type. A routed value was
    checked when it was recorded, but a literal, or an Array assembled from several
    sources, is first seen whole here."""
    node = tuple(node)
    process = dataflow.process_of.get(node)
    if process is None:
        raise RunnerError(f"no activity at node {node!r}")
    result: dict[str, Any] = {}
    for port in dataflow.in_ports.get(node, ()):
        if ports is not None and port not in ports:
            continue
        source = dataflow.sources[(node, port)]
        value = resolve(source, store)
        resolved = contracts.input_type(process, port)
        if not conforms(value, resolved):
            if type(source).__name__ == "SourceLiteral":
                raise RunnerError(f"static literal for input {port!r} does not conform to its type")
            raise RunnerError(
                f"value for input {port!r} ({describe(source)}) does not conform to its type"
            )
        # A value bound to a static-view type gets its static fields projected (D35).
        result[port] = with_static_views(value, resolved)
    return result


def unproduced_inputs(dataflow, store: ValueStore, node) -> list[str]:
    """The input ports of `node` whose source has not recorded a value yet.

    A non-empty result at dispatch time means the activity was started while a
    predecessor was still running, which the plan's precedence is supposed to
    prevent; see the caller for how that arises and why it is an error. A boundary
    source never appears -- `seed_entry` seeds every entry port before anything is
    dispatched -- nor does a literal."""
    node = tuple(node)
    return [
        port
        for port in dataflow.in_ports.get(node, ())
        if not is_available(dataflow.sources[(node, port)], store)
    ]


def record_outputs(store: ValueStore, node, outputs: dict) -> None:
    """Record a completed node's produced outputs (`{port: value}`) into the store,
    keyed by `(node, port)`, so downstream consumers and the final returns can read
    them."""
    for port, value in outputs.items():
        store.put(node, port, value)


def collect_outputs(dataflow, store: ValueStore) -> dict:
    """Assemble the whole-workflow outputs from the store, resolving each `main`
    output port's `Source`. A return whose source is not yet available is omitted:
    only a job that stopped can end with one, and the caller holds every other job
    to having them all."""
    return {
        name: resolve(source, store)
        for name, source in dataflow.returns.items()
        if is_available(source, store)
    }
