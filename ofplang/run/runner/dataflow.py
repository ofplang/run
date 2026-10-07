"""Workflow dataflow view for value routing (dev-notes design.md D26).

The runner owns the *value* layer: it routes each producer output port's view
value to the consumer input port it feeds (D26). To do that it needs the
workflow's port-level dataflow graph -- which output feeds which input, for both
Object-bearing (`state`) and Pure Data (`bind`) arcs, across nested composite
boundaries and the invocations of an expanded `map` / `fold`.

Rather than re-parse and re-flatten the workflow here (which would risk diverging
from the scheduler's node-path convention, and silently mis-key the value store),
this module is a *thin adapter* over the scheduler's own flattener,
`ofplang.schedule.scheduler.workflow.parse_workflow` (D26-0). That flattener is
the single source of the node paths that also appear in the plan the runner
drives, so the two always agree.

Where a value comes from is read as the flattener's `Source` trees (schedule D57):
`SourceRef(node, port, index)` -- the value recorded at `(node, port)`, or one
element of it -- `SourceLiteral(value)` and `SourceSeq(items)`, an Array assembled
element by element. They say everything the older per-kind fields (`arcs`,
`data_arcs`, `data_literals`, `exit_outputs`, ...) say, and also what an expanded
`map` / `fold` needs and those fields cannot: an Array gathered from several
invocations, one element of another's Array.

This adapter reads only the graph *structure* (node paths, ports, sources,
boundary). It does not resolve types or view schemas -- that is `contracts.py`'s
job (§7) -- so no §7 / §5.7 machinery is pulled in here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .runner import RunnerError

# A node path is the scheduler's identity for an atomic activity: the node ids
# from the entry composite's body down to the atomic, as a tuple -- with an
# iteration index (an int) after each expanded `map` / `fold` node. The empty tuple
# `()` denotes the workflow boundary (a `main`-level entry input / final output),
# matching the plan's `node: []` boundary convention.
NodePath = tuple  # tuple[str | int, ...]


@dataclass(frozen=True)
class CompositeBoundary:
    """One nested composite invocation's value-layer boundary (D34), for its contract
    checks. `inputs` / `outputs` map each of the composite's own ports to the
    `Source` its value comes from. `process` is the composite process name (its
    contracts are keyed by it)."""

    process: str
    inputs: dict   # port -> Source
    outputs: dict  # port -> Source


@dataclass(frozen=True)
class Dataflow:
    """The routing view of a workflow, derived from the scheduler's flattened graph.

    All node paths use the scheduler's convention (and so match the plan). A
    `SourceRef` whose node is `()` is a boundary entry input (seeded, not produced).
    """

    # node path -> the process it invokes (debug / provenance).
    process_of: dict
    # node path -> its input / output port names (every port, Object and Pure Data).
    in_ports: dict
    out_ports: dict
    # (consumer node, input port) -> the `Source` its value comes from. Every input
    # of every activity has one: `from_workflow` refuses a workflow where one does not
    # (v0 §11 binds every input port, and defines no default for any).
    sources: dict
    # every `main`-level input port name (seeded at the boundary at run start).
    entry_ports: tuple
    # `main`-level output port name -> the `Source` of the final output.
    returns: dict
    # Nested composite invocation boundaries (D34), keyed by the composite's node
    # path -> `CompositeBoundary`, so the runner can evaluate the composite's
    # contracts against those values even though the composite is flattened away.
    # The top-level entry composite `()` is not here (its contracts are checked via
    # the whole-workflow handles, D33).
    composites: dict
    # Lengths the plan was built on that only a value can confirm (schedule
    # `LengthCheck`): an `each` source zipped with one whose length was known.
    length_checks: tuple = ()


def from_workflow(
    workflow, interface: dict | None = None, expansion: dict | None = None
) -> Dataflow:
    """Build the routing view by reusing the scheduler's flattener (D26-0).

    `workflow` is either a path to a workflow YAML file or an already-loaded document
    (a mapping). `interface` is the run's §6.8 boundary (spots only): an Array of
    Objects at the boundary takes its length from the spots it is bound to, so a
    workflow traversing one expands to as many invocations as the scheduler plans --
    the same call, with the same binding, gives the same graph. `expansion` (§6.13)
    does the same for a Pure Data Array at the boundary: the lengths the run counted
    off its values (`job.expansion_of`).

    Raises `RunnerError` if the workflow cannot be flattened (e.g. it contains a
    structured node the scheduler does not expand, or has no entry) -- the same
    diagnostics the scheduler would raise -- or if an input of an activity has no
    source at all.
    """
    # Import lazily: like `schedule_client`, so importing the runner package does
    # not hard-require the scheduler to be installed until `run` actually uses it.
    from ofplang.schedule.core.diagnostics import ERROR
    from ofplang.schedule.core.identifiers import format_node_path
    from ofplang.schedule.scheduler.workflow import parse_workflow

    workflow, diags = parse_workflow(
        workflow if isinstance(workflow, dict) else str(workflow),
        interface=interface,
        expansion=expansion,
    )
    errors = [d for d in diags.items if d.severity == ERROR]
    if workflow is None or errors:
        codes = ", ".join(sorted({str(getattr(d, "code", d)) for d in errors}))
        raise RunnerError(f"cannot read workflow dataflow ({codes or 'no workflow'})")

    # Per-node process and port names. `workflow.processes` holds the signatures of
    # exactly the atomic processes the activities invoke.
    process_of = {a.path: a.process for a in workflow.activities}
    in_ports = {
        a.path: tuple(p.name for p in workflow.processes[a.process].inputs)
        for a in workflow.activities
    }
    out_ports = {
        a.path: tuple(p.name for p in workflow.processes[a.process].outputs)
        for a in workflow.activities
    }

    # Every input of every activity, keyed by (node, port) -- the value store's key
    # convention. An input the flattener found no source for is refused here, before
    # anything runs: v0 binds every input port (§11) and defines no default for one,
    # so there is no value the runner could give it that would be the workflow's.
    # The flattener refuses such a document itself (its guards, schedule D60), so
    # this holds it to that rather than catching anything new.
    sources = {(ep.node, ep.port): source for ep, source in workflow.input_sources.items()}
    unsourced = [
        f"{format_node_path(node)}.{port}"
        for node, ports in in_ports.items()
        for port in ports
        if (node, port) not in sources
    ]
    if unsourced:
        raise RunnerError(
            f"input(s) {', '.join(unsourced)} have no source: every input port must be "
            f"bound (v0 §11), and the runner gives none a default"
        )

    composites = {
        path: CompositeBoundary(
            process=io.process,
            inputs=dict(io.input_sources),
            outputs=dict(io.output_sources),
        )
        for path, io in workflow.composites.items()
    }

    # Final outputs likewise, Object-bearing and Pure Data alike. v0 has a composite
    # return every output it declares (spec 12.3, revision 0.5), and the scheduler
    # records a source for every one -- an Object returned untouched included (its
    # D60) -- so one without is the flattener's to explain, and the run would
    # otherwise end without a value it was asked for.
    returns = dict(workflow.output_sources)
    unreturned = [name for name in workflow.exit_output_ports if name not in returns]
    if unreturned:
        raise RunnerError(
            f"final output(s) {', '.join(unreturned)} have no source: a composite returns "
            f"every output it declares (v0 12.3)"
        )

    return Dataflow(
        process_of=process_of,
        in_ports=in_ports,
        out_ports=out_ports,
        sources=sources,
        entry_ports=tuple(workflow.entry_input_ports.keys()),
        returns=returns,
        composites=composites,
        length_checks=tuple(workflow.length_checks),
    )
