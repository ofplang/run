"""Workflow dataflow view for value routing (dev-notes design.md D26).

The runner owns the *value* layer: it routes each producer output port's view
value to the consumer input port it feeds (D26). To do that it needs the
workflow's port-level dataflow graph -- which output feeds which input, for both
Object-bearing (`state`) and Pure Data (`bind`) arcs, across nested composite
boundaries.

Rather than re-parse and re-flatten the workflow here (which would risk diverging
from the scheduler's node-path convention, and silently mis-key the value store),
this module is a *thin adapter* over the scheduler's own flattener,
`ofplang.schedule.scheduler.workflow.parse_workflow` (D26-0). That flattener is
the single source of the node paths that also appear in the plan the runner
drives, so the two always agree. The scheduler discards the port-level mapping of
Pure Data arcs (it keeps only node-level precedence) and the static literal
`value:` bindings entirely; D26-0 added `data_arcs` / `data_entry_inputs`, and D30
added `data_literals`, to expose these for us here (value-independent metadata the
scheduler itself never reads).

This adapter reads only the graph *structure* (node paths, ports, arcs,
boundary). It does not resolve types or view schemas -- that is `contracts.py`'s
job (§7) -- so no §7 / §5.7 machinery is pulled in here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .runner import RunnerError

# A node path is the scheduler's identity for an atomic activity: the node ids
# from the entry composite's body down to the atomic, as a tuple. The empty tuple
# `()` denotes the workflow boundary (a `main`-level entry input / final output),
# matching the plan's `node: []` boundary convention.
NodePath = tuple  # tuple[str, ...]


@dataclass(frozen=True)
class Dataflow:
    """The routing view of a workflow, derived from the scheduler's flattened graph.

    All node paths use the scheduler's convention (and so match the plan). A
    "source" is a `(node, port)` pair identifying the producing output port; a
    source whose node is `()` is a boundary entry input (seeded, not produced).
    """

    # node path -> the process it invokes (debug / provenance).
    process_of: dict
    # node path -> its input / output port names (every port, Object and Pure Data).
    in_ports: dict
    out_ports: dict
    # (consumer node, input port) -> the source (node, port) that feeds it. An
    # input with no entry here is unconnected (an unbound input) and is dummy-filled,
    # unless it appears in `literals` below. A source node of `()` is a boundary
    # entry input.
    input_source: dict
    # every `main`-level input port name (seeded at the boundary at run start).
    entry_ports: tuple
    # `main`-level output port name -> the producing (node, port) (for the final
    # whole-workflow outputs). Covers Object and Pure Data returns alike.
    returns: dict
    # (consumer node, input port) -> a static literal `value:` bound to it (§11,
    # Pure Data). The runner seeds these as the port's value in place of a typed
    # default. Recorded by the scheduler's flattener (`data_literals`, D30).
    literals: dict


def from_workflow(workflow_path) -> Dataflow:
    """Build the routing view by reusing the scheduler's flattener (D26-0).

    Raises `RunnerError` if the workflow cannot be flattened (e.g. it contains a
    structured node, which is out of v0 scope, or has no entry) -- the same
    diagnostics the scheduler would raise.
    """
    # Import lazily: like `schedule_client`, so importing the runner package does
    # not hard-require the scheduler to be installed until `run` actually uses it.
    from ofplang.schedule.core.diagnostics import ERROR
    from ofplang.schedule.scheduler.workflow import parse_workflow

    workflow, diags = parse_workflow(str(workflow_path))
    errors = [d for d in diags.items if d.severity == ERROR]
    if workflow is None or errors:
        codes = ", ".join(sorted({str(getattr(d, "code", d)) for d in errors}))
        raise RunnerError(f"cannot read workflow dataflow ({codes or 'no workflow'})")

    # Per-node process and port names. `workflow.processes` holds the signatures of
    # exactly the atomic processes the activities invoke.
    process_of = {a.path: a.process for a in workflow.activities}
    in_ports = {a.path: tuple(p.name for p in workflow.processes[a.process].inputs) for a in workflow.activities}
    out_ports = {a.path: tuple(p.name for p in workflow.processes[a.process].outputs) for a in workflow.activities}

    # Invert every arc to a per-consumer-input source. Object (`arcs`) and Pure
    # Data (`data_arcs`) are routed identically at the value layer -- the physical
    # difference (a transport vs a precedence edge) does not matter for the value.
    input_source: dict = {}
    for arc in workflow.arcs + workflow.data_arcs:
        input_source[(arc.dst.node, arc.dst.port)] = (arc.src.node, arc.src.port)
    # Boundary entry inputs (Object via `entry_inputs`, Pure Data via
    # `data_entry_inputs`): the consuming input is fed by the boundary `()` node.
    for main_port, endpoint in {**workflow.entry_inputs, **workflow.data_entry_inputs}.items():
        input_source[(endpoint.node, endpoint.port)] = ((), main_port)

    # Every main input port to seed at run start, and every main output port with
    # the atomic that produces it (`exit_outputs` records both Object and Pure Data
    # returns; see D26-0).
    entry_ports = tuple(workflow.entry_input_ports.keys())
    returns = {name: (endpoint.node, endpoint.port) for name, endpoint in workflow.exit_outputs.items()}

    # Static literal bindings (§11), keyed by the consuming (node, port) -- the same
    # key convention as `input_source`, so the value layer can look them up the same
    # way. Recorded by the flattener (D30) so nested-composite literals are already
    # spliced to the leaf atomic that consumes them.
    literals = {(endpoint.node, endpoint.port): value for endpoint, value in workflow.data_literals.items()}

    return Dataflow(process_of, in_ports, out_ports, input_source, entry_ports, returns, literals)
