"""Trace the value layer: how view values flow producer -> consumer to the output.

Scenario (see data_flow.workflow.yaml): an Object plate is measured (emitting a
Pure Data reading), a nested `Analyzer` composite turns the reading into a score,
and the score is both a gate for the final Object step and the workflow's returned
output. The runner owns the value layer (dev-notes design.md D26): the backend
generates a value at each output port, the runner routes it along the workflow's
arcs (across the composite boundary), and assembles the whole-workflow output from
the `returns`.

This prints a text trace of that flow -- each activity's assembled inputs and the
outputs the backend produced, then the final whole-workflow outputs -- so the
otherwise-internal value layer is visible. The backend generates typed default
values shaped by each type's view schema (§7.4): a `Reading` becomes
`{mean: 0.0, n: 0}`, a `Score` `{value: 0.0, ok: false}`. Entry inputs are seeded
with typed defaults here (a run may instead supply a job of entry values);
producer outputs do not yet depend on inputs -- that arrives with a device model.

Run it:

    python examples/render_data_flow.py

It writes examples/outputs/data_flow.trace.txt and prints a summary. Requires the
sibling `ofplang-schedule` (pip install -e ../ofplang-schedule): the runner
replans through it and reuses its workflow flattener for the routing view.
"""

from __future__ import annotations

from pathlib import Path

from ofplang.run.runner import RollingRunner, load_document
from ofplang.run.runner.values import assemble_inputs

HERE = Path(__file__).parent
OUT = HERE / "outputs"
WORKFLOW = HERE / "data_flow.workflow.yaml"
ENVIRONMENT = HERE / "data_flow.env.yaml"
DOCUMENT = HERE / "data_flow.document.yaml"


def _fmt_node(node) -> str:
    """A workflow node path (tuple) as a readable dotted string; `()` is the
    workflow boundary."""
    return "/".join(node) if node else "(boundary)"


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Drive the workflow to completion. Event-boundary advance keeps the times
    # exact for a clean trace; the value layer is identical under either poll mode.
    interface = load_document(DOCUMENT)["interface"]
    runner = RollingRunner(str(WORKFLOW), str(ENVIRONMENT), interface, poll_interval=None, random_seed=0)
    status = runner.run()

    df = runner.dataflow
    lines: list[str] = []
    lines.append("value flow (producer -> consumer, typed view values)")
    lines.append("=" * 58)

    # Boundary seeds: the whole-workflow entry inputs the runner supplied.
    lines.append("entry inputs (seeded at the boundary):")
    for port in df.entry_ports:
        lines.append(f"  {port:<12} = {runner.values.get((), port)}")

    # Per activity, in the order they were committed: the inputs it draws from
    # upstream (routed by the dataflow view) and the outputs the backend produced.
    lines.append("")
    lines.append("activities (assembled inputs -> produced outputs):")
    for record in runner.log.records():
        node = record.activity.get("node")
        if node is None:  # a transport / bookkeeping leg carries no value
            continue
        node = tuple(node)
        inputs = assemble_inputs(df, runner.contracts, runner.values, node)
        outputs = {port: runner.values.snapshot().get((node, port)) for port in df.out_ports.get(node, ())}
        lines.append(f"  {_fmt_node(node)} [{record.activity.get('process')}]")
        lines.append(f"      in : {inputs or '(none)'}")
        lines.append(f"      out: {outputs or '(none)'}")

    # The whole-workflow outputs, each traced back to the producer it came from.
    lines.append("")
    lines.append("whole-workflow outputs (returns):")
    for name, (node, port) in df.returns.items():
        lines.append(f"  {name:<12} = {runner.outputs.get(name)}   <- {_fmt_node(node)}.{port}")

    text = "\n".join(lines) + "\n"
    (OUT / "data_flow.trace.txt").write_text(text, encoding="utf-8")

    print(text, end="")
    print(f"makespan = {status['now']}")
    print(f"wrote {OUT / 'data_flow.trace.txt'}")


if __name__ == "__main__":
    main()
