"""Trace a run that combines a static value literal (§11 / D30), Python script
processes (§22 / D31), and runtime contracts (§9 / D32) -- the value layer
computing and checking real values.

Scenario (see script_literal.workflow.yaml): a measured `raw` from the run
boundary and a `threshold` embedded as a static literal (`value: 60`) both feed a
`score` script process, which computes `margin` and `passed`; a second `report`
script turns those into a summary string. `score` declares a `requires`
precondition and `ensures` postconditions (and `report` an `ensures`), which the
runner checks at runtime against the computed view values. Unlike
render_data_flow.py (where the backend produced typed *defaults*), here the
outputs are genuinely computed by the scripts, and one input is a workflow-embedded
constant rather than a routed value.

This prints a text trace making all three features visible: which inputs were
seeded at the boundary, which came from a static literal, the assembled inputs and
computed outputs per activity, the declared contracts (checked at runtime), and
the whole-workflow outputs.

Run it:

    python examples/render_script_literal.py

It writes examples/outputs/script_literal.trace.txt and prints a summary. Requires
the sibling `ofplang-schedule` (pip install -e ../ofplang-schedule): the runner
replans through it and reuses its workflow flattener for the routing view (which
also carries the static literals, D30).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ofplang.run.runner import RollingRunner
from ofplang.run.runner.values import assemble_inputs

HERE = Path(__file__).parent
OUT = HERE / "outputs"
WORKFLOW = HERE / "script_literal.workflow.yaml"
ENVIRONMENT = HERE / "script_literal.env.yaml"

# The run boundary (D28): `raw` is Pure Data, so it occupies no spot -- only its
# view value is supplied. A measured 72 against the embedded threshold 60 gives a
# margin of 12 and a PASS.
BOUNDARY = {"boundary": {"inputs": {"raw": {"view": 72}}}}


def _fmt_node(node) -> str:
    """A workflow node path (tuple) as a readable dotted string; `()` is the
    workflow boundary."""
    return "/".join(node) if node else "(boundary)"


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Drive the workflow to completion. Event-boundary advance keeps the times exact
    # for a clean trace; the value layer is identical under either poll mode.
    runner = RollingRunner(str(WORKFLOW), str(ENVIRONMENT), BOUNDARY, poll_interval=None, random_seed=0)
    status = runner.run()

    df = runner.dataflow
    lines: list[str] = []
    lines.append("static value + python script -- computed value flow")
    lines.append("=" * 58)

    # Boundary seeds: the whole-workflow entry inputs the runner supplied.
    lines.append("entry inputs (seeded at the boundary):")
    for port in df.entry_ports:
        lines.append(f"  {port:<12} = {runner.values.get((), port)!r}")

    # Static literals (§11 / D30): constants embedded in the workflow, keyed by the
    # consuming (node, port). These are supplied to the script in place of a routed
    # value or a typed default.
    lines.append("")
    lines.append("static value literals (embedded in the workflow):")
    if df.literals:
        for (node, port), value in df.literals.items():
            lines.append(f"  {_fmt_node(node)}.{port:<10} = {value!r}")
    else:
        lines.append("  (none)")

    # Per activity, in commit order: the inputs it drew (routed from upstream, seeded
    # at the boundary, or a static literal) and the outputs its script computed.
    lines.append("")
    lines.append("activities (assembled inputs -> script-computed outputs):")
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
        lines.append(f"  {name:<12} = {runner.outputs.get(name)!r}   <- {_fmt_node(node)}.{port}")

    # The contracts declared on each process (§9), checked by the runner at runtime
    # against the computed view values. The run reaching here without failing means
    # every requires / ensures held; a violation would have stopped it (D25).
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    verdict = "all held" if not runner.failed else "VIOLATED (run stopped)"
    lines.append("")
    lines.append(f"contracts (checked at runtime -- {verdict}):")
    any_contract = False
    for pname, pdef in (workflow.get("processes") or {}).items():
        for section in ("requires", "ensures"):
            for item in ((pdef or {}).get("contracts") or {}).get(section) or []:
                lines.append(f"  {pname}.{section:<8} {item['expr']}")
                any_contract = True
    if not any_contract:
        lines.append("  (none)")

    text = "\n".join(lines) + "\n"
    (OUT / "script_literal.trace.txt").write_text(text, encoding="utf-8")

    print(text, end="")
    print(f"makespan = {status['now']}")
    print(f"wrote {OUT / 'script_literal.trace.txt'}")


if __name__ == "__main__":
    main()
