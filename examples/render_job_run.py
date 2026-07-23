"""Run a workflow with supplied inputs and see the outputs (dev-notes D28 / D27 F4).

Scenario (see count_chain.workflow.yaml): a `Count` value (view {value: Int})
enters from the run boundary, flows through two device-less `inc` steps, and is
returned as the whole-workflow output. This is the full value story the runner now
supports:

  * the caller supplies the whole-workflow inputs in the boundary, contract-checked;
  * a device model computes each step's outputs from its inputs (F4b) -- here `inc`
    adds one, so the value is transformed, not merely carried;
  * the runner assembles the whole-workflow outputs and echoes them back into the
    result boundary (the same schema as the supplied boundary).

So a boundary input of `{start: {value: 42}}` becomes `{result: {value: 44}}`
(42 -> +1 -> +1). The device model is the dummy stand-in for real device
computation; a real backend would plug a real model at the same seam.

Run it:

    python examples/render_job_run.py

It prints the boundary inputs, each step's value, and the whole-workflow outputs,
and writes examples/outputs/job_run.trace.txt and job_run.boundary.yaml (the latter
is the result boundary, as `ofp-run run --boundary-out FILE` would write it).
Requires the sibling `ofplang-schedule` (the runner replans through it).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ofplang.run.runner import RollingRunner

HERE = Path(__file__).parent
OUT = HERE / "outputs"
WORKFLOW = HERE / "count_chain.workflow.yaml"
ENVIRONMENT = HERE / "count_chain.env.yaml"

# The run boundary (D28): the whole-workflow inputs supplied by the caller. `start`
# is Pure Data (a Count view), so it carries a `view` and no spot.
BOUNDARY = {"boundary": {"inputs": {"start": {"view": {"value": 42}}}}}


def inc_model(process, mode, inputs, output_schema, definition):
    """A device model for `inc`: each output Count is the input Count plus one.

    It computes outputs from inputs (F4b), the dummy stand-in for what a real device
    would do. `inputs["x"]` is the routed input value; every output port (here `y`)
    gets the incremented Count."""
    n = inputs["x"]["value"]
    return {port: {"value": n + 1} for port in output_schema}


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Event-boundary advance keeps the times exact for a clean trace; the value
    # layer is identical under either poll mode.
    runner = RollingRunner(
        str(WORKFLOW), str(ENVIRONMENT), BOUNDARY, device_model=inc_model,
        poll_interval=None, random_seed=0,
    )
    status = runner.run()

    lines: list[str] = []
    lines.append("job run (supplied inputs -> computed outputs)")
    lines.append("=" * 46)
    lines.append(f"boundary inputs             : {BOUNDARY['boundary']['inputs']}")
    lines.append("")
    lines.append("per-step values (device model `inc`: value + 1):")
    # The two inc steps, in order; show the Count each produced.
    for node in (("S1",), ("S2",)):
        lines.append(f"  {'/'.join(node)}.y = {runner.values.get(node, 'y')}")
    lines.append("")
    lines.append(f"whole-workflow outputs      : {runner.outputs}")

    text = "\n".join(lines) + "\n"
    (OUT / "job_run.trace.txt").write_text(text, encoding="utf-8")
    # The result boundary artifact, as `ofp-run run --boundary-out FILE` would write
    # it: the same schema as the supplied boundary, with each output's view filled in.
    (OUT / "job_run.boundary.yaml").write_text(
        yaml.safe_dump(runner.result_boundary, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(text, end="")
    print(f"makespan = {status['now']}")
    print(f"wrote {OUT / 'job_run.trace.txt'}")
    print(f"wrote {OUT / 'job_run.boundary.yaml'}")


if __name__ == "__main__":
    main()
