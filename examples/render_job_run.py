"""Run a workflow with supplied inputs and see the outputs (dev-notes D27 F4/F5).

Scenario (see count_chain.workflow.yaml): a `Count` value (view {value: Int})
enters from a supplied *job*, flows through two device-less `inc` steps, and is
returned as the whole-workflow output. This is the full value story the runner now
supports:

  * the caller supplies the whole-workflow inputs (the job), contract-checked (F4a);
  * a device model computes each step's outputs from its inputs (F4b) -- here `inc`
    adds one, so the value is transformed, not merely carried;
  * the runner assembles the whole-workflow outputs and exposes them (F5).

So a job of `{start: {value: 42}}` becomes `{result: {value: 44}}` (42 -> +1 -> +1).
The device model is the dummy stand-in for real device computation; a real backend
would plug a real model at the same seam.

Run it:

    python examples/render_job_run.py

It prints the job, each step's value, and the whole-workflow outputs, and writes
examples/outputs/job_run.trace.txt and examples/outputs/job_run.outputs.yaml (the
latter is what `ofp-run run --outputs FILE` would write). Requires the sibling
`ofplang-schedule` (the runner replans through it).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ofplang.run.runner import RollingRunner

HERE = Path(__file__).parent
OUT = HERE / "outputs"
WORKFLOW = HERE / "count_chain.workflow.yaml"
ENVIRONMENT = HERE / "count_chain.env.yaml"

# The whole-workflow inputs supplied by the caller (a job).
JOB = {"start": {"value": 42}}


def inc_model(process, mode, inputs, output_schema):
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
        str(WORKFLOW), str(ENVIRONMENT), job=JOB, device_model=inc_model,
        poll_interval=None, random_seed=0,
    )
    status = runner.run()

    lines: list[str] = []
    lines.append("job run (supplied inputs -> computed outputs)")
    lines.append("=" * 46)
    lines.append(f"job (whole-workflow inputs) : {JOB}")
    lines.append("")
    lines.append("per-step values (device model `inc`: value + 1):")
    # The two inc steps, in order; show the Count each produced.
    for node in (("S1",), ("S2",)):
        lines.append(f"  {'/'.join(node)}.y = {runner.values.get(node, 'y')}")
    lines.append("")
    lines.append(f"whole-workflow outputs      : {runner.outputs}")

    text = "\n".join(lines) + "\n"
    (OUT / "job_run.trace.txt").write_text(text, encoding="utf-8")
    # The outputs artifact, as `ofp-run run --outputs FILE` would write it.
    (OUT / "job_run.outputs.yaml").write_text(
        yaml.safe_dump(runner.outputs, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(text, end="")
    print(f"makespan = {status['now']}")
    print(f"wrote {OUT / 'job_run.trace.txt'}")
    print(f"wrote {OUT / 'job_run.outputs.yaml'}")


if __name__ == "__main__":
    main()
