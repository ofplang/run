"""Run a chain carrying an Int and an Object together (dev-notes D27 F4/F5).

Like render_job_run.py, but each `step` carries two values at once (see
plate_chain.workflow.yaml): a Pure Data `Int` counter and an Object-bearing
`Plate` -- the same plate in and out (an in-place transform). The device model
increments the Int and passes the plate through unchanged, so:

  * the Int is *transformed* down the chain (42 -> 43 -> 44), while
  * the Plate's view value (its barcode) is *carried* through unchanged, its
    identity tracked physically by the simulator (it is loaded, processed on the
    worker, and delivered to the unloader).

So a job of `{start: 42, sample: {barcode: "ABC"}}` yields
`{result: 44, plate_final: {barcode: "ABC"}}`. This shows Pure Data and
Object-bearing values flowing together through the same value seam.

Run it:

    python examples/render_plate_chain.py

It prints the job, each step's Int and Plate, and the whole-workflow outputs, and
writes examples/outputs/plate_chain.trace.txt and plate_chain.outputs.yaml.
Requires the sibling `ofplang-schedule` (the runner replans through it).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ofplang.run.runner import RollingRunner, load_document
from ofplang.run.simulator import default_device_model

HERE = Path(__file__).parent
OUT = HERE / "outputs"
WORKFLOW = HERE / "plate_chain.workflow.yaml"
ENVIRONMENT = HERE / "plate_chain.env.yaml"
DOCUMENT = HERE / "plate_chain.document.yaml"

# The whole-workflow inputs supplied by the caller: an Int counter and a Plate.
JOB = {"start": 42, "sample": {"barcode": "ABC"}}


def step_model(process, mode, inputs, output_schema, definition):
    """A device model built on the simulator's built-in default.

    `default_device_model` already does the generic work: type defaults for every
    output, plus carrying each Object output declared in `objects.map` through from
    its input (so the plate pass-through needs no code here). This model only adds
    the one genuinely computed output on top: `next = current + 1`."""
    outputs = default_device_model(process, mode, inputs, output_schema, definition)
    if process == "step":
        outputs["next"] = inputs["current"] + 1
    return outputs


def main() -> None:
    OUT.mkdir(exist_ok=True)

    interface = load_document(DOCUMENT)["interface"]
    runner = RollingRunner(
        str(WORKFLOW), str(ENVIRONMENT), interface, job=JOB, device_model=step_model,
        poll_interval=None, random_seed=0,
    )
    status = runner.run()

    lines: list[str] = []
    lines.append("plate chain (Int carried alongside an Object)")
    lines.append("=" * 46)
    lines.append(f"job (whole-workflow inputs) : {JOB}")
    lines.append("")
    lines.append("per-step values (device model: next = current + 1, plate carried through):")
    for node in (("S1",), ("S2",)):
        name = "/".join(node)
        lines.append(f"  {name}.next  = {runner.values.get(node, 'next')}")
        lines.append(f"  {name}.plate = {runner.values.get(node, 'plate')}")
    lines.append("")
    lines.append(f"whole-workflow outputs      : {runner.outputs}")

    text = "\n".join(lines) + "\n"
    (OUT / "plate_chain.trace.txt").write_text(text, encoding="utf-8")
    (OUT / "plate_chain.outputs.yaml").write_text(
        yaml.safe_dump(runner.outputs, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(text, end="")
    print(f"makespan = {status['now']}")
    print(f"wrote {OUT / 'plate_chain.trace.txt'}")
    print(f"wrote {OUT / 'plate_chain.outputs.yaml'}")


if __name__ == "__main__":
    main()
