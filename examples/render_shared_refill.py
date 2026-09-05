"""Run one job, then two, and show the refill that only the pair needs.

Both runs use the same workflow file (shared_refill.workflow.yaml: a plate made,
assayed, discarded) against the same laboratory (shared_refill.env.yaml: one reader
holding at most 6 units of reagent, starting with 2). Each assay draws 2 units.

  one job    2 needed, 2 in stock  -> no replenishment at all
  two jobs   4 needed, 2 in stock  -> exactly one refill, covering both

Neither workflow mentions a resource and neither asks to be refilled. The refill
appears because the stock belongs to the *device* (SPEC §4.7), so running the two
jobs against one laboratory is what puts them on one stock -- and one visit tops it
up for whichever assays follow, from either job. That refill carries no `job`: the
scheduler decided to run it, and it serves both.

The two-job run is exactly what the CLI does with a run document, injecting nothing:

    ofp-run run --jobs examples/shared_refill.run.yaml \
        --env examples/shared_refill.env.yaml

This script exists for the *contrast* -- one run cannot show what the other job
changed. Run it:

    python examples/render_shared_refill.py

It prints both schedules and writes examples/outputs/shared_refill.txt. Requires
the sibling `ofplang-schedule` (the runner replans through it).
"""

from __future__ import annotations

from pathlib import Path

from ofplang.run.runner import JobRequest, RollingRunner, load_document

HERE = Path(__file__).parent
OUT = HERE / "outputs"
WORKFLOW = HERE / "shared_refill.workflow.yaml"
ENVIRONMENT = HERE / "shared_refill.env.yaml"

# What the reader holds at the start of the run (§6.10) -- the laboratory's, not any
# one job's, which is why it is stated for the run rather than in a boundary.
INVENTORIES = {"levels": {"reader": {"reagent": 2}}}


def _run(job_ids: list[str]) -> dict:
    workflow = load_document(WORKFLOW)
    runner = RollingRunner(
        [JobRequest(id=job_id, workflow=workflow) for job_id in job_ids],
        str(ENVIRONMENT),
        poll_interval=None,
        random_seed=0,
        inventories=INVENTORIES,
    )
    return runner.run()


def _render(title: str, status: dict) -> list[str]:
    lines = [title, "-" * len(title)]
    for activity in sorted(status["activities"], key=lambda a: (a["start"], a["end"])):
        # A replenishment belongs to no job (it serves whoever assays after it), so
        # the job column is blank for it -- the visible half of the same fact.
        job = activity.get("job") or ""
        what = activity.get("process") or activity.get("device") or activity["kind"]
        lines.append(
            f"  {activity['start']:>3} -> {activity['end']:>3}  "
            f"{job:<10} {activity['kind']:<14} {what}"
        )
    refills = sum(1 for a in status["activities"] if a["kind"] == "replenishment")
    lines.append(f"  makespan {status['now']}, {refills} replenishment(s)")
    return lines


def main() -> None:
    OUT.mkdir(exist_ok=True)

    lines = ["one plate assayed, and the refill that only two of them need"]
    lines.append("=" * 60)
    lines.append("")
    lines += _render("one job: 2 units needed, 2 in stock", _run(["morning"]))
    lines.append("")
    lines += _render(
        "two jobs: 4 units needed, 2 in stock", _run(["morning", "afternoon"])
    )

    text = "\n".join(lines) + "\n"
    (OUT / "shared_refill.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"wrote {OUT / 'shared_refill.txt'}")


if __name__ == "__main__":
    main()
