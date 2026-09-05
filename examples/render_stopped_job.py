"""One job's plate cracks in the oven. What happens to the other two?

Three jobs run the same workflow (shared_refill.workflow.yaml: a plate made, assayed,
discarded) in a two-tray oven (stopped_job.env.yaml). `assay` declares
`device_access: false`, so the plate rests on its tray for twenty seconds without
holding the oven -- which makes the *trays* the scarce thing, and makes what a stopped
job leaves behind cost the others something measurable.

The scenario: every assay on `tray_1` fails. That is injected from Python, like the
device fault in render_reroute.py -- it is a property of the laboratory that day, not
of the workflow, and there is no CLI flag for it. It is also self-limiting: the first
job to reach tray_1 fails there and its plate stays, so tray_1 is declared occupied
and no later job is ever sent to it.

Two runs, differing only in `--on-job-failure`:

  continue (default)  job1 stops; job2 and job3 finish, sharing tray_2 one after the
                      other, because tray_1 is held by a plate nobody can move.
  stop                the first failure ends the run: job2 and job3 are abandoned
                      wherever they had got to.

🔴 The `occupied` section in the `continue` status is what makes it *runnable*. The
scheduler models occupancy through activity intervals, and the failed assay's interval
has ended -- so without that section the model believes tray_1 free and would carry
job3's plate onto the tray job1's plate is still sitting on. Serialising job2 and job3
onto one tray is the true price of the crack, and a plan that did not pay it could not
be executed.

Run it:

    python examples/render_stopped_job.py

It prints both runs and writes examples/outputs/stopped_job.txt. Requires the sibling
`ofplang-schedule` (the runner replans through it).
"""

from __future__ import annotations

from pathlib import Path

from ofplang.run.runner import JobRequest, RollingRunner, load_document
from ofplang.run.simulator import Simulator

HERE = Path(__file__).parent
OUT = HERE / "outputs"
WORKFLOW = HERE / "shared_refill.workflow.yaml"
ENVIRONMENT = HERE / "stopped_job.env.yaml"
JOBS = ("job1", "job2", "job3")


def _run(policy: str) -> tuple[dict, RollingRunner]:
    workflow = load_document(WORKFLOW)
    runner = RollingRunner(
        [JobRequest(id=job_id, workflow=workflow) for job_id in JOBS],
        str(ENVIRONMENT),
        poll_interval=None,
        random_seed=0,
        on_job_failure=policy,
    )
    # Every assay on tray_1 fails (D25). A scenario concern, injected here -- failure
    # injection is the Simulator's, not part of the `Backend` contract the runner
    # drives, so the backend is narrowed before it is asked.
    assert isinstance(runner.sim, Simulator)
    runner.sim.schedule_process_failure("assay", "tray_1")
    return runner.run(), runner


def _render(policy: str) -> list[str]:
    status, runner = _run(policy)
    title = f"--on-job-failure {policy}"
    lines = [title, "-" * len(title)]
    for activity in sorted(status["activities"], key=lambda a: (a["start"], a["end"])):
        what = activity.get("process") or activity.get("to_spot") or activity["kind"]
        lines.append(
            f"  {activity['start']:>3} -> {activity['end']:>3}  "
            f"{activity.get('job') or '':<6} {activity['status']:<10} "
            f"{activity.get('mode') or '':<7} {what}"
        )
    for entry in status.get("occupied") or []:
        lines.append(
            f"  occupied: {entry['spot']} since {entry['since']}"
            f" (left by {entry.get('job')})"
        )
    done = [job.id for job in runner.jobs if not job.stopped]
    stopped = [job.id for job in runner.jobs if job.stopped]
    lines.append(f"  makespan {status['now']}; finished {done or '-'}; stopped {stopped}")
    return lines


def main() -> None:
    OUT.mkdir(exist_ok=True)

    lines = ["one plate cracks in the oven; what happens to the other two"]
    lines.append("=" * 58)
    lines.append("")
    lines += _render("continue")
    lines.append("")
    lines += _render("stop")

    text = "\n".join(lines) + "\n"
    (OUT / "stopped_job.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"wrote {OUT / 'stopped_job.txt'}")


if __name__ == "__main__":
    main()
