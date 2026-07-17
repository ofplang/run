"""Show the schedule drift caused by fixed-interval polling.

Reuses the reroute example's workflow and environment, but injects *no* device
fault -- this is purely about polling timing. The same workflow is driven two
ways and both are rendered under outputs/ so the drift is visible side by side:

  - event boundary (exact): completion is observed the instant it happens, so the
    schedule is exact -- source 0-2, transport 2-3, target 3-5, makespan 5.
  - fixed interval D=3: completion is only seen at the next poll, so each activity
    is recorded as finishing at that poll (an upper bound) and its successors slip
    -- source 0-3, transport 3-6, target 6-9, makespan 9.

The gap is the mismatch between when an activity actually finishes and when the
runner next looks: the backend's history shows the true finishes (2, 4, 8) while
the polled schedule reports 3, 6, 9.

Run it:

    python examples/render_poll_drift.py

Requires the sibling `ofplang-schedule` to be installed (the runner replans
through it and its visualizer draws the charts).
"""

from __future__ import annotations

from pathlib import Path

from ofplang.run.runner import RollingRunner
from ofplang.schedule.scheduler.visualize import render_svg

HERE = Path(__file__).parent
OUT = HERE / "outputs"
WORKFLOW = HERE / "reroute.workflow.yaml"
ENVIRONMENT = HERE / "reroute.env.yaml"
INTERVAL = 3


def main() -> None:
    OUT.mkdir(exist_ok=True)

    exact = _drive(poll_interval=None)
    polled = _drive(poll_interval=INTERVAL)

    _write("poll_drift.exact.svg", exact)
    _write("poll_drift.polled.svg", polled)

    print(f"exact makespan = {exact['now']}; polled (interval {INTERVAL}) makespan = {polled['now']}")
    print(f"wrote {OUT / 'poll_drift.exact.svg'}")
    print(f"wrote {OUT / 'poll_drift.polled.svg'}")


def _drive(poll_interval: int | None) -> dict:
    runner = RollingRunner(str(WORKFLOW), str(ENVIRONMENT), random_seed=0, poll_interval=poll_interval)
    status = runner.run()
    # The status carries no solver objective; label the chart with its makespan.
    status.setdefault("objective", {"kind": "makespan", "value": status.get("now")})
    return status


def _write(name: str, plan: dict) -> None:
    (OUT / name).write_text(render_svg(plan, view="device"), encoding="utf-8")


if __name__ == "__main__":
    main()
