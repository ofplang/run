"""Render the re-routing example: the initial plan vs the actual run.

Scenario (see reroute.workflow.yaml / reroute.env.yaml): `target` can run on
station_1 (cheap) or station_2. The scheduler's initial plan routes it to
station_1. Mid-run, station_1 goes down just after the sample is delivered there,
so the rolling-horizon runner re-routes target to station_2 via a relay and a
re-transport. Device up/down is a simulator concern driven from Python (there is
no CLI knob for it): here we call `sim.schedule_device_down` directly.

Run it:

    python examples/render_reroute.py

It writes two SVG Gantt charts under examples/outputs/:
  - reroute.initial.svg  -- the plan as first proposed (target on station_1)
  - reroute.final.svg    -- the actual execution (target re-routed to station_2)

Requires the sibling `ofplang-schedule` to be installed (pip install -e
../ofplang-schedule): the runner replans through it, and its visualizer draws
the charts.
"""

from __future__ import annotations

from pathlib import Path

from ofplang.run.runner import RollingRunner
from ofplang.schedule.scheduler.api import schedule
from ofplang.schedule.scheduler.visualize import render_svg

HERE = Path(__file__).parent
OUT = HERE / "outputs"
WORKFLOW = HERE / "reroute.workflow.yaml"
ENVIRONMENT = HERE / "reroute.env.yaml"

# The device where target first runs, and the virtual time it fails at -- just
# after the sample has been delivered to it (source 0-2, transport 2-3).
DOWN_DEVICE = "station_1"
DOWN_AT = 3


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # 1. The initial plan: what the scheduler proposes up front, on the full
    #    environment, before anything goes wrong (target on station_1).
    initial = schedule(str(WORKFLOW), str(ENVIRONMENT), random_seed=0)
    _write("reroute.initial.svg", initial.plan)

    # 2. The actual run: drive the workflow while station_1 goes down mid-run, so
    #    the runner re-routes. The final execution status is the schedule as it
    #    actually happened (target on station_2).
    runner = RollingRunner(str(WORKFLOW), str(ENVIRONMENT), random_seed=0)
    runner.sim.schedule_device_down(DOWN_AT, DOWN_DEVICE)
    final = runner.run()
    # The status carries no solver objective; label the chart with its makespan.
    final.setdefault("objective", {"kind": "makespan", "value": final.get("now")})
    _write("reroute.final.svg", final)

    print(f"initial makespan = {initial.makespan}; final (re-routed) makespan = {final['now']}")
    print(f"wrote {OUT / 'reroute.initial.svg'}")
    print(f"wrote {OUT / 'reroute.final.svg'}")


def _write(name: str, plan: dict) -> None:
    (OUT / name).write_text(render_svg(plan, view="device"), encoding="utf-8")


if __name__ == "__main__":
    main()
