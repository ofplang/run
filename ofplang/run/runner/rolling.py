"""Rolling-horizon runner (spec §7; dev-notes D9/D20 milestone 2b-1).

Drives a workflow to completion by replanning as it goes. Each tick it renders
its committed history as an execution status (§6/§7), calls the scheduler for a
fresh plan, dispatches the pending activities that can start now, then advances
the virtual clock and polls the backend. It repeats until the scheduler returns a
plan with no pending work.

Milestone 2b-1 is happy-path: no device faults and no duration variance, so the
run follows the plan and the clock steps to planned event boundaries -- actual
times equal planned times. What 2b-1 exercises is the *round-trip*: every tick
the committed history (completed / running activities, `now`, the `interface`
boundary) is fed back to the scheduler, which fixes it and re-optimises the rest
(D9). Device up/down, re-routing and completion-time estimation are 2b-2.

Only committed history is stable across replans (D9); pending identities may be
regenerated each replan, so pending work is always re-read from the fresh plan.
"""

from __future__ import annotations

from ..simulator import Simulator
from .provenance import Committed, CommitLog
from .runner import RunnerError
from .schedule_client import replan
from .status import build_status


class RollingRunner:
    """Drives workflow + environment (+ interface) to completion by replanning."""

    def __init__(
        self,
        workflow_path,
        environment_path,
        interface: dict | None = None,
        *,
        running_task_margin: int = 0,
        random_seed: int | None = None,
        max_ticks: int = 100_000,
    ):
        self.workflow_path = str(workflow_path)
        self.environment_path = str(environment_path)
        self.sim = Simulator(environment_path)  # the backend reads the environment itself
        self.interface = interface or {}
        self.margin = running_task_margin
        self.seed = random_seed
        self.max_ticks = max_ticks

        self.log = CommitLog()
        self.now = 0
        self.ticks = 0  # number of replan cycles (a test asserts >1: history round-trips)
        self._last_time = None  # `time` section echoed from the most recent plan

    def run(self) -> dict:
        """Drive to completion and return the final execution status (§6/§7). Raises
        `RunnerError` if a replan produces no plan (infeasible) or the run cannot
        progress; `SimulatorError` propagates if the backend rejects a dispatch."""
        # Seed the boundary inputs: the entry Objects sit on their interface spots
        # at the start of the run (§6.8).
        for _port, spot in (self.interface.get("inputs") or {}).items():
            self.sim.place(spot)

        while True:
            self.ticks += 1
            if self.ticks > self.max_ticks:
                raise RunnerError("exceeded max ticks (possible non-termination)")

            # 1. Render committed history as a status and replan.
            status_doc = build_status(self.log.records(), self.now, self.interface)
            report = replan(
                self.workflow_path,
                self.environment_path,
                status_doc,
                running_task_margin=self.margin,
                random_seed=self.seed,
            )
            if not report.ok:
                raise RunnerError(self._failure_message(report))
            plan = report.plan
            self._last_time = plan.get("time")

            # 2. Pending work is what carries no status (relays are scheduler-derived
            #    and never dispatched, §7).
            pending = [
                a
                for a in plan.get("activities", [])
                if a.get("status") in (None, "pending") and a.get("kind") != "relay"
            ]
            if not pending:
                break  # everything is committed -> done

            # 3. Dispatch everything that can start now. Pending is optimised at/after
            #    `now`, so these are the entries at exactly `now`; their predecessors
            #    finished by now (we polled on the previous advance), so the backend's
            #    preconditions hold.
            for act in pending:
                if int(act["start"]) <= self.now:
                    self._commit_start(act)

            # 4. Advance to the next event boundary: the earliest future pending start
            #    or running-operation finish.
            future = [int(a["start"]) for a in pending if int(a["start"]) > self.now]
            future += [r.end for r in self.log.running() if r.end > self.now]
            if not future:
                # Only now-work this tick (e.g. same-spot no-ops); settle and replan.
                self.sim.advance(self.now)
                self._poll()
                continue
            self.now = min(future)
            self.sim.advance(self.now)
            self._poll()

        # Drain any operations still running when the last work was dispatched.
        while self.log.running():
            self.now = max(r.end for r in self.log.running())
            self.sim.advance(self.now)
            self._poll()

        return build_status(self.log.records(), self.now, self.interface, self._last_time)

    def _commit_start(self, activity: dict) -> None:
        """Start a pending activity now: dispatch it to the backend (or record a
        same-spot no-op as bookkeeping) and add it to the committed history."""
        kind = activity["kind"]
        start = self.now
        duration = int(activity["end"]) - int(activity["start"])  # reproduce planned timing
        end = start + duration

        # A same-spot transport is a physical no-op: no backend operation, completed
        # at once (D14/D19). It is still a committed leg, so it is recorded (the
        # scheduler pins the chain by it on the next replan).
        if kind == "transport" and activity.get("from_spot") == activity.get("to_spot"):
            self.log.add(Committed(activity, kind, "completed", start, end, uuid=None))
            return

        if kind == "processing":
            uuid = self.sim.dispatch_processing(activity["process"], activity["mode"], duration=duration)
        elif kind == "transport":
            uuid = self.sim.dispatch_transport(
                activity.get("transporter"), activity["from_spot"], activity["to_spot"], duration=duration
            )
        else:  # pragma: no cover - schema guarantees processing/transport/relay
            raise RunnerError(f"unknown activity kind: {kind!r}")
        self.log.add(Committed(activity, kind, "running", start, end, uuid=uuid))

    def _poll(self) -> None:
        """Mark running operations the backend reports as finished (status-only, D18)."""
        for rec in self.log.running():
            if rec.uuid is not None and self.sim.state(rec.uuid)["status"] == "completed":
                rec.status = "completed"

    @staticmethod
    def _failure_message(report) -> str:
        codes = ", ".join(str(getattr(d, "code", d)) for d in report.diagnostics)
        detail = f" ({codes})" if codes else ""
        return f"scheduler produced no plan; outcome={report.outcome}{detail}"
