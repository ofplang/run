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

import copy

from ..simulator import Simulator
from .loader import load_document
from .provenance import Committed, CommitLog
from .runner import RunnerError
from .schedule_client import replan
from .status import build_status


def _normalize_mode_ids(environment: dict) -> dict:
    """Return a copy of `environment` with an explicit `id` on every process mode
    that lacks one.

    This must happen before any reduction: dropping a mode renumbers the remaining
    position-based ids, so a reduced-env plan's mode id would no longer map to the
    same physical mode in the backend's full environment. Pinning ids up front
    keeps the id -> mode mapping stable across reduction (D21). Ids are `m<i>`
    rather than the bare position, because a mode id must be a v0 identifier
    (§8.1) and so cannot start with a digit.
    """
    env = copy.deepcopy(environment)
    for process in (env.get("processes") or {}).values():
        for i, mode in enumerate(process.get("modes") or []):
            if mode.get("id") is None:
                mode["id"] = f"m{i}"
    return env


def _reduce_environment(environment: dict, down: set[str]) -> dict:
    """Return a copy of `environment` with every process mode that uses a down
    device removed, keeping the device/spot/transport definitions (spec §7, D21).

    Dropping only the modes is how a re-route is triggered: committed transports to
    a down device's spot stay valid, and a re-transport can still route through it,
    but no new processing is scheduled there.
    """
    reduced = copy.deepcopy(environment)
    for process in (reduced.get("processes") or {}).values():
        process["modes"] = [
            mode for mode in (process.get("modes") or []) if not (set(mode.get("devices") or []) & down)
        ]
    return reduced


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
        poll_interval: int | None = None,
        max_ticks: int = 100_000,
    ):
        self.workflow_path = str(workflow_path)
        self.environment_path = str(environment_path)
        # Keep the environment as a dict too: when devices go down we schedule
        # against a reduced copy of it (D21), while the backend keeps the full one.
        # Mode ids are pinned up front so they stay stable when modes are dropped.
        self._environment = _normalize_mode_ids(load_document(environment_path))
        self.sim = Simulator(self._environment)  # the backend reads the environment itself
        self.interface = interface or {}
        self.margin = running_task_margin
        self.seed = random_seed
        # None -> advance to plan event boundaries (deterministic, exact times, the
        # default). An integer -> poll every that many ticks: the realistic mode
        # where completion is only seen at a poll and its time is estimated (D22).
        self.poll_interval = poll_interval
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

            # 1. Discover which devices are down and schedule against the normalized
            #    environment reflecting it: the full env when nothing is down, or a
            #    reduced copy (down devices' process modes dropped) that triggers a
            #    re-route (D21). Always the normalized dict, so the scheduler and the
            #    backend agree on mode ids. Committed history is fed back so it is
            #    fixed and the rest re-optimised.
            down = set(self.sim.down_devices())
            environment = _reduce_environment(self._environment, down) if down else self._environment
            status_doc = build_status(self.log.records(), self.now, self.interface)
            report = replan(
                self.workflow_path,
                environment,
                status_doc,
                running_task_margin=self.margin,
                random_seed=self.seed,
            )
            if not report.ok:
                raise RunnerError(self._failure_message(report))
            plan = report.plan
            self._last_time = plan.get("time")

            # 2. Pending work is what carries no status (relays are scheduler-derived
            #    and never dispatched, §7). The run is done when there is neither
            #    unstarted work nor anything still running.
            pending = [
                a
                for a in plan.get("activities", [])
                if a.get("status") in (None, "pending") and a.get("kind") != "relay"
            ]
            if not pending and not self.log.running():
                break

            # 3. Dispatch everything that can start now. Pending is optimised at/after
            #    `now`, so these are the entries at exactly `now`; their predecessors
            #    finished by now (we polled on the previous advance), so the backend's
            #    preconditions hold.
            for act in pending:
                if int(act["start"]) <= self.now:
                    self._commit_start(act)

            # 4. Advance the clock, then poll. The advance policy is the only thing
            #    that differs between the two modes (D22).
            self.now = self._next_time(pending)
            self.sim.advance(self.now)
            self._poll()

        return build_status(self.log.records(), self.now, self.interface, self._last_time)

    def _next_time(self, pending: list[dict]) -> int:
        """The virtual time to advance to next. In fixed-interval mode, one poll
        interval on; in event-boundary mode, the earliest future pending start or
        running-operation finish (or `now` if there is none, letting a settle pass
        clear zero-duration work)."""
        if self.poll_interval is not None:
            return self.now + self.poll_interval
        future = [int(a["start"]) for a in pending if int(a["start"]) > self.now]
        future += [r.end for r in self.log.running() if r.end > self.now]
        return min(future) if future else self.now

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
        """Mark running operations the backend reports as finished (status-only, D18).

        The completion time is recorded as the current poll time `now`. In
        event-boundary mode `now` is exactly the planned end (we advanced to it); in
        fixed-interval mode it is the poll at which completion was first seen -- an
        upper bound on the true finish, the best a poll-only observer can know (D22).
        """
        for rec in self.log.running():
            if rec.uuid is not None and self.sim.state(rec.uuid)["status"] == "completed":
                rec.status = "completed"
                rec.end = self.now

    @staticmethod
    def _failure_message(report) -> str:
        codes = ", ".join(str(getattr(d, "code", d)) for d in report.diagnostics)
        detail = f" ({codes})" if codes else ""
        return f"scheduler produced no plan; outcome={report.outcome}{detail}"
