"""Drive an execution plan to completion on a backend (spec §6/§7; dev-notes D19).

This is milestone **2a**: replay a given plan (§6) on the simulator, with no
replanning. The plan is already a feasible, fully-timed schedule; the runner
reproduces its timeline against the backend and confirms every activity actually
executes, then emits an execution status (§6/§7) recording the run.

How it drives the backend:

* Seed the workflow's boundary inputs (`interface.inputs`, §6.8) onto their spots
  with `place()` -- that is where the entry Objects sit at the start.
* Sweep the plan's event times (every activity start and end) in order. At each
  time: `advance` the virtual clock to it, poll the backend for completions, then
  dispatch every activity that starts then. Because dispatch is now-start only
  (D15) and the plan is feasible, an activity's predecessors have completed by the
  time it starts, so the backend's preconditions (D16) hold.
* Relays and zero-distance same-spot transports carry no physical operation
  (D14/D19): they are bookkeeping and are marked complete without a dispatch.

Each real dispatch passes the plan's own duration (`end - start`), so the backend
reproduces the plan's timeline exactly rather than re-deriving it from the
environment. The runner holds the provenance the backend lacks (D10): which
backend operation each plan activity became.

The rolling-horizon replanning loop (poll -> status -> schedule -> dispatch, D9)
is milestone **2b** and is not built here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..simulator import Simulator


class RunnerError(Exception):
    """The run could not be completed (e.g. an activity never finished). Backend
    contract violations surface as `ofplang.run.simulator.SimulatorError`."""


@dataclass
class _Record:
    """The runner's per-activity bookkeeping: the original §6 activity plus how it
    was executed (whether it became a backend operation, and its final status)."""

    activity: dict
    kind: str
    start: int
    end: int
    dispatched: bool = False  # a real backend operation was sent for this activity
    uuid: str | None = None
    status: str = "pending"  # "pending" -> "running" -> "completed"


def _is_bookkeeping(activity: dict) -> bool:
    """A relay, or a zero-distance same-spot transport, has no physical operation
    to dispatch (D14/D19); the runner just tracks it."""
    kind = activity.get("kind")
    if kind == "relay":
        return True
    if kind == "transport" and activity.get("from_spot") == activity.get("to_spot"):
        return True
    return False


class Runner:
    """Replays an execution plan (§6) on a backend (milestone 2a)."""

    def __init__(self, plan: dict, environment):
        # `plan` is a parsed §6 document; `environment` is anything the simulator
        # accepts (an Environment, a §5 mapping, or a path to a §5 YAML file).
        self.plan = plan
        self.sim = Simulator(environment)
        self._records: list[_Record] = []

    def run(self) -> dict:
        """Drive the plan to completion and return the resulting status document
        (§6/§7). Raises `RunnerError` if any activity fails to complete;
        `SimulatorError` propagates if the backend rejects a dispatch (the plan is
        physically inconsistent)."""
        activities = self.plan.get("activities") or []
        self._records = [
            _Record(activity=a, kind=a.get("kind"), start=int(a["start"]), end=int(a["end"]))
            for a in activities
        ]

        # Seed the boundary inputs: the entry Objects sit on their interface spots
        # at the start of the run (§6.8). Must happen before any dispatch.
        interface = self.plan.get("interface") or {}
        for _port, spot in (interface.get("inputs") or {}).items():
            self.sim.place(spot)

        # Sweep every event time (activity starts and ends) in order. Ends are
        # included so the clock stops at each completion; starts, so each activity
        # is dispatched at its planned time.
        times = sorted({r.start for r in self._records} | {r.end for r in self._records})
        for t in times:
            # Advance to this time (completing backend operations up to it), see
            # what finished, then start what begins now.
            self.sim.advance(t)
            self._poll()
            for rec in self._records:
                if rec.status == "pending" and rec.start == t:
                    self._start(rec)

        # Settle any zero-duration operation dispatched at the final time, then
        # confirm the whole plan executed.
        self.sim.advance(self.sim.now)
        self._poll()
        incomplete = [r for r in self._records if r.status != "completed"]
        if incomplete:
            raise RunnerError(f"{len(incomplete)} activities did not complete")

        return self._build_status()

    def _start(self, rec: _Record) -> None:
        """Begin an activity: dispatch a real operation to the backend, or mark a
        bookkeeping activity (relay / same-spot) complete outright."""
        if _is_bookkeeping(rec.activity):
            rec.status = "completed"
            return

        act = rec.activity
        duration = rec.end - rec.start  # reproduce the plan's own timing
        if rec.kind == "processing":
            rec.uuid = self.sim.dispatch_processing(act["process"], act["mode"], duration=duration)
        elif rec.kind == "transport":
            rec.uuid = self.sim.dispatch_transport(
                act.get("transporter"), act["from_spot"], act["to_spot"], duration=duration
            )
        else:  # pragma: no cover - schema guarantees processing/transport/relay
            raise RunnerError(f"unknown activity kind: {rec.kind!r}")
        rec.dispatched = True
        rec.status = "running"

    def _poll(self) -> None:
        """Observe the backend (status only, D18) and mark newly finished
        operations complete."""
        for rec in self._records:
            if rec.status == "running" and rec.dispatched:
                if self.sim.state(rec.uuid)["status"] == "completed":
                    rec.status = "completed"

    def _build_status(self) -> dict:
        """Assemble the execution status (§6/§7) recording the completed run: every
        activity marked `completed` at its actual times, with `now` at the makespan
        and the `interface` carried through unchanged (§6.8)."""
        out_activities = []
        for rec in self._records:
            entry = dict(rec.activity)  # preserve provenance (node / arc / seq)
            entry["status"] = "completed"
            entry["start"] = rec.start
            entry["end"] = rec.end
            out_activities.append(entry)

        makespan = max((r.end for r in self._records), default=0)

        # Emit a readable top-level order: time, now, interface, activities, meta.
        status: dict = {}
        if "time" in self.plan:
            status["time"] = self.plan["time"]
        status["now"] = makespan
        if "interface" in self.plan:
            status["interface"] = self.plan["interface"]
        status["activities"] = out_activities
        if "meta" in self.plan:
            status["meta"] = self.plan["meta"]
        return status
