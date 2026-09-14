"""The plan the runner carries from one replan to the next (design.md D48/D49).

A rolling run used to *rebuild* the document it handed the scheduler: read the last
plan, write the committed history out as a fresh status, and let the scheduler work
the rest out again. Everything the scheduler had decided and the document could not
re-derive -- each job's promise, its release, the moment the stock levels are stated
for, what a stopped job was holding -- had to be caught on the way past and handed
back in, and each of those was a round trip the runner could silently drop.

This carries the plan instead. The scheduler's answer *is* the next question, and
what the runner adds to it is only:

  - the history it has made since (`stamp`): a status and the times it observed;
  - the clock (`now`);
  - the facts about the world the plan cannot contain -- a job that has arrived
    (`admit`), a spot someone has loaded or cleared (`freeze` / `free_spot`).

Everything else is echoed untouched, which is what makes the round trips disappear:
a promise the runner never rewrites is a promise it cannot lose.

🔴 **The document is not handed out.** Everything above is a method, and the one
accessor returns a copy, so "what the runner may edit" is a list of five operations
rather than an unwritten rule about a dict. That list is the invariant; see
`Echo.stamp` for the one part of it that is not obvious.
"""

from __future__ import annotations

import copy

from .provenance import Committed, activity_key


def _job_of(activity: dict) -> str:
    """Which job an activity belongs to, as the roster names it.

    A single-workflow plan writes no `job` on anything and its one job is named by the
    empty string, so the two spellings of "the only job there is" have to read alike --
    otherwise a single workflow's abandoned work would never be marked cancelled, and
    its final document would report work that never ran as still to come.
    """
    return activity.get("job") or ""


class Echo:
    """The execution document a rolling run carries, and the only way to change it.

    Opened from what the run was asked to do, replaced wholesale by each plan the
    scheduler answers with (`adopt`), and edited in between only through the methods
    below.
    """

    def __init__(
        self,
        *,
        interface: dict | None = None,
        inventories: dict | None = None,
        occupied: list[dict] | None = None,
        jobs: list[dict] | None = None,
    ) -> None:
        """Open a run's document, before anything has happened.

        `now` at 0 and no activities, plus the four things the caller said about the
        run: where the boundary material sits (`interface`, §6.8 -- the single-workflow
        form, since a run of named jobs carries one per roster entry instead), what the
        stocks hold to begin with (`inventories`, §6.10), what the laboratory was
        already holding (`occupied`, §6.12), and the roster (`jobs`, §6.11).

        🔴 This is the only document the runner ever builds. Every one after it is a
        plan the scheduler wrote.

        The failure *reason* (D36) is deliberately in none of them: the document has to
        stay a valid §6 execution document, so the reason is exposed out of band through
        `RollingRunner.failure` and the CLI's stderr.
        """
        # Readable top-level order: now, jobs, interface, inventories, occupied,
        # activities -- the order a rendered plan uses, so the opening document and
        # every plan after it read the same way.
        doc: dict = {"now": 0}
        if jobs:
            doc["jobs"] = copy.deepcopy(jobs)
        if interface:
            doc["interface"] = copy.deepcopy(interface)
        if inventories:
            doc["inventories"] = copy.deepcopy(inventories)
        if occupied:
            doc["occupied"] = copy.deepcopy(occupied)
        doc["activities"] = []
        self._doc = doc

    # -- what the scheduler is handed, and what it answers with ------------------

    @property
    def document(self) -> dict:
        """A copy of the document as it stands, for handing to the scheduler.

        Copied rather than shared: the scheduler reads an in-memory document as it
        stands and never writes to it, but a caller holding this would be able to,
        and the point of this class is that there is no such caller.
        """
        return copy.deepcopy(self._doc)

    def adopt(self, plan: dict) -> None:
        """Take the scheduler's answer as the document to carry from here.

        Wholesale, not merged. The plan is a complete §6 document that already echoes
        every planning input it was given, so merging could only reintroduce the thing
        this class exists to remove: a field the runner decides to carry over itself.
        What the plan drops -- a job that has left the roster, the work it did -- is
        dropped because the scheduler was asked to drop it.
        """
        self._doc = copy.deepcopy(plan)

    # -- what the runner adds --------------------------------------------------

    def stamp(self, records: list[Committed], now: int, stopped: set[str]) -> None:
        """Write this run's history onto the plan, and set the clock to `now`.

        Each committed record finds the entry it was dispatched from by provenance
        (`activity_key`) and leaves its status and its times there. A completed
        record's `end` is the finish that was observed; a running one's is still the
        plan's expectation, which is all the runner knows until it sees otherwise.

        🔴 **A stopped job's remaining work is stamped `cancelled` here**, and this is
        not bookkeeping: a terminal activity is the *only* way the scheduler learns a
        job has stopped (§6.2), and a job can stop with nothing of its own having
        failed -- a contract it did not satisfy, a policy that stops every job when one
        does, a replan that could not be made. Without this the scheduler would go on
        planning that job's work, and -- because what a stopped job is still holding is
        derived from its being stopped (§6.12) -- would send another job's material
        onto the spot its plate is sitting on. Zero-length at `now`: it reached a
        terminal state without executing.

        🔴 **A relay is never stamped.** The runner does not dispatch one, so it has no
        record of one; the junctions are in the plan this carries, and the scheduler
        regenerates them from the committed legs either way (§6.4.1, §7).
        """
        self._doc["now"] = now
        history: dict = {activity_key(record.activity): record for record in records}
        for activity in self._doc.get("activities") or []:
            if activity.get("kind") == "relay":
                continue
            ran = history.get(activity_key(activity))
            if ran is not None:
                activity["status"] = ran.status
                activity["start"] = ran.start
                activity["end"] = ran.end
            elif activity.get("status") in (None, "pending") and _job_of(activity) in stopped:
                activity["status"] = "cancelled"
                activity["start"] = activity["end"] = now

    def declare_objective(self, stages: tuple[str, ...]) -> None:
        """State what this run is being optimised for (§4.8, §6.1).

        🔴 The one planning input the runner restates rather than echoes, and it is the
        roster that makes it so: with nothing declared the objective *is* a function of
        how many jobs there are (§4.8), and the runner is the author of the roster. Echo
        it instead and a run that started with one job would keep the one-job objective
        after a second arrived, and one that shrank to a single job would go on
        minimising a sum of completions there is no longer a choice about.

        Only `kind` is written. The `value` a plan carries alongside it is the last
        solve's report, not an input, and restating it would be handing back an answer
        as if it were a question.
        """
        self._doc["objective"] = {"kind": list(stages) if len(stages) > 1 else stages[0]}

    def admit(self, entry: dict) -> None:
        """Put an arriving job on the roster (§6.11).

        🔴 The release is stated here rather than left to the scheduler's default for an
        arrival. That default recognises an arrival as a job the roster does not name --
        but a job with boundary material *has* to be named, since the `interface` is read
        from its roster entry, so the default cannot fire for one. Measured: leaving it
        out released a job admitted at 23 at 24, the moment of the next replan, which is
        not when it arrived.

        Last on the roster, which is last in priority (§6.11): what a later arrival owes
        the jobs already being planned is to be fitted around their promises.
        """
        self._doc.setdefault("jobs", []).append(copy.deepcopy(entry))

    def freeze(self, spot: str, since: int) -> None:
        """Say a spot is held by something the plan does not otherwise account for
        (§6.12) -- material a person put there, and nothing else.

        🔴 What a job left behind is **not** written here. It follows from that job's own
        history, which this document carries, so the scheduler derives it every solve;
        stating it as well says the same hold twice and is refused
        (`occupied_already_derived`). An entry belongs here only for a hold nothing can
        derive.
        """
        occupied = self._doc.setdefault("occupied", [])
        if any(entry.get("spot") == spot for entry in occupied):
            return
        occupied.append({"spot": spot, "since": since})

    def free_spot(self, spot: str) -> bool:
        """Release a freeze this document is carrying. True if there was one.

        Named for what it does rather than for what happened. `occupied` says a spot is
        **not to be used**, not that something is on it -- a failed transport freezes
        both of its ends, though its plate is at one of them -- so lifting it is a
        declaration, and why (it was collected, it was never there, it was looked at and
        found fine) is not the document's business.

        A hold that is *derived* cannot be lifted here, and that is the same rule read
        forwards: it is not in this section, so there is nothing to remove. What ends a
        stopped job's hold is the job leaving the plan.
        """
        occupied = self._doc.get("occupied")
        if not occupied:
            return False
        kept = [entry for entry in occupied if entry.get("spot") != spot]
        if len(kept) == len(occupied):
            return False
        if kept:
            self._doc["occupied"] = kept
        else:
            del self._doc["occupied"]
        return True

    # -- what the runner reads -------------------------------------------------

    @property
    def now(self) -> int:
        return int(self._doc.get("now", 0))

    @property
    def inventories(self) -> dict:
        """What the stocks hold, as of the moment the document states (§6.10).

        Read from the document rather than remembered: the scheduler is the one that
        replays the levels and moves the moment they are stated for, and a second copy
        here is the round trip this class removes.
        """
        return copy.deepcopy(self._doc.get("inventories") or {})

    @property
    def occupied(self) -> list[dict]:
        """The freezes the document states (§6.12) -- not the ones it implies."""
        return copy.deepcopy(self._doc.get("occupied") or [])

    @property
    def roster(self) -> list[dict]:
        return copy.deepcopy(self._doc.get("jobs") or [])

    def pending(self) -> list[dict]:
        """The work this plan has not started, as dispatchable entries.

        Pending is what carries no status. A relay is excluded whatever its status: it
        is an instantaneous junction with no physical operation behind it, so there is
        nothing to dispatch and nothing to commit.
        """
        return [
            copy.deepcopy(activity)
            for activity in self._doc.get("activities") or []
            if activity.get("status") in (None, "pending") and activity.get("kind") != "relay"
        ]

    def result(self) -> dict:
        """The document to report when the run is over.

        The carried plan as it stands, less `outcome`: that is the *solve's* verdict --
        whether the search proved its answer optimal -- and a run's verdict is not the
        same question. Everything else stays, `meta` and `objective` included, so what
        the run reports is a document that can be planned from again.
        """
        doc = copy.deepcopy(self._doc)
        doc.pop("outcome", None)
        return doc
