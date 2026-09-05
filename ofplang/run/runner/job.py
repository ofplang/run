"""One workflow being run, and everything derived from it.

A rolling run used to be a run *of a workflow*: the dataflow, the resolved
contracts, the value store and the run boundary all sat on the runner, because
there was only ever one of each. Planning several workflows together
(`ofplang-schedule` SPEC §6.11) makes a run one *of a laboratory*, with several
jobs in it — so everything derived from a workflow lives here, and the runner
holds a list of these.

Building one is a function of its description alone — an id, a workflow, a
boundary, a release time — and deliberately so. **A job that arrives while the run
is already going is the same call, made later**: nothing here reads the runner, the
clock, or the other jobs, so admitting one mid-run is appending to a list rather
than reworking anything.

What is *not* here is what belongs to the laboratory rather than to a workflow: the
starting stock (`inventories`, §6.10) is one per run however many jobs draw on it,
and stays on the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .boundary import Boundary, parse_boundary
from .contract_eval import parse as parse_contract
from .contract_eval import referenced_ports
from .contracts import Contracts, to_descriptor
from .dataflow import from_workflow
from .values import ValueStore


@dataclass
class Job:
    """One workflow's run: its graph, its contracts, its boundary, and its values.

    `id` names it in the plan (§6.11) and is the empty string for the single-workflow
    run, which carries no job identity at all — matching the scheduler, whose
    single-workflow path prefixes nothing and labels nothing.

    `release` is the earliest time any of its activities may start, and `bound` the
    completion the scheduler promised it. Both live here for the same reason: the
    runner rebuilds the roster it hands the scheduler on every replan, so anything it
    does not hold survives the first tick and vanishes on the second.
    """


    id: str
    workflow: dict
    dataflow: Any
    contracts: Contracts
    boundary: Boundary
    release: int = 0

    # 🔴 What the scheduler promised this job, read back from each plan's roster
    # (§6.11) and handed straight back on the next replan. It has to live here: the
    # status is rebuilt from the commit log every tick, so a bound the runner did not
    # hold would be gone by the second one -- and every job would look like a new
    # arrival with no promise, re-derived each tick. The guarantee that an earlier job
    # is not disturbed by a later one would hold inside one solve and nowhere else.
    bound: int | None = None
    # Likewise the digest of the workflow this job runs. Carrying it back is what lets
    # the scheduler check that the workflows handed over are the ones it planned for;
    # dropping it is safe (the check is skipped) but costs the check for nothing.
    fingerprint: str | None = None

    # Whether this job's boundary material has been placed. Entry material is *there*,
    # given, from the job's release (§6.8) -- so the runner places it when the clock
    # reaches that, not at run start, or the scheduler would be planning around a spot
    # that is physically full while it believes it free. This is also exactly what a
    # job arriving mid-run does, which is why it is a step in the tick rather than a
    # line in the constructor.
    placed: bool = False

    # Derived from the workflow, all read on the dispatch path.
    output_schemas: dict = field(default_factory=dict)
    process_defs: dict = field(default_factory=dict)
    contract_asts: dict = field(default_factory=dict)
    entry_is_composite: bool = False
    composites: dict = field(default_factory=dict)

    # Mutable run state: which composite contracts have already fired (each `requires`
    # / `ensures` fires once, when its values become available, D34), the produced
    # view values, and the whole-workflow outputs assembled at the end.
    checked_requires: set = field(default_factory=set)
    checked_ensures: set = field(default_factory=set)
    values: ValueStore = field(default_factory=ValueStore)
    outputs: dict = field(default_factory=dict)

    def roster_entry(self) -> dict:
        """This job as a `jobs` entry of the execution document (§6.11).

        Everything the scheduler needs to recognise it again: who it is, when it may
        start, what it was promised, which workflow it runs, and where its boundary
        material sits.
        """
        entry: dict = {"id": self.id}
        if self.release:
            entry["release"] = self.release
        if self.bound is not None:
            entry["bound"] = self.bound
        if self.fingerprint is not None:
            entry["fingerprint"] = self.fingerprint
        if self.interface:
            entry["interface"] = self.interface
        return entry

    @property
    def interface(self) -> dict:
        """The §6.8 boundary constraint handed to the scheduler — spots only, never
        view values (D9/D26)."""
        return self.boundary.interface

    @property
    def entry_values(self) -> dict:
        """The whole-workflow input values seeded at run start: `{entry_port: view}`.

        Named for what it is. It was `job` — from the `--job` flag the boundary
        document replaced (D28) — which stopped being a workable name the moment a
        *job* also meant one of several workflows being run together.
        """
        return self.boundary.entry_values


@dataclass(frozen=True)
class JobRequest:
    """A job as *asked for*: what the run document (§6.11) says about one.

    The description, not the machinery — the four things a caller can state about a
    job, from which `build_job` derives everything else. It is a type of its own
    because a run of several jobs has to be handed several of these, and a tuple of
    positional arguments would leave the reader of a call site guessing which is the
    release and which the id.
    """

    id: str
    workflow: dict
    boundary: dict | None = None
    release: int = 0


def build_job(
    workflow: dict,
    boundary: dict | None,
    *,
    id: str = "",
    release: int = 0,
) -> Job:
    """Everything a workflow needs to be run, derived from it once.

    Called once per job at run start — and, when a job may arrive mid-run, once more
    at that point. It reads nothing but its arguments for that reason.
    """
    dataflow = from_workflow(workflow)
    contracts = Contracts.from_workflow(workflow)
    process_defs = (workflow or {}).get("processes") or {}
    return Job(
        id=id,
        workflow=workflow,
        dataflow=dataflow,
        contracts=contracts,
        # Structural boundary errors (an unknown port, a missing / stray spot) surface
        # here, up front; a supplied view value's conformance is checked when it is
        # seeded.
        boundary=parse_boundary(boundary, contracts),
        release=release,
        # Resolved port types (D27 F1): the per-process output descriptors, so the
        # backend can generate typed values (F2).
        output_schemas={
            name: {port: to_descriptor(rt) for port, rt in pc.outputs.items()}
            for name, pc in contracts.processes.items()
        },
        # The raw process definitions, passed to the device model at dispatch so it
        # can act on a process's declared structure (D27 F4b / principle A).
        process_defs=process_defs,
        contract_asts=parse_contracts(process_defs, contracts),
        # Whether the entry process is a composite (the usual case). Its contracts are
        # the whole-workflow envelope, checked at run start / run end (D33); an atomic
        # entry is instead a single activity, checked on the activity path.
        entry_is_composite=(process_defs.get(contracts.entry) or {}).get("kind") == "composite",
        # Nested composite invocation boundaries, keyed by node path (D34).
        composites=dataflow.composites,
    )


def parse_contracts(process_defs: dict, contracts: Contracts) -> dict:
    """Parse every process's `contracts` (v0 §9) into ASTs, keyed by process and
    section. All process kinds are parsed here; where each is *checked* is decided
    at run time by the process's role -- an atomic process on the activity path
    (D32), the entry composite at the run boundary (D33), a nested composite when
    its values become available (D34). A process with no `contracts` produces no
    entry.

    For an atomic process, `requires` is split by phase (D37): an expression
    referencing only run/graph-phase inputs is knowable at run start and goes to
    `requires_preflight` (checked before dispatch); one reading any data-phase input
    stays in `requires` (checked at dispatch). Composite `requires` is not split
    (the entry composite is already a run-boundary check, D33)."""
    result: dict = {}
    for name, pdef in process_defs.items():
        pdef = pdef or {}
        contract_section = pdef.get("contracts") or {}
        if not contract_section:
            continue
        is_atomic = pdef.get("kind") == "atomic"
        parsed: dict = {}
        for section in ("requires", "ensures"):
            exprs = [
                (item["expr"], parse_contract(item["expr"]))
                for item in (contract_section.get(section) or [])
                if item and item.get("expr") is not None
            ]
            if not exprs:
                continue
            if section == "requires" and is_atomic:
                # `requires` references only inputs (v0 §9.1); an expression is
                # preflightable iff every input it reads is non-data phase (v0 §6),
                # hence knowable at run start.
                preflight = [
                    (expr, ast)
                    for expr, ast in exprs
                    if all(
                        contracts.input_phase(name, port) != "data"
                        for _s, port in referenced_ports(ast)
                    )
                ]
                runtime = [pair for pair in exprs if pair not in preflight]
                if preflight:
                    parsed["requires_preflight"] = preflight
                if runtime:
                    parsed["requires"] = runtime
            else:
                parsed[section] = exprs
        if parsed:
            result[name] = parsed
    return result
