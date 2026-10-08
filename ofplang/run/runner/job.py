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
from .contracts import ArrayType, Contracts, default_value, is_object_bearing, to_descriptor
from .dataflow import from_workflow
from .failure import Failure
from .runner import RunnerError
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
    # What this job's values say about how its workflow expands (schedule SPEC §6.13):
    # the length of every Pure Data Array entry input, counted off the value the run
    # was given. The scheduler is never given values, so this is how a `map` / `fold`
    # over one is planned (`expansion_of`). None where there is nothing to state.
    expansion: dict | None = None

    # 🔴 What the scheduler promised this job (`bound`) and the digest of the workflow
    # it was planned for (`fingerprint`) are deliberately NOT here. They used to be:
    # the runner rebuilt the roster from its own job list every tick, so a promise it
    # did not hold on to was gone by the second one, and every job looked like a new
    # arrival. The runner now carries the plan itself (`echo.Echo`), and the roster in
    # it is the scheduler's own -- so the promise is never rewritten, and cannot be
    # lost by failing to copy it.

    # Whether this job's boundary material has been placed. Entry material is *there*,
    # given, from the job's release (§6.8) -- so the runner places it when the clock
    # reaches that, not at run start, or the scheduler would be planning around a spot
    # that is physically full while it believes it free. This is also exactly what a
    # job arriving mid-run does, which is why it is a step in the tick rather than a
    # line in the constructor.
    placed: bool = False

    # Whether this job has stopped: something of its own failed, so no more of ITS
    # work is dispatched (SPEC §6.2). The other jobs of the run are unaffected -- a
    # laboratory does not down tools because one plate cracked -- which is the whole
    # point of the job being the unit here rather than the run.
    #
    # 🔴 The scheduler needs no telling. A terminal activity in the status stops the
    # job on its side too, and it answers the next replan with that job's remaining
    # work already `cancelled`. This flag exists because there is a *window* before
    # that answer arrives: a tick may legitimately decide it need not replan, and the
    # stale plan still lists this job's work as dispatchable.
    stopped: bool = False
    # Why it stopped (D36). The run records the first failure of the whole run; this
    # records the one that stopped this job, so a run where two jobs failed for
    # different reasons can say both.
    failure: Failure | None = None
    # 🔴 What the failing activity was holding is not recorded here either. The spots a
    # failed activity touched are claimed whatever the backend says -- a failed
    # transport claims both ends, since nothing says which one its plate is at -- but
    # that is read off the `failed` activity in the document, by the scheduler, on every
    # solve (§6.12). The runner stamps the failure and the claim follows from it.

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
    # The `LengthCheck`s (by position in `dataflow.length_checks`) already made: each
    # is made once, as soon as the value it is about exists.
    checked_lengths: set = field(default_factory=set)

    # What the run has to say about this job without failing it (D59): an entry input
    # the boundary did not supply, which runs on a typed default, and a final output
    # the run cannot report. Derived from the job's description alone, so they are
    # known before anything runs.
    warnings: list = field(default_factory=list)

    def roster_entry(self) -> dict:
        """This job as a `jobs` entry of the execution document (§6.11), *as the
        runner states it* -- when the run opens, and when the job arrives.

        Four fields, and they are the four the runner knows: who the job is, the
        earliest it may start, where its boundary material sits, and how long the lists
        of values it was given are.

        🔴 What the **scheduler** decides is not here. A job's promise (`bound`) and the
        digest of the workflow it was planned for (`fingerprint`) are written by the
        plan and carried back to it untouched, because the document the runner hands
        over *is* the last plan (`echo.Echo`). This entry is only ever the opening
        statement about a job the plan does not know yet; everything after that is the
        plan's own words.

        `release` is always written, 0 included. An absent release means 0 for a job the
        roster names and `now` for one it does not, so stating it is what makes the
        entry say what it means -- and for an arriving job the difference is real: the
        moment it arrived, not the moment of the next replan.
        """
        entry: dict = {"id": self.id, "release": self.release}
        if self.interface:
            entry["interface"] = self.interface
        if self.expansion:
            entry["expansion"] = self.expansion
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
    contracts = Contracts.from_workflow(workflow)
    # Structural boundary errors (an unknown port, a missing / stray spot) surface
    # here, up front; a supplied view value's conformance is checked when it is
    # seeded.
    parsed_boundary = parse_boundary(boundary, contracts)
    # Flattened with the same `interface` and `expansion` the scheduler is handed, so
    # an Array at the boundary -- of Objects, or of values -- expands to the
    # invocations the plan names.
    expansion = expansion_of(contracts, parsed_boundary, workflow)
    dataflow = from_workflow(
        workflow, interface=parsed_boundary.interface or None, expansion=expansion
    )
    process_defs = (workflow or {}).get("processes") or {}
    contract_asts = parse_contracts(process_defs, contracts)
    _check_composite_sources(dataflow, contracts, contract_asts)
    return Job(
        id=id,
        workflow=workflow,
        dataflow=dataflow,
        contracts=contracts,
        boundary=parsed_boundary,
        release=release,
        expansion=expansion,
        warnings=_warnings(dataflow, parsed_boundary, id),
        # Resolved port types (D27 F1): the per-process output descriptors, so the
        # backend can generate typed values (F2).
        output_schemas={
            name: {port: to_descriptor(rt) for port, rt in pc.outputs.items()}
            for name, pc in contracts.processes.items()
        },
        # The raw process definitions, passed to the device model at dispatch so it
        # can act on a process's declared structure (D27 F4b / principle A).
        process_defs=process_defs,
        contract_asts=contract_asts,
        # Whether the entry process is a composite (the usual case). Its contracts are
        # the whole-workflow envelope, checked at run start / run end (D33); an atomic
        # entry is instead a single activity, checked on the activity path.
        entry_is_composite=(process_defs.get(contracts.entry) or {}).get("kind") == "composite",
        # Nested composite invocation boundaries, keyed by node path (D34).
        composites=dataflow.composites,
    )


def expansion_of(
    contracts: Contracts, boundary: Boundary, workflow: dict | None = None
) -> dict | None:
    """The `expansion` section (schedule SPEC §6.13) this job's values give: the
    length of every Pure Data Array entry input, in declaration order, and -- given
    the `workflow` -- the arm of every branch whose condition the boundary holds
    (`_decided_arms`).

    **Every one**, not only those a `map` / `fold` traverses: which ones are traversed
    is known once the workflow is expanded, which is the scheduler's to do, and it
    passes over a length nothing reads (design.md D62 H2). An Array of Objects is left
    out -- its length is its `interface` binding's -- and so is everything that is not
    an Array.

    The length is the value's that will be seeded: the one supplied, or the empty list
    an unsupplied Array defaults to (`contracts.default_value`; `entry_input_defaulted`
    says so). A supplied value that is not a list has no length to state; seeding it
    refuses it as not conforming to its type, before anything runs."""
    entry = contracts.entry
    entry_inputs = contracts.processes[entry].inputs if entry is not None else {}
    lengths = []
    for port, resolved in entry_inputs.items():
        if not isinstance(resolved, ArrayType) or is_object_bearing(resolved):
            continue
        value = boundary.entry_values.get(port, [])
        if isinstance(value, list):
            lengths.append({"node": [], "port": port, "length": len(value)})
    expansion: dict = {"lengths": lengths} if lengths else {}
    if workflow is not None:
        arms = _decided_arms(workflow, entry_inputs, boundary, expansion)
        if arms:
            expansion["arms"] = arms
    return expansion or None


# More rounds than any workflow nests branches. Each round decides at least one branch
# or stops, so this is a guard against a defect, never a limit a workflow meets.
_MAX_ARM_ROUNDS = 1000


def _decided_arms(workflow: dict, entry_inputs: dict, boundary: Boundary,
                  expansion: dict) -> list[dict]:
    """The arm of every branch the expansion reaches whose condition is an entry
    input -- or one element of one, for a branch inside a `map` (schedule D63).

    Asked of the scheduler's expansion rather than worked out here, so the runner
    never reads the workflow's structure a second way: `undecided_branches` names the
    branches it reached without an arm and where each condition comes from. Those the
    boundary holds are decided -- `true` is `then` -- and the expansion is asked
    again, since a branch inside an arm appears only once that arm is decided. A
    condition produced during the run is not the runner's to decide yet; it is left,
    and the scheduler refuses the workflow by name.

    The value is the one that will be seeded: supplied, or the type's default --
    `false`, the `else` arm -- for an entry input the boundary leaves out
    (`entry_input_defaulted` says so). A supplied value that is not a Boolean is
    refused here, as seeding would refuse it, since it cannot decide an arm."""
    from ofplang.schedule.scheduler.model import SourceRef
    from ofplang.schedule.scheduler.workflow import undecided_branches

    interface = boundary.interface or None
    arms: list[dict] = []
    for _round in range(_MAX_ARM_ROUNDS):
        undecided = undecided_branches(
            workflow, interface=interface, expansion={**expansion, "arms": arms}
        )
        decided = []
        for path, source in undecided.items():
            if not (isinstance(source, SourceRef) and source.node == ()):
                continue  # produced during the run: not decidable before it
            port = source.port
            if port in boundary.entry_values:
                value = boundary.entry_values[port]
            elif port in entry_inputs:
                value = default_value(entry_inputs[port])
            else:
                continue
            for i in source.index:
                value = value[i] if isinstance(value, list) and i < len(value) else None
            if not isinstance(value, bool):
                raise RunnerError(
                    f"job value for entry input {port!r} does not conform to its type: "
                    f"it decides branch {'/'.join(str(step) for step in path)} and is "
                    "not a Boolean"
                )
            decided.append({"node": list(path), "arm": "then" if value else "else"})
        if not decided:
            return arms
        arms.extend(decided)
    raise RunnerError("deciding the branches' arms did not settle")  # pragma: no cover


@dataclass(frozen=True)
class RunWarning:
    """Something a run says without failing (D59): a value it made up. `job` is the
    job's id, empty for a single-workflow run."""

    code: str
    message: str
    job: str = ""


def _warnings(dataflow, boundary: Boundary, job_id: str) -> list:
    """What a job's description alone tells the run to warn about.

    - `entry_input_defaulted`: an entry input the boundary supplied no value for runs
      on its type's default -- the one value the runner makes up, so it never does so
      without saying (an Object's whole view as much as a Pure Data value).

    (A final output with no source used to be warned about here, as
    `output_unreported`: an entry Object returned untouched, which the scheduler left
    out of scope. It records one now, so a missing one is refused as the defect it
    is -- `dataflow.from_workflow`.)
    """
    found: list = []
    for port in dataflow.entry_ports:
        if port not in boundary.entry_values:
            found.append(RunWarning(
                "entry_input_defaulted",
                f"entry input {port!r} was not supplied a value at the boundary; "
                f"it runs on its type's default",
                job_id,
            ))
    return found


def _check_composite_sources(dataflow, contracts: Contracts, contract_asts: dict) -> None:
    """Refuse a nested composite with contracts whose ports are not all sourced.

    A contract reads every port it names, so a port with no value would fail it with
    a lookup error mid-run -- or, worse, be read as something it is not. v0 binds
    every input port and returns every output (§11), so this is the unvalidated
    document's problem, caught before anything runs."""
    from ofplang.schedule.core.identifiers import format_node_path

    for path, boundary in dataflow.composites.items():
        if not contract_asts.get(boundary.process):
            continue
        declared = contracts.processes.get(boundary.process)
        if declared is None:
            continue
        missing = [f"inputs.{p}" for p in declared.inputs if p not in boundary.inputs]
        missing += [f"outputs.{p}" for p in declared.outputs if p not in boundary.outputs]
        if missing:
            raise RunnerError(
                f"composite {format_node_path(path)} ({boundary.process}) has contracts "
                f"but no source for {', '.join(missing)}"
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
