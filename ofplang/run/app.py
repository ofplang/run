"""Library front door for running a workflow (the seam the CLIs sit on).

`RollingRunner` is the trusting core: it assumes valid, supported v0 input and never
validates. A *front door* validates once, up front, so a malformed or unsupported
workflow fails with clear diagnostics instead of being mis-run. Historically that
front door lived inside the `ofp-run` CLI; extracting it here lets any CLI -- the
`ofp-run` one, an umbrella `ofp`, or a dialect wrapper like `lc` -- share the exact
same front door and injection seam without copying it (so they cannot drift).

Two pieces, kept separate so the trusting core stays honest:

* `front_door_check(workflow)` -- the shared front door: the full `ofplang-validate`
  pass (extension-tolerant), `$import` resolution (spec §3), and the capability gate
  over the expanded document (reject valid-v0 features the runner cannot run:
  generics). It returns that expanded document so the caller runs exactly what was
  checked. A CLI calls it, prints its diagnostics, and maps a failure to a usage
  error; a library caller can too. Expansion and the gate always run (even when
  `validate=False`), because `$import` is structural and an unsupported feature
  would otherwise surface as a confusing deep error. The workflow is a path *or* an
  already-loaded document, so the in-memory route -- the recommended one for an
  embedding caller, since the runner and the scheduler both take documents -- has a
  front door too, instead of being the one route nobody checks.
* `run_workflow(...)` -- run a workflow to completion, optionally injecting a custom
  `backend_factory` (a real-hardware / subprocess backend). With `validate=True`
  (the default) it runs the front door first for convenience; a CLI that has already
  called `front_door_check` itself passes `validate=False` so validation happens
  exactly once. It returns a `RunResult`; input errors and execution failures
  propagate as exceptions for the caller to map to exit codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ofplang.validate import EXTENSION_TOLERANT, Diagnostic, expand
from ofplang.validate import validate as validate_workflow
from ofplang.validate.yamlnode import YamlError

from .runner import DEFAULT_MAX_TICKS, RollingRunner


def _import_key_present(obj: Any) -> bool:
    """True if a `$import` key appears anywhere in the document (spec 3)."""
    if isinstance(obj, dict):
        return "$import" in obj or any(_import_key_present(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_import_key_present(v) for v in obj)
    return False


UNEXPANDED_IMPORT = "workflow contains a $import; it must be expanded before running"

# Structured node kind -> the v0 feature it requires (spec 4.3). The runner has no
# representation for a structured node -- it reshapes dataflow, lifting an output to
# an Array (`map`), threading a value across iterations (`fold` / `do_while`) or
# leaving one arm unrun (`branch`) -- and the scheduler it plans through refuses them
# for the same reason. Named here so the gate can answer with the feature, and so a
# workflow meets that answer before anything runs rather than mid-flight.
_STRUCTURED_KINDS = {
    "map": "node_map",
    "fold": "node_fold",
    "do_while": "node_do_while",
    "branch": "node_branch",
}


def capability_gate(document: dict | None) -> str | None:
    """Return a reason if the (import-expanded) workflow uses a valid-v0 feature the
    runner does not support (so it is rejected cleanly instead of mis-run); else None.

    `document` is the expanded workflow document the front door already resolved
    (see `front_door_check`), so `$import` is normally gone by the time we get here;
    the `$import` check remains as a defense for a caller that hands over an
    unexpanded document. Two features are gated: the runner neither instantiates
    generic processes (`generic_processes`) nor executes a structured node
    (`node_map` / `node_fold` / `node_do_while` / `node_branch`), and either would
    otherwise surface as a confusing deep error -- the structured node as a *failed
    run*, though nothing ever ran.

    `None` (a load/expand failure) gates nothing -- that failure is already a
    diagnostic. Always checked, independent of the validate pass, which is also why
    every step here is shape-guarded: the gate is handed whatever the caller has,
    including a document validate has already found errors in, and answering `None`
    for a malformed one leaves the complaining to validate instead of raising."""
    if not isinstance(document, dict):
        return None
    if _import_key_present(document):
        return UNEXPANDED_IMPORT
    processes = document.get("processes")
    if not isinstance(processes, dict):
        return None
    for name, proc in processes.items():
        if not isinstance(proc, dict):
            continue
        if proc.get("type_params") is not None:
            return (
                f"process {name!r} uses generic type parameters "
                "(generic_processes), which the runner does not support"
            )
        body = proc.get("body")
        nodes = body.get("nodes") if isinstance(body, dict) else None
        for node in nodes if isinstance(nodes, list) else []:
            kind = node.get("kind") if isinstance(node, dict) else None
            feature = _STRUCTURED_KINDS.get(kind) if isinstance(kind, str) else None
            if feature is not None:
                return (
                    f"process {name!r} contains structured node {node.get('id')!r} "
                    f"({feature}), which the runner does not support"
                )
    return None


@dataclass
class FrontDoorResult:
    """Outcome of `front_door_check`: `ok` iff the workflow both validates and passes
    the capability gate. `diagnostics` are the validate diagnostics (to print);
    `unsupported` is the capability-gate reason, or None. `document` is the
    import-expanded workflow (plain dict) when load + `$import` resolution succeeded,
    so a caller runs exactly what was checked instead of re-reading the file; it is
    None when expansion itself failed."""

    ok: bool
    diagnostics: list = field(default_factory=list)
    unsupported: str | None = None
    document: dict | None = None


def front_door_check(
    workflow_path: str | Path | dict, *, validate: bool = True
) -> FrontDoorResult:
    """Run the shared run front door over `workflow_path`: the full ofplang-validate
    pass (extension-tolerant, skipped when `validate=False`), `$import` resolution,
    and the always-on capability gate over the expanded document. Returns a
    `FrontDoorResult` carrying that expanded `document`; the caller decides how to
    report a failure (a CLI prints the diagnostics / reason and exits with a usage
    error).

    `workflow_path` is either a path to a workflow YAML file or an already-loaded
    workflow document (a mapping) -- a caller holding one in memory (a dialect front
    door that rewrote it, a generator) gets the same check as one holding a file. Such
    a document must already be import-expanded: there is no base directory to resolve
    a relative `$import` against, so one is rejected as an unsupported feature (the
    reason the capability gate already gives) rather than expanded here. Its
    diagnostics carry no `file:line:col`, only their logical `path`, there being no
    file to point into. A document holding a value YAML cannot spell (a `datetime`, a
    `set`) is a caller error rather than a malformed workflow, so validate's
    `ValueError` -- which names the position -- propagates instead of being reported
    as a finding.

    `$import` expansion is structural (spec 2.2 step 1), so it runs even under
    `validate=False`; only the validation pass is skipped. A structural expansion
    failure (unreadable target, cycle, ...) surfaces as a single diagnostic with the
    document left None."""
    diagnostics: list = []
    document: dict | None = None
    if isinstance(workflow_path, dict) and _import_key_present(workflow_path):
        # Short-circuited because validate *raises* on an unexpanded in-memory
        # document (it has no base directory to resolve against), while this
        # function's contract is to return the rejection. Answering with the gate's
        # own reason keeps `validate=True` and `validate=False` saying the same thing.
        return FrontDoorResult(
            ok=False, unsupported=UNEXPANDED_IMPORT, document=workflow_path
        )
    if validate:
        result = validate_workflow(workflow_path, mode=EXTENSION_TOLERANT, expand=True)
        diagnostics = list(result.diagnostics)
        document = result.document
    elif isinstance(workflow_path, dict):
        # Nothing to expand: an in-memory document is already expanded (checked just
        # above). Handed on as given rather than copied -- the runner treats the
        # workflow as read-only (schedule's D30 convention). Note the asymmetry with
        # the branch above, where validate returns a fresh plain copy of the tree.
        document = workflow_path
    else:
        # Validation skipped, but still resolve $import structurally so an import
        # workflow runs the expanded form (not the raw file the gate would reject).
        try:
            document = expand(workflow_path)
        except YamlError as exc:
            pos = exc.pos
            diagnostics = [
                Diagnostic(
                    code=exc.code,
                    message=exc.message,
                    file=pos.file if pos else None,
                    line=pos.line if pos else None,
                    col=pos.col if pos else None,
                )
            ]
    unsupported = capability_gate(document)
    ok = not diagnostics and unsupported is None
    return FrontDoorResult(
        ok=ok, diagnostics=diagnostics, unsupported=unsupported, document=document
    )


class FrontDoorError(Exception):
    """Raised by `run_workflow(validate=True)` when the front door rejects the
    workflow. Carries the `FrontDoorResult` so a caller can report the diagnostics /
    unsupported reason and map it to a usage error."""

    def __init__(self, result: FrontDoorResult):
        self.result = result
        super().__init__("workflow failed the run front door")


@dataclass
class RunResult:
    """Outcome of a completed `run_workflow`: the §6/§7 `status` document, the result
    `boundary` (produced output views, D28), whether the run `failed` (an activity
    error or a whole-workflow contract violation), the structured `failure` reason
    (D36) or None, and the scheduler warnings the run collected. A run that could not
    even start -- malformed input, an infeasible replan -- does not return a
    `RunResult`; it raises (see `run_workflow`).

    `scheduler_warnings` are the scheduler's warning diagnostics, one per distinct
    code, in the order first seen. They are handed up rather than printed because
    this is a library: a CLI decides where they go. Defaulted so a caller that builds
    a `RunResult` itself (labcode does) is not broken by their arrival."""

    status: dict
    result_boundary: dict
    failed: bool
    failure: Any
    scheduler_warnings: list = field(default_factory=list)


def run_workflow(
    workflow,
    env,
    boundary: dict | None = None,
    *,
    running_task_margin: int = 0,
    random_seed: int | None = None,
    poll_interval: int | None = 1,
    backend_factory=None,
    validate: bool = True,
    observation_out: str | None = None,
    max_ticks: int | None = DEFAULT_MAX_TICKS,
    ignore_resources: bool = False,
    inventories: dict | None = None,
    occupied: list[dict] | None = None,
) -> RunResult:
    """Drive `workflow` (against `env`, optional run `boundary`) to completion and
    return a `RunResult`.

    `workflow` may instead be a list of `JobRequest`s -- a run of several named jobs,
    planned together (SPEC §6.11), read from a run document (`runner.rundoc`). Each
    carries its own workflow, boundary and release, so `boundary` must be left unset;
    `inventories` and `occupied` then say what the laboratory itself starts with,
    there being no single boundary to say it in. Every job's workflow goes through the
    same front door, one at a time, so a rejection names the job it came from.

    `workflow` and `env` are each either a path to a YAML file or an already-loaded
    document (a mapping) -- the former lets a caller run a workflow it rewrote in memory
    without a temp file, the latter lets one that already read the environment (a dialect
    front door inspecting `x-` keys) not have it read again. Both routes are checked the
    same way: `validate=True` front-doors an in-memory workflow exactly as it front-doors
    a file (such a document must already be import-expanded -- see `front_door_check`).
    The environment is validated on neither route; the front door does not read it.

    With `validate=True` (default) the front door runs first and a rejection raises
    `FrontDoorError`; a CLI that already called `front_door_check` passes
    `validate=False` so validation happens once. `backend_factory` injects an
    alternative execution backend (e.g. `subprocess_backend_factory(...)` for real,
    out-of-process execution); None uses the default in-process simulator.
    `observation_out`, if given, streams the observation document (D38: completed
    activities' I/O views) to that path as the run proceeds. `max_ticks` bounds the loop
    iterations before the run is called non-terminating -- one iteration per poll interval,
    so it also bounds the makespan a fixed-interval run can reach; None lifts it.
    `ignore_resources` switches the consumable model off (SPEC §4.7.3), so an environment
    whose modes consume runs without the boundary saying what its stocks started with.

    Malformed workflow/environment YAML or an unparsable contract (`yaml.YAMLError`,
    `ContractSyntaxError`) and execution failures (`SimulatorError`, `RunnerError`)
    propagate to the caller, which maps them to exit codes -- keeping this a thin
    front door, not an error-swallowing wrapper."""
    if validate:
        if isinstance(workflow, list):
            checked = []
            for request in workflow:
                fd = front_door_check(request.workflow, validate=True)
                if not fd.ok:
                    raise FrontDoorError(fd)
                checked.append(replace(request, workflow=fd.document))
            workflow = checked
        else:
            fd = front_door_check(workflow, validate=True)
            if not fd.ok:
                raise FrontDoorError(fd)
            # Run exactly what the front door validated/expanded: the import-resolved
            # document, not a re-read of the raw file (so a `$import` workflow runs).
            workflow = fd.document

    runner = RollingRunner(
        workflow,
        env,
        boundary,
        running_task_margin=running_task_margin,
        random_seed=random_seed,
        poll_interval=poll_interval,
        backend_factory=backend_factory,
        observation_out=observation_out,
        max_ticks=max_ticks,
        ignore_resources=ignore_resources,
        inventories=inventories,
        occupied=occupied,
    )
    try:
        status = runner.run()
    finally:
        # A real backend (e.g. SubprocessBackend) may hold child processes / temp
        # files; give it a chance to clean up whether the run finished or raised.
        close = getattr(runner.sim, "close", None)
        if callable(close):
            close()

    return RunResult(
        status=status,
        result_boundary=runner.result_boundary,
        failed=runner.failed,
        failure=runner.failure,
        scheduler_warnings=runner.scheduler_warnings,
    )
