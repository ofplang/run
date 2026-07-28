"""Library front door for running a workflow (the seam the CLIs sit on).

`RollingRunner` is the trusting core: it assumes valid, supported v0 input and never
validates. A *front door* validates once, up front, so a malformed or unsupported
workflow fails with clear diagnostics instead of being mis-run. Historically that
front door lived inside the `ofp-run` CLI; extracting it here lets any CLI -- the
`ofp-run` one, an umbrella `ofp`, or a dialect wrapper like `lc` -- share the exact
same front door and injection seam without copying it (so they cannot drift).

Two pieces, kept separate so the trusting core stays honest:

* `front_door_check(workflow)` -- the shared front door: the full `ofplang-validate`
  pass (extension-tolerant) plus the capability gate (reject valid-v0 features the
  runner cannot run: generics, `$import`). A CLI calls it, prints its diagnostics,
  and maps a failure to a usage error; a library caller can too. The capability gate
  always runs (even when `validate=False`), because those features would otherwise
  surface as a confusing deep error.
* `run_workflow(...)` -- run a workflow to completion, optionally injecting a custom
  `backend_factory` (a real-hardware / subprocess backend). With `validate=True`
  (the default) it runs the front door first for convenience; a CLI that has already
  called `front_door_check` itself passes `validate=False` so validation happens
  exactly once. It returns a `RunResult`; input errors and execution failures
  propagate as exceptions for the caller to map to exit codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from ofplang.validate import EXTENSION_TOLERANT
from ofplang.validate import validate as validate_workflow

from .runner import RollingRunner


def _import_key_present(obj: Any) -> bool:
    """True if a `$import` key appears anywhere in the document (spec 3)."""
    if isinstance(obj, dict):
        return "$import" in obj or any(_import_key_present(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_import_key_present(v) for v in obj)
    return False


def capability_gate(workflow_path: str) -> str | None:
    """Return a reason if the workflow uses a valid-v0 feature the runner does not
    support (so it is rejected cleanly instead of mis-run); else None.

    The runner handles a subset of valid v0: generic processes (`generic_processes`)
    are not instantiated and `$import` is not resolved (an expanded document is
    assumed). Either would otherwise surface as a confusing deep error. Always
    checked, independent of the validate pass."""
    try:
        data = yaml.safe_load(Path(workflow_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None  # a parse problem is handled by the validate pass / runner
    if not isinstance(data, dict):
        return None
    if _import_key_present(data):
        return "workflow contains a $import; it must be expanded before running"
    for name, proc in (data.get("processes") or {}).items():
        if isinstance(proc, dict) and proc.get("type_params") is not None:
            return (
                f"process {name!r} uses generic type parameters "
                "(generic_processes), which the runner does not support"
            )
    return None


@dataclass
class FrontDoorResult:
    """Outcome of `front_door_check`: `ok` iff the workflow both validates and passes
    the capability gate. `diagnostics` are the validate diagnostics (to print);
    `unsupported` is the capability-gate reason, or None."""

    ok: bool
    diagnostics: list = field(default_factory=list)
    unsupported: str | None = None


def front_door_check(workflow_path: str, *, validate: bool = True) -> FrontDoorResult:
    """Run the shared run front door over `workflow_path`: the full ofplang-validate
    pass (extension-tolerant, skipped when `validate=False`) plus the always-on
    capability gate. Returns a `FrontDoorResult`; the caller decides how to report it
    (a CLI prints the diagnostics / reason and exits with a usage error)."""
    diagnostics: list = []
    if validate:
        result = validate_workflow(workflow_path, mode=EXTENSION_TOLERANT)
        diagnostics = list(result.diagnostics)
    unsupported = capability_gate(workflow_path)
    ok = not diagnostics and unsupported is None
    return FrontDoorResult(ok=ok, diagnostics=diagnostics, unsupported=unsupported)


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
    error or a whole-workflow contract violation), and the structured `failure`
    reason (D36), or None. A run that could not even start -- malformed input, an
    infeasible replan -- does not return a `RunResult`; it raises (see `run_workflow`)."""

    status: dict
    result_boundary: dict
    failed: bool
    failure: Any


def run_workflow(
    workflow,
    env: str,
    boundary: dict | None = None,
    *,
    running_task_margin: int = 0,
    random_seed: int | None = None,
    poll_interval: int | None = 1,
    backend_factory=None,
    validate: bool = True,
    observation_out: str | None = None,
) -> RunResult:
    """Drive `workflow` (against `env`, optional run `boundary`) to completion and
    return a `RunResult`.

    `workflow` is either a path to a workflow YAML file or an already-loaded document
    (a mapping) -- the latter lets a caller run a workflow it rewrote in memory without
    a temp file. An in-memory document requires `validate=False`: the front door
    (`front_door_check`) validates a file, so a caller passing a document must have
    validated it beforehand (e.g. front-doored the original path).

    With `validate=True` (default) the front door runs first and a rejection raises
    `FrontDoorError`; a CLI that already called `front_door_check` passes
    `validate=False` so validation happens once. `backend_factory` injects an
    alternative execution backend (e.g. `subprocess_backend_factory(...)` for real,
    out-of-process execution); None uses the default in-process simulator.
    `observation_out`, if given, streams the observation document (D38: completed
    activities' I/O views) to that path as the run proceeds.

    Malformed workflow/environment YAML or an unparsable contract (`yaml.YAMLError`,
    `ContractSyntaxError`) and execution failures (`SimulatorError`, `RunnerError`)
    propagate to the caller, which maps them to exit codes -- keeping this a thin
    front door, not an error-swallowing wrapper."""
    if validate and isinstance(workflow, dict):
        raise ValueError(
            "run_workflow with an in-memory workflow document requires validate=False "
            "(the front door validates a file path); validate the document beforehand"
        )
    if validate:
        fd = front_door_check(workflow, validate=True)
        if not fd.ok:
            raise FrontDoorError(fd)

    runner = RollingRunner(
        workflow,
        env,
        boundary,
        running_task_margin=running_task_margin,
        random_seed=random_seed,
        poll_interval=poll_interval,
        backend_factory=backend_factory,
        observation_out=observation_out,
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
    )
