"""Command-line interface for ofplang.run.

Thin presentation layer over the library. Subcommands:

    ofp-run run <workflow> --env <env>
        [--boundary <doc>] [--boundary-out FILE]
        [--seed N] [--margin M] [--poll-interval D] [-o OUT]
        drive a workflow to completion by replanning (rolling-horizon)
    ofp-run replay <plan> --env <env> [-o OUT]
        replay a given execution plan on the simulator

Device up/down and duration variance are simulator/scenario concerns driven from
Python (a callback / the sim's fault API), not exposed on the CLI.

All real logic lives in the library (`ofplang.run.runner` / `ofplang.run.simulator`)
so the CLI cannot drift from it; this file only parses arguments, reports errors,
and maps outcomes to exit codes.

`run` first runs the workflow through `ofplang-validate` as a one-shot front door
(extension-tolerant) so a malformed workflow fails with clear diagnostics rather
than being silently mis-run; the runner library itself trusts its input, and the
per-tick replans never re-validate. Pass `--no-validate` to skip this (e.g. when
already validated upstream). A separate capability gate — always on — then rejects
valid v0 that uses a feature the runner does not support (generic processes,
`$import`) as a clean usage error. `replay` takes a plan, not a workflow, so it is
not front-door validated here.

Exit codes:
    0  success (the workflow / plan ran to completion)
    1  execution failed (an activity errored, or a replan is infeasible)
    2  usage / input error
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import yaml
from ofplang.validate import EXTENSION_TOLERANT
from ofplang.validate import validate as validate_workflow

from ofplang.run.runner import (
    ContractSyntaxError,
    RollingRunner,
    Runner,
    RunnerError,
    load_document,
    serialize_document,
)
from ofplang.run.simulator import SimulatorError

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ofp-run",
        description="Run ofplang v0 workflows / plans.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # `run` -- rolling-horizon: drive a workflow to completion, replanning as it goes.
    r = sub.add_parser("run", help="drive a workflow to completion (rolling-horizon)")
    r.add_argument("workflow", metavar="WORKFLOW", help="ofplang v0 workflow YAML")
    r.add_argument("--env", required=True, metavar="ENV", help="execution environment YAML (§5)")
    r.add_argument(
        "--boundary",
        metavar="DOC",
        help="run boundary document (§6.8 / value layer): a `boundary:` mapping with "
        "per-port {spot, view} descriptors for the workflow's entry inputs and final "
        "outputs. `spot` places a boundary Object; `view` supplies an input value "
        "(unsupplied entry inputs default)",
    )
    r.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="scheduler random seed (reproducible replans)",
    )
    r.add_argument(
        "--margin",
        type=int,
        default=0,
        metavar="M",
        help="running-task margin for replans",
    )
    r.add_argument(
        "--poll-interval",
        type=int,
        default=1,
        metavar="D",
        help="poll every D time units (fixed-interval, with completion-time estimation; default 1)",
    )
    r.add_argument(
        "-o",
        "--output",
        metavar="OUT",
        help="write the final status YAML here (default: stdout)",
    )
    r.add_argument(
        "--boundary-out",
        metavar="FILE",
        help="write the result boundary document here (YAML): the same schema as "
        "--boundary, with each produced output's `view` filled in; a run-local "
        "artifact, not part of the §6/§7 status document",
    )
    r.add_argument(
        "--no-validate",
        action="store_true",
        help="skip the one-shot ofplang-validate front-door check of the workflow "
        "(use when it was already validated upstream, e.g. by the `ofp` umbrella CLI)",
    )

    # `replay` -- replay a pre-made execution plan on the simulator (no replanning).
    p = sub.add_parser("replay", help="replay an execution plan on the simulator")
    p.add_argument("plan", metavar="PLAN", help="execution plan YAML (from ofp-schedule)")
    p.add_argument("--env", required=True, metavar="ENV", help="execution environment YAML (§5)")
    p.add_argument(
        "-o",
        "--output",
        metavar="OUT",
        help="write the resulting status YAML here (default: stdout)",
    )

    return parser


def _read_document(path, what: str) -> tuple[dict | None, int | None]:
    """Load a YAML document, returning (doc, None) or (None, EXIT_USAGE) on error."""
    try:
        return load_document(path), None
    except (OSError, yaml.YAMLError) as exc:
        print(f"ofp-run: cannot read {what} {str(path)!r}: {exc}", file=sys.stderr)
        return None, EXIT_USAGE


def _emit(status: dict, output) -> None:
    text = serialize_document(status)
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _front_door_validate(workflow_path: str) -> bool:
    """Validate a workflow as portable v0 before running (spec §2/§3, etc.).

    The runner trusts its input; this CLI front door runs the full
    ofplang-validate pass once so a malformed workflow fails with clear
    diagnostics instead of being silently mis-run. Returns True if the workflow
    is valid; otherwise prints each error and returns False. Uses
    extension-tolerant mode so `x-` extension keys are not rejected at the door.
    """
    result = validate_workflow(workflow_path, mode=EXTENSION_TOLERANT)
    if result.ok:
        return True
    for diag in result.diagnostics:
        if diag.file and diag.line:
            locator = f"{diag.file}:{diag.line}:{diag.col}"
        else:
            locator = diag.path or "<root>"
        detail = f"  {diag.path}" if diag.file and diag.path else ""
        message = f"  {diag.message}" if diag.message else ""
        print(f"{locator}: error {diag.code}{detail}{message}", file=sys.stderr)
    return False


def _import_key_present(obj) -> bool:
    """True if a `$import` key appears anywhere in the document (spec 3)."""
    if isinstance(obj, dict):
        return "$import" in obj or any(_import_key_present(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_import_key_present(v) for v in obj)
    return False


def _capability_gate(workflow_path: str) -> str | None:
    """Return a reason if the workflow uses a valid-v0 feature the runner does
    not support, so it is rejected cleanly instead of mis-run; else None.

    The runner handles a subset of valid v0. Generic processes (`generic_processes`)
    are not instantiated here, and `$import` is not resolved (the runner assumes an
    expanded document); either would otherwise surface as a confusing deep error.
    Always checked, independent of the `--no-validate` front door."""
    try:
        data = yaml.safe_load(Path(workflow_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None  # a parse problem is handled by the front door / runner
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


def _cmd_run(args) -> int:
    # Inputs must exist; a missing file is a usage error, not a failure.
    for label, path in (("workflow", args.workflow), ("environment", args.env)):
        if not Path(path).is_file():
            print(f"ofp-run: {label} not found: {path!r}", file=sys.stderr)
            return EXIT_USAGE

    # Front door: validate the workflow as portable v0 once, unless suppressed.
    # A malformed workflow never ran, so this is a usage/input error (EXIT_USAGE),
    # kept distinct from an execution failure (EXIT_FAILED). The runner library —
    # including the per-tick replans that call ofplang.schedule — is not invoked.
    if not args.no_validate and not _front_door_validate(args.workflow):
        return EXIT_USAGE

    # Capability gate (always on, even under --no-validate): reject valid-v0
    # features the runner cannot run, as a clean usage error rather than a deep
    # mis-run. A validated workflow can still use generics / $import, which the
    # runner does not support.
    unsupported = _capability_gate(args.workflow)
    if unsupported is not None:
        print(f"ofp-run: unsupported: {unsupported}", file=sys.stderr)
        return EXIT_USAGE

    # The run boundary (D28): the single run-facing I/O document (spot placement +
    # input views). Passed to the runner verbatim; it parses / validates it against
    # the workflow's contracts.
    boundary = None
    if args.boundary:
        boundary, err = _read_document(args.boundary, "boundary document")
        if err is not None:
            return err
        if not isinstance(boundary, dict):
            print(
                f"ofp-run: boundary document must be a mapping: {args.boundary!r}",
                file=sys.stderr,
            )
            return EXIT_USAGE

    try:
        runner = RollingRunner(
            args.workflow,
            args.env,
            boundary,
            running_task_margin=args.margin,
            random_seed=args.seed,
            poll_interval=args.poll_interval,
        )
        status = runner.run()
    except (yaml.YAMLError, ContractSyntaxError) as exc:
        # Malformed workflow / environment YAML or an unparsable contract
        # expression is an input error, not an execution failure -- the runner is
        # the untrusted boundary even though valid v0 input never hits this.
        print(f"ofp-run: invalid input: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (SimulatorError, RunnerError) as exc:
        print(f"ofp-run: execution failed: {exc}", file=sys.stderr)
        return EXIT_FAILED

    # The result boundary is a run-local artifact (D28): the same schema as the
    # supplied boundary with the produced output views filled in, written separately
    # so the §6/§7 status document stays value-free.
    if args.boundary_out:
        Path(args.boundary_out).write_text(
            serialize_document(runner.result_boundary), encoding="utf-8"
        )

    # An activity failure stops the run without raising: the status is still emitted
    # (it carries the failed / cancelled activities), but the run counts as failed.
    _emit(status, args.output)
    if runner.failed:
        # Report the failure reason (D36): its code and human-readable detail, from
        # the structured `runner.failure` (a contract violation, a script error, or a
        # generic activity failure). Falls back to a generic line if unset.
        failure = runner.failure
        if failure is not None:
            print(f"ofp-run: execution failed: {failure.kind}: {failure.detail}", file=sys.stderr)
        else:
            print("ofp-run: execution failed: an activity failed", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


def _cmd_replay(args) -> int:
    plan, err = _read_document(args.plan, "plan")
    if err is not None:
        return err
    assert plan is not None  # err is None => a document was loaded
    if not Path(args.env).is_file():
        print(f"ofp-run: environment not found: {args.env!r}", file=sys.stderr)
        return EXIT_USAGE

    try:
        runner = Runner(plan, args.env)
        status = runner.run()
    except (yaml.YAMLError, ContractSyntaxError) as exc:
        print(f"ofp-run: invalid input: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (SimulatorError, RunnerError) as exc:
        print(f"ofp-run: execution failed: {exc}", file=sys.stderr)
        return EXIT_FAILED

    _emit(status, args.output)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    # Emit UTF-8 to stdout regardless of the console's default encoding (e.g. a
    # cp932 Windows console), so piped output never hits an encode error.
    # AttributeError/ValueError when stdout is not a real TextIO (e.g. under capture).
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "replay":
        return _cmd_replay(args)
    return EXIT_USAGE  # pragma: no cover - argparse enforces a subcommand


if __name__ == "__main__":  # pragma: no cover - module entry
    raise SystemExit(main())
