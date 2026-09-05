"""Command-line interface for ofplang.run.

Thin presentation layer over the library. Subcommands:

    ofp-run run <workflow> --env <env>
        [--boundary <doc>] [--boundary-out FILE] [--observation-out FILE]
        [--seed N] [--margin M] [--poll-interval D] [--max-ticks N] [-o OUT]
        drive a workflow to completion by replanning (rolling-horizon)
    ofp-run run --jobs <run doc> --env <env> [...]
        the same, for several workflows run together (SPEC §6.11): the run
        document names each job, its workflow and boundary, and when it may start
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
from dataclasses import replace
from pathlib import Path

import yaml

from ofplang.run.app import FrontDoorResult, front_door_check, run_workflow
from ofplang.run.runner import (
    DEFAULT_MAX_TICKS,
    ContractSyntaxError,
    Runner,
    RunnerError,
    load_document,
    parse_run_document,
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
    r.add_argument(
        "workflow",
        metavar="WORKFLOW",
        nargs="?",
        help="ofplang v0 workflow YAML (omit when --jobs names a run document)",
    )
    r.add_argument(
        "--jobs",
        metavar="RUNDOC",
        help="run document (YAML): a `jobs:` list naming each job's id, workflow, "
        "boundary and release time, plus the laboratory's own `inventories` (§6.10) "
        "and `occupied` spots (§6.12). The jobs are planned TOGETHER (§6.11), so they "
        "share the machines and draw on the same stocks. Mutually exclusive with a "
        "WORKFLOW argument and with --boundary, which each job carries its own of",
    )
    r.add_argument("--env", required=True, metavar="ENV", help="execution environment YAML (§5)")
    r.add_argument(
        "--boundary",
        metavar="DOC",
        help="run boundary document (§6.8 / value layer): a `boundary:` mapping with "
        "per-port {spot, view} descriptors for the workflow's entry inputs and final "
        "outputs. `spot` places a boundary Object; `view` supplies an input value "
        "(unsupplied entry inputs default). An `inventories: {levels: ...}` section "
        "(§6.10) says what each device-local stock holds at the START of the run; "
        "required when some mode consumes",
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
        "--max-ticks",
        type=int,
        default=DEFAULT_MAX_TICKS,
        metavar="N",
        help=(
            "give up after N ticks as non-terminating, or 0 for no limit "
            f"(default {DEFAULT_MAX_TICKS}); one tick is one poll interval, so this also "
            "caps the makespan a run can reach"
        ),
    )
    r.add_argument(
        "--ignore-resources",
        action="store_true",
        help="switch the consumable model off (§4.7.3): the environment's resource "
        "declarations are shape-checked but nothing is applied, so a lab that declares "
        "stocks runs without the boundary saying what they started with",
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
        "artifact, not part of the §6/§7 status document. `inventories` is not "
        "echoed -- it names the stock this run STARTED with, so feeding it back "
        "would give the next run stock this one spent; feed back the status instead",
    )
    r.add_argument(
        "--observation-out",
        metavar="FILE",
        help="stream the observation document here (YAML multi-document): completed "
        "activities' concrete input/output view values, keyed by provenance and "
        "appended as each activity finishes; a run-local artifact, not part of the "
        "§6/§7 status document",
    )
    r.add_argument(
        "--on-job-failure",
        choices=("continue", "stop"),
        default="continue",
        metavar="POLICY",
        help="what one job's failure does to the rest of a --jobs run (§6.11): "
        "`continue` (default) stops that job alone and lets the others finish -- which "
        "is why they were planned together -- while `stop` stops the whole run. A "
        "single workflow is a single job, so this makes no difference to it",
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


def _write(text: str, output, what: str) -> int | None:
    """Write `text` to the `output` path (or stdout when unset), returning EXIT_USAGE
    if the file could not be written.

    An unwritable output path is an input error like an unreadable one
    (`_read_document`), not an execution failure -- but it is discovered after the run
    has already done its work, so the caller reports it, still attempts the other
    outputs, and lets the run's own outcome win the exit code when the run failed.

    `labcode`'s `lc run` carries a copy of this (the two CLIs are near-duplicates by
    intent, so labcode does not depend on the `ofplang` umbrella): change one, change
    the other.
    """
    if not output:
        sys.stdout.write(text)
        return None
    try:
        Path(output).write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"ofp-run: cannot write {what} {str(output)!r}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return None


def _emit(status: dict, output) -> int | None:
    return _write(serialize_document(status), output, "status")


def _print_front_door(fd: FrontDoorResult) -> None:
    """Print a failed front-door check to stderr: each validate diagnostic (in the
    `file:line:col: error code path message` form) and, if present, the
    capability-gate reason."""
    for diag in fd.diagnostics:
        if diag.file and diag.line:
            locator = f"{diag.file}:{diag.line}:{diag.col}"
        else:
            locator = diag.path or "<root>"
        detail = f"  {diag.path}" if diag.file and diag.path else ""
        message = f"  {diag.message}" if diag.message else ""
        print(f"{locator}: error {diag.code}{detail}{message}", file=sys.stderr)
    if fd.unsupported is not None:
        print(f"ofp-run: unsupported: {fd.unsupported}", file=sys.stderr)


def _cmd_run(args) -> int:
    # Exactly one of the two ways of saying what to run. Both is a contradiction and
    # neither is nothing to run, so each is refused rather than resolved by precedence.
    if bool(args.workflow) == bool(args.jobs):
        print(
            "ofp-run: give either a WORKFLOW or --jobs RUNDOC, not both",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.jobs and args.boundary:
        # Each job in a run document carries its own boundary -- that is what makes
        # them different runs of the same workflow -- so a run-level one has no job to
        # belong to.
        print(
            "ofp-run: --boundary applies to a single workflow; with --jobs each job "
            "carries its own",
            file=sys.stderr,
        )
        return EXIT_USAGE

    # Inputs must exist; a missing file is a usage error, not a failure.
    for label, path in (
        ("workflow", args.workflow),
        ("run document", args.jobs),
        ("environment", args.env),
    ):
        if path is not None and not Path(path).is_file():
            print(f"ofp-run: {label} not found: {path!r}", file=sys.stderr)
            return EXIT_USAGE

    run_doc = None
    if args.jobs:
        doc, err = _read_document(args.jobs, "run document")
        if err is not None:
            return err
        assert doc is not None  # err is None => a document was loaded
        try:
            run_doc = parse_run_document(doc, Path(args.jobs).parent)
        except RunnerError as exc:
            print(f"ofp-run: {exc}", file=sys.stderr)
            return EXIT_USAGE

    # Front door (shared with any CLI, `ofplang.run.app`): the full ofplang-validate
    # pass (skipped under --no-validate) plus the always-on capability gate. A
    # malformed / unsupported workflow never ran, so it is a usage error (EXIT_USAGE),
    # distinct from an execution failure; the runner library is not invoked.
    #
    # A run document's workflows go through it one at a time, and the failing job is
    # named: several jobs commonly run the same workflow, and a diagnostic that does
    # not say which job it came from sends the reader to the wrong file.
    if run_doc is not None:
        requests = []
        for request in run_doc.jobs:
            fd = front_door_check(request.workflow, validate=not args.no_validate)
            if not fd.ok:
                print(f"ofp-run: job {request.id!r}:", file=sys.stderr)
                _print_front_door(fd)
                return EXIT_USAGE
            assert fd.document is not None  # ok => load + expand succeeded
            requests.append(replace(request, workflow=fd.document))
        target: object = requests
    else:
        fd = front_door_check(args.workflow, validate=not args.no_validate)
        if not fd.ok:
            _print_front_door(fd)
            return EXIT_USAGE
        target = fd.document

    # `--max-ticks 0` is the way to say "no limit" (the library spells that None); a
    # negative count is not a limit at all, so it is an input error rather than something
    # to interpret.
    if args.max_ticks < 0:
        print(f"ofp-run: --max-ticks must not be negative: {args.max_ticks}", file=sys.stderr)
        return EXIT_USAGE
    max_ticks = args.max_ticks or None

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
        # Validation + `$import` expansion already happened at the front door above,
        # so run trusting on the expanded document (not a re-read of the raw file).
        result = run_workflow(
            target,
            args.env,
            boundary,
            running_task_margin=args.margin,
            random_seed=args.seed,
            poll_interval=args.poll_interval,
            validate=False,
            observation_out=args.observation_out,
            max_ticks=max_ticks,
            ignore_resources=args.ignore_resources,
            inventories=run_doc.inventories if run_doc else None,
            occupied=run_doc.occupied if run_doc else None,
            on_job_failure=args.on_job_failure,
        )
    except (yaml.YAMLError, ContractSyntaxError) as exc:
        # Malformed workflow / environment YAML or an unparsable contract
        # expression is an input error, not an execution failure -- the runner is
        # the untrusted boundary even though valid v0 input never hits this.
        print(f"ofp-run: invalid input: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (SimulatorError, RunnerError) as exc:
        print(f"ofp-run: execution failed: {exc}", file=sys.stderr)
        return EXIT_FAILED

    # What the scheduler warned about, once per distinct code (a replan repeats the
    # same warnings every tick). These do not fail a run -- a deprecated section still
    # works, a switched-off model still schedules -- but a run that never says so is
    # how a deprecation is discovered by its removal.
    for diag in result.scheduler_warnings:
        where = f" ({diag.path})" if getattr(diag, "path", None) else ""
        print(f"ofp-run: scheduler: {diag.code}{where}: {diag.message}", file=sys.stderr)

    # The result boundary is a run-local artifact (D28): the same schema as the
    # supplied boundary with the produced output views filled in, written separately
    # so the §6/§7 status document stays value-free.
    write_err = None
    if args.boundary_out:
        write_err = _write(
            serialize_document(result.result_boundary), args.boundary_out, "result boundary"
        )

    # An activity failure stops the run without raising: the status is still emitted
    # (it carries the failed / cancelled activities), but the run counts as failed.
    write_err = _emit(result.status, args.output) or write_err
    if result.failed:
        # Report the failure reason (D36): its code and human-readable detail, from
        # the structured `failure` (a contract violation, a script error, or a
        # generic activity failure). Falls back to a generic line if unset.
        #
        # A run of named jobs reports one line per job that stopped, because they
        # stopped for unrelated reasons and naming only the first would hide the rest.
        # A failure belonging to no job -- an unplannable replan, a refill that failed
        # -- leaves that list empty and falls through to the run-level line below,
        # which is also the only line a single workflow ever prints.
        if result.job_failures:
            for job_id, failure in result.job_failures:
                print(
                    f"ofp-run: job {job_id!r} failed: {failure.kind}: {failure.detail}",
                    file=sys.stderr,
                )
        elif result.failure is not None:
            failure = result.failure
            print(f"ofp-run: execution failed: {failure.kind}: {failure.detail}", file=sys.stderr)
        else:
            print("ofp-run: execution failed: an activity failed", file=sys.stderr)
        return EXIT_FAILED
    return write_err or EXIT_OK


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

    return _emit(status, args.output) or EXIT_OK


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
