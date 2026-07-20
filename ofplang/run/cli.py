"""Command-line interface for ofplang.run.

Thin presentation layer over the library. Subcommands:

    ofp-run run <workflow> --env <env>
        [--interface <doc>] [--job <doc>] [--seed N] [--margin M] [--poll-interval D] [-o OUT]
        drive a workflow to completion by replanning (rolling-horizon)
    ofp-run replay <plan> --env <env> [-o OUT]
        replay a given execution plan on the simulator

Device up/down and duration variance are simulator/scenario concerns driven from
Python (a callback / the sim's fault API), not exposed on the CLI.

All real logic lives in the library (`ofplang.run.runner` / `ofplang.run.simulator`)
so the CLI cannot drift from it; this file only parses arguments, reports errors,
and maps outcomes to exit codes.

Exit codes:
    0  success (the workflow / plan ran to completion)
    1  execution failed (an activity errored, or a replan is infeasible)
    2  usage / input error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ofplang.run.runner import (
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
    parser = argparse.ArgumentParser(prog="ofp-run", description="Run ofplang v0 workflows / plans.")
    sub = parser.add_subparsers(dest="command", required=True)

    # `run` -- rolling-horizon: drive a workflow to completion, replanning as it goes.
    r = sub.add_parser("run", help="drive a workflow to completion (rolling-horizon)")
    r.add_argument("workflow", metavar="WORKFLOW", help="ofplang v0 workflow YAML")
    r.add_argument("--env", required=True, metavar="ENV", help="execution environment YAML (§5)")
    r.add_argument(
        "--interface",
        metavar="DOC",
        help="execution document carrying the interface boundary constraint (§6.8)",
    )
    r.add_argument(
        "--job",
        metavar="DOC",
        help="whole-workflow input values as {entry_port: view value} (YAML/JSON); "
        "unsupplied entry inputs default (§7 view defaults)",
    )
    r.add_argument("--seed", type=int, metavar="N", help="scheduler random seed (reproducible replans)")
    r.add_argument("--margin", type=int, default=0, metavar="M", help="running-task margin for replans")
    r.add_argument(
        "--poll-interval",
        type=int,
        default=1,
        metavar="D",
        help="poll every D time units (fixed-interval, with completion-time estimation; default 1)",
    )
    r.add_argument("-o", "--output", metavar="OUT", help="write the final status YAML here (default: stdout)")

    # `replay` -- replay a pre-made execution plan on the simulator (no replanning).
    p = sub.add_parser("replay", help="replay an execution plan on the simulator")
    p.add_argument("plan", metavar="PLAN", help="execution plan YAML (from ofp-schedule)")
    p.add_argument("--env", required=True, metavar="ENV", help="execution environment YAML (§5)")
    p.add_argument("-o", "--output", metavar="OUT", help="write the resulting status YAML here (default: stdout)")

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


def _cmd_run(args) -> int:
    # Inputs must exist; a missing file is a usage error, not a failure.
    for label, path in (("workflow", args.workflow), ("environment", args.env)):
        if not Path(path).is_file():
            print(f"ofp-run: {label} not found: {path!r}", file=sys.stderr)
            return EXIT_USAGE

    interface = None
    if args.interface:
        doc, err = _read_document(args.interface, "interface document")
        if err is not None:
            return err
        interface = doc.get("interface") if isinstance(doc, dict) else None

    job = None
    if args.job:
        doc, err = _read_document(args.job, "job document")
        if err is not None:
            return err
        if not isinstance(doc, dict):
            print(f"ofp-run: job document must be a mapping: {args.job!r}", file=sys.stderr)
            return EXIT_USAGE
        job = doc

    runner = RollingRunner(
        args.workflow,
        args.env,
        interface,
        job=job,
        running_task_margin=args.margin,
        random_seed=args.seed,
        poll_interval=args.poll_interval,
    )
    try:
        status = runner.run()
    except (SimulatorError, RunnerError) as exc:
        print(f"ofp-run: execution failed: {exc}", file=sys.stderr)
        return EXIT_FAILED

    # An activity failure stops the run without raising: the status is still emitted
    # (it carries the failed / cancelled activities), but the run counts as failed.
    _emit(status, args.output)
    if runner.failed:
        print("ofp-run: execution failed: an activity failed", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


def _cmd_replay(args) -> int:
    plan, err = _read_document(args.plan, "plan")
    if err is not None:
        return err
    if not Path(args.env).is_file():
        print(f"ofp-run: environment not found: {args.env!r}", file=sys.stderr)
        return EXIT_USAGE

    runner = Runner(plan, args.env)
    try:
        status = runner.run()
    except (SimulatorError, RunnerError) as exc:
        print(f"ofp-run: execution failed: {exc}", file=sys.stderr)
        return EXIT_FAILED

    _emit(status, args.output)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    # Emit UTF-8 to stdout regardless of the console's default encoding (e.g. a
    # cp932 Windows console), so piped output never hits an encode error.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - not a real TextIO (e.g. under capture)
        pass

    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "replay":
        return _cmd_replay(args)
    return EXIT_USAGE  # pragma: no cover - argparse enforces a subcommand


if __name__ == "__main__":  # pragma: no cover - module entry
    raise SystemExit(main())
