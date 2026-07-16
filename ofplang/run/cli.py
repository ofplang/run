"""Command-line interface for ofplang.run.

Thin presentation layer over the library. Subcommand:

    ofp-run run <plan> --env <env> [-o OUT]   # replay a plan on the simulator

All real logic lives in the library (`ofplang.run.runner` / `ofplang.run.simulator`)
so the CLI cannot drift from it; this file only parses arguments, reports errors,
and maps outcomes to exit codes.

Exit codes:
    0  success (a plan ran to completion)
    1  execution failed (an activity errored, or the plan is infeasible)
    2  usage / input error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ofplang.run.runner import Runner, RunnerError, load_document, serialize_document
from ofplang.run.simulator import SimulatorError

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ofp-run", description="Run ofplang v0 execution plans.")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="replay an execution plan on the simulator")
    r.add_argument("plan", metavar="PLAN", help="execution plan YAML (from ofp-schedule)")
    r.add_argument("--env", required=True, metavar="ENV", help="execution environment YAML (§5)")
    r.add_argument(
        "-o", "--output", metavar="OUT", help="write the resulting status YAML here (default: stdout)"
    )

    return parser


def _cmd_run(args) -> int:
    # Read the inputs. A missing or malformed plan / environment is an input
    # (usage) error, not an execution failure.
    try:
        plan = load_document(args.plan)
    except (OSError, yaml.YAMLError) as exc:
        print(f"ofp-run: cannot read plan {args.plan!r}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        runner = Runner(plan, args.env)  # constructing the runner loads the environment
    except (OSError, yaml.YAMLError) as exc:
        print(f"ofp-run: cannot read environment {args.env!r}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    # Drive the plan. A backend rejection (an inconsistent plan) or an activity
    # that never completes is an execution failure.
    try:
        status = runner.run()
    except (SimulatorError, RunnerError) as exc:
        print(f"ofp-run: execution failed: {exc}", file=sys.stderr)
        return EXIT_FAILED

    text = serialize_document(status)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
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
    return EXIT_USAGE  # pragma: no cover - argparse enforces a subcommand


if __name__ == "__main__":  # pragma: no cover - module entry
    raise SystemExit(main())
