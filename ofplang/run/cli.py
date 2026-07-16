"""Command-line interface for ofplang.run.

Thin presentation layer over the library (to be). Planned subcommand:

    ofp-run run <plan> [...]      # drive an execution plan to completion

All real logic will live in the library (`ofplang.run.runner` /
`ofplang.run.simulator`) so the CLI cannot drift from it. This file is currently
a scaffold: the parser shape is in place but the command handler is a stub.

Exit codes:
    0  success (a plan ran to completion)
    1  execution failed (an activity errored, or the plan is infeasible)
    2  usage / input error
"""

from __future__ import annotations

import argparse
import sys

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ofp-run", description="Run ofplang v0 execution plans.")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="drive an execution plan to completion")
    r.add_argument("plan", metavar="PLAN", help="execution plan YAML (from ofp-schedule)")

    return parser


def _cmd_run(args) -> int:
    # Placeholder: the runner is not implemented yet. Kept as a stub so the CLI
    # shape and exit-code contract are exercised while the library is built out.
    print("ofp-run: 'run' is not implemented yet", file=sys.stderr)
    return EXIT_USAGE


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
