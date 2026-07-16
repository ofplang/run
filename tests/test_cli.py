"""Smoke tests for the ofp-run CLI scaffold.

These pin the CLI's shape and exit-code contract while the library is built out;
they are intentionally light and will grow as `run` gains real behaviour.
"""

from __future__ import annotations

import pytest

from ofplang.run.cli import EXIT_USAGE, main


def test_help_exits_zero(capsys):
    # `--help` is handled by argparse and exits with code 0.
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "ofp-run" in capsys.readouterr().out


def test_missing_subcommand_is_usage_error():
    # A subcommand is required; omitting it is a usage error (argparse -> exit 2).
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == EXIT_USAGE


def test_run_is_stubbed(capsys):
    # The `run` handler is a placeholder for now; it reports "not implemented".
    assert main(["run", "plan.yaml"]) == EXIT_USAGE
    assert "not implemented" in capsys.readouterr().err
