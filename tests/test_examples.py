"""The committed example outputs are what the examples produce today.

`examples/README.md` says re-running an example reproduces its committed files byte
for byte, and the files are read as a record of what the runner does. Nothing held
the two together: `shared_refill.txt` and `stopped_job.txt` drifted for weeks after
the scheduler started breaking ties between equally good schedules differently. This
runs every `render_*.py` in a copy of `examples/` and compares what it writes with
what is committed, so the next drift is a failing test rather than a stale record.

Line endings are normalised: a checkout on Windows has CRLF where Linux has LF, and
the content is what is being pinned. The examples solve with a fixed seed, which
pins CP-SAT to one worker, and were checked to agree between Windows and Linux
(Python 3.10 and 3.12).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(EXAMPLES.glob("render_*.py"))


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> Path:
    """Every example re-run in a copy, its outputs in `<copy>/outputs`."""
    copy = tmp_path_factory.mktemp("examples") / "examples"
    shutil.copytree(EXAMPLES, copy, ignore=shutil.ignore_patterns("outputs", "__pycache__"))
    env = dict(os.environ, PYTHONHASHSEED="0")
    for script in SCRIPTS:
        done = subprocess.run(
            [sys.executable, str(copy / script.name)],
            cwd=copy, env=env, capture_output=True, text=True,
        )
        assert done.returncode == 0, f"{script.name}:\n{done.stdout}\n{done.stderr}"
    return copy / "outputs"


@pytest.mark.parametrize(
    "committed", sorted((EXAMPLES / "outputs").glob("*.*")), ids=lambda p: p.name
)
def test_a_committed_output_is_what_its_example_writes(regenerated, committed):
    fresh = regenerated / committed.name
    assert fresh.is_file(), f"no example writes {committed.name} any more"
    assert fresh.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8"), (
        f"{committed.name} is not what its example writes now; re-run it "
        f"(PYTHONHASHSEED=0 python examples/render_<name>.py) and commit the result"
    )


def test_every_output_an_example_writes_is_committed(regenerated):
    committed = {p.name for p in (EXAMPLES / "outputs").glob("*.*")}
    assert {p.name for p in regenerated.glob("*.*")} <= committed
