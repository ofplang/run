"""The environment can arrive as a document, and a replan writes nothing (R6 / S2).

Two halves of the same seam. The runner used to take the environment only as a path,
so a caller that had already read it -- a dialect front door inspecting `x-` keys --
made it be read twice; and the scheduler used to take only paths, so every replan
wrote the environment and the status to temporary files and deleted them again.
Both now pass documents straight through.

What must not change is what the run produces: an environment handed over as a
document has to schedule exactly as the same environment read from its file.
Requires `ofplang-schedule` >= 0.1.6 (the floor in pyproject), which is where the
scheduler takes an environment and an execution document in memory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.app import run_workflow  # noqa: E402
from ofplang.run.runner import RollingRunner  # noqa: E402
from ofplang.run.runner.schedule_client import replan  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")


def _env_document() -> dict:
    return yaml.safe_load(Path(SIMPLE_ENV).read_text(encoding="utf-8"))


def test_an_environment_document_runs_exactly_as_its_file():
    from_path = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0).run()
    from_document = RollingRunner(SIMPLE_WF, _env_document(), random_seed=0).run()
    assert from_document == from_path


def test_the_front_door_takes_an_environment_document_too():
    result = run_workflow(SIMPLE_WF, _env_document(), validate=False)
    assert not result.failed
    assert result.status["now"] == 5


def test_the_callers_environment_document_is_left_alone():
    """Mode ids are pinned in a copy before anything else happens (D21), so the
    document the caller still holds is the one it passed."""
    document = _env_document()
    before = yaml.safe_dump(document, sort_keys=False)
    RollingRunner(SIMPLE_WF, document, random_seed=0).run()
    assert yaml.safe_dump(document, sort_keys=False) == before


def test_provenance_names_the_file_when_there_is_one():
    """`environment_path` is what the plan's `meta.environment` records: the file the
    environment came from, or None when it was handed over as a document (the
    scheduler writes `<in-memory>` for that)."""
    assert RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0).environment_path == SIMPLE_ENV
    assert RollingRunner(SIMPLE_WF, _env_document(), random_seed=0).environment_path is None


def test_a_replan_records_the_environment_file_it_came_from():
    """Even though the environment handed to the scheduler is a normalized (and, when
    machines are down, reduced) dict rather than the file's own text."""
    status = {"time": {"unit": "second"}, "now": 0, "activities": []}
    report = replan(
        yaml.safe_load(Path(SIMPLE_WF).read_text(encoding="utf-8")),
        _env_document(),
        status,
        random_seed=0,
        environment_source=SIMPLE_ENV,
    )
    assert report.ok
    assert report.plan["meta"]["environment"] == SIMPLE_ENV
    assert report.plan["meta"]["status"] == "<in-memory>"


def test_no_temporary_file_is_written_during_a_run(monkeypatch):
    """The whole point of the switch: a replan used to write the environment and the
    status out and read them back. Nothing in the run may reach for a temp file."""

    def refuse(*args, **kwargs):
        raise AssertionError("a run must not create temporary files")

    monkeypatch.setattr(tempfile, "mkstemp", refuse)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)
    monkeypatch.setattr(tempfile, "TemporaryFile", refuse)

    status = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0).run()
    assert status["now"] == 5
