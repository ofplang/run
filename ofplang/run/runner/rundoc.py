"""The run document: what a run of a *laboratory* is asked to do.

A run used to be a run of a workflow, so what it needed was said on the command
line: the workflow, its boundary, and — as a section of that boundary — what the
stocks held. Planning several workflows together (SPEC §6.11) makes a run one of a
laboratory, and there is no command line that comfortably says *three workflows,
each with its own boundary and its own release time*. So it is a document.

    jobs:
      - id: plate1
        workflow: plate.workflow.yaml
        boundary: plate1.boundary.yaml
        release: 0
      - id: plate2
        workflow: plate.workflow.yaml
        boundary: plate2.boundary.yaml
        release: 60
    inventories:
      levels:
        incubator: { medium: 4 }
    occupied:
      - { spot: freezer.slot3 }

The shape is deliberately the plan's own roster (§6.11) read as an *input*: `id` and
`release` mean here exactly what they mean there, and the two laboratory-wide
sections are the §6.10 / §6.12 ones under their own names. What the run document adds
is the two things a plan never carries — which workflow a job runs and where its
material is — because a plan is about *when*, and those are about *what*. (The
roster's `interface` is the spots half of the boundary, so it is derived from the
boundary named here rather than written twice.)

Relative paths resolve against the run document's own directory, so a directory of
run inputs can be moved or copied whole.

`inventories` is a run-level section here and stays a boundary section too: every
existing run says it there, and a single-job run has exactly one boundary to say it
in. Where a run has several, the boundary is the wrong place — the stock belongs to
the laboratory, not to any one job — which is why it can be said here, and why two
jobs disagreeing about it is an error rather than a merge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .job import JobRequest
from .loader import load_document
from .runner import RunnerError

JOB_KEYS = {"id", "workflow", "boundary", "release"}
DOC_KEYS = {"jobs", "inventories", "occupied"}
# `since` -- when the spot became occupied (§6.12) -- is accepted but rarely written:
# a run document says what the laboratory holds *before the run*, so it defaults to 0.
OCCUPIED_KEYS = {"spot", "since"}


@dataclass
class RunDocument:
    """A parsed run document: the jobs to run, and what the laboratory starts with."""

    jobs: list[JobRequest] = field(default_factory=list)
    inventories: dict | None = None
    occupied: list[dict] | None = None


def parse_run_document(doc: dict, base_dir: str | Path | None = None) -> RunDocument:
    """Read a run document into the jobs and laboratory state a run is built from.

    `base_dir` is what relative `workflow` / `boundary` paths resolve against — the
    directory the document was read from. None resolves against the working
    directory, which is what an in-memory caller with no file gets.

    Structural problems raise `RunnerError`: this is a hand-written input document
    and a typo in it should be met with the typo, not with a run that silently omits
    a job. An unknown key is refused for the same reason — a misspelled `realease`
    that is quietly ignored is a job that starts at the wrong time.
    """
    if not isinstance(doc, dict):
        raise RunnerError("run document must be a mapping")
    unknown = sorted(set(doc) - DOC_KEYS)
    if unknown:
        raise RunnerError(f"run document: unknown key(s) {unknown}")
    entries = doc.get("jobs")
    if not isinstance(entries, list) or not entries:
        raise RunnerError("run document: `jobs` must be a non-empty list")

    root = Path(base_dir) if base_dir is not None else Path()
    jobs = [_job(entry, index, root) for index, entry in enumerate(entries)]

    inventories = doc.get("inventories")
    if inventories is not None and not isinstance(inventories, dict):
        raise RunnerError("run document: `inventories` must be a mapping")
    occupied = doc.get("occupied")
    if occupied is not None:
        if not isinstance(occupied, list):
            raise RunnerError("run document: `occupied` must be a list")
        for index, entry in enumerate(occupied):
            where = f"run document: occupied[{index}]"
            if not isinstance(entry, dict):
                raise RunnerError(f"{where} must be a mapping")
            unknown_entry = sorted(set(entry) - OCCUPIED_KEYS)
            if unknown_entry:
                raise RunnerError(f"{where}: unknown key(s) {unknown_entry}")
            if not isinstance(entry.get("spot"), str) or not entry["spot"]:
                raise RunnerError(f"{where}: `spot` is required")
    return RunDocument(jobs=jobs, inventories=inventories, occupied=occupied)


def _job(entry, index: int, root: Path) -> JobRequest:
    where = f"run document: jobs[{index}]"
    if not isinstance(entry, dict):
        raise RunnerError(f"{where} must be a mapping")
    unknown = sorted(set(entry) - JOB_KEYS)
    if unknown:
        raise RunnerError(f"{where}: unknown key(s) {unknown}")

    job_id = entry.get("id")
    if not isinstance(job_id, str) or not job_id:
        # The id is how every activity, every commit and every promise is keyed back
        # to this job, so a run cannot be asked for one that has none. It is also why
        # it is required rather than defaulted to the position: a job named by where
        # it happened to sit in a list would be renamed by inserting another.
        raise RunnerError(f"{where}: `id` is required and must be a non-empty string")

    workflow = entry.get("workflow")
    if isinstance(workflow, str):
        workflow_doc = _read(root / workflow, f"{where}: workflow")
    elif isinstance(workflow, dict):
        workflow_doc = workflow
    else:
        raise RunnerError(f"{where}: `workflow` must be a path or a mapping")

    boundary = entry.get("boundary")
    if boundary is None:
        boundary_doc = None
    elif isinstance(boundary, str):
        boundary_doc = _read(root / boundary, f"{where}: boundary")
    elif isinstance(boundary, dict):
        boundary_doc = boundary
    else:
        raise RunnerError(f"{where}: `boundary` must be a path or a mapping")

    release = entry.get("release", 0)
    if not isinstance(release, int) or isinstance(release, bool) or release < 0:
        raise RunnerError(f"{where}: `release` must be a non-negative integer")

    return JobRequest(
        id=job_id, workflow=workflow_doc, boundary=boundary_doc, release=release
    )


def _read(path: Path, what: str) -> dict:
    try:
        doc = load_document(path)
    except OSError as exc:
        raise RunnerError(f"{what}: cannot read {str(path)!r}: {exc}") from exc
    if not isinstance(doc, dict):
        raise RunnerError(f"{what}: {str(path)!r} is not a mapping")
    return doc
