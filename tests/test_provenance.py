"""What the runner tells a backend about where an activity came from.

Provenance is an optional extension of the `Backend` protocol, asked for **one
keyword at a time**: a backend is offered `node` (which workflow node this is) and
`job` (which job of a joint run it belongs to, §6.11) only if its signature accepts
that keyword. A backend from before an extension is driven exactly as it was.

The job half exists because the node half is not enough. Two jobs of one workflow
render the *same* node paths and can move between the same pair of spots, so a
backend that keeps a record or mints identities from provenance -- labcode's `_id`
and its trace -- would give one job's plate the other's name.

🔴 And it is offered only where there *is* a job: a single-workflow run calls its one
job by the empty string internally, and passing that would tell a backend about a
distinction the run does not have -- changing what a provenance-keyed backend produces
for every run that has always been one workflow.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import JobRequest, RollingRunner, load_document  # noqa: E402
from ofplang.run.simulator import VirtualTimeSimulator  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = FIXTURES / "simple.workflow.yaml"
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")


class Provenance(VirtualTimeSimulator):
    """A backend that declares the whole provenance and records what it was handed."""

    def __init__(self, environment):
        super().__init__(environment)
        self.processing: list[tuple] = []  # (process, node, job)
        self.transports: list[tuple] = []  # (from_spot, to_spot, job)

    def dispatch_processing(self, process, mode, duration=None, output_schema=None,
                            inputs=None, definition=None, node=None, job=None) -> str:
        self.processing.append((process, tuple(node) if node is not None else None, job))
        return super().dispatch_processing(
            process, mode, duration=duration, output_schema=output_schema,
            inputs=inputs, definition=definition, node=node, job=job,
        )

    def dispatch_transport(self, transporter, from_spot, to_spot,
                           duration=None, view=None, job=None) -> str:
        self.transports.append((from_spot, to_spot, job))
        return super().dispatch_transport(
            transporter, from_spot, to_spot, duration=duration, view=view, job=job,
        )


class NodeOnly(VirtualTimeSimulator):
    """A backend from before the job half existed: it asks for the node and nothing
    else, and must go on being driven exactly as it was."""

    def __init__(self, environment):
        super().__init__(environment)
        self.processing: list[tuple] = []

    def dispatch_processing(self, process, mode, duration=None, output_schema=None,
                            inputs=None, definition=None, node=None) -> str:
        self.processing.append((process, tuple(node) if node is not None else None))
        return super().dispatch_processing(
            process, mode, duration=duration, output_schema=output_schema,
            inputs=inputs, definition=definition, node=node,
        )


class Neither(VirtualTimeSimulator):
    """A backend that asks for no provenance at all -- the minimal `Backend`."""

    def __init__(self, environment):
        super().__init__(environment)
        self.dispatched = 0

    def dispatch_processing(self, process, mode, duration=None, output_schema=None,
                            inputs=None, definition=None) -> str:
        self.dispatched += 1
        return super().dispatch_processing(
            process, mode, duration=duration, output_schema=output_schema,
            inputs=inputs, definition=definition,
        )


def _capture(cls):
    """A `backend_factory` for `cls`, plus the instance it built."""
    built: dict = {}

    def factory(environment):
        built["backend"] = cls(environment)
        return built["backend"]

    return factory, built


def _jobs(*ids: str) -> list[JobRequest]:
    doc = load_document(SIMPLE_WF)
    return [JobRequest(id=job_id, workflow=doc) for job_id in ids]


def test_a_single_workflow_run_names_no_job():
    """🔴 The invariant that keeps every existing run identical: one workflow is not a
    laboratory, so nothing is said about jobs."""
    factory, built = _capture(Provenance)
    runner = RollingRunner(str(SIMPLE_WF), SIMPLE_ENV, backend_factory=factory, random_seed=0)
    runner.run()

    backend = built["backend"]
    assert backend.processing, "the run should have dispatched something"
    assert {job for _p, _n, job in backend.processing} == {None}
    assert {job for _f, _t, job in backend.transports} == {None}
    # The node half is still offered, as it always was.
    assert all(node is not None for _p, node, _j in backend.processing)


def test_a_joint_run_names_the_job_on_every_dispatch():
    factory, built = _capture(Provenance)
    runner = RollingRunner(_jobs("job1", "job2"), SIMPLE_ENV, backend_factory=factory,
                           random_seed=0)
    runner.run()
    assert not runner.failed

    backend = built["backend"]
    assert {job for _p, _n, job in backend.processing} == {"job1", "job2"}
    # 🔴 The point of it: the same node path is dispatched for both jobs, and only the
    # job tells them apart. Without it a backend keyed on provenance gives one job's
    # material the other's identity.
    by_node: dict = {}
    for _process, node, job in backend.processing:
        by_node.setdefault(node, set()).add(job)
    assert any(jobs == {"job1", "job2"} for jobs in by_node.values())


def test_transports_carry_the_job_too():
    """Two jobs of one workflow move between the same pair of spots, so the move alone
    does not say whose plate it was."""
    factory, built = _capture(Provenance)
    runner = RollingRunner(_jobs("job1", "job2"), SIMPLE_ENV, backend_factory=factory,
                           random_seed=0)
    runner.run()

    backend = built["backend"]
    assert backend.transports
    assert {job for _f, _t, job in backend.transports} == {"job1", "job2"}
    by_route: dict = {}
    for from_spot, to_spot, job in backend.transports:
        by_route.setdefault((from_spot, to_spot), set()).add(job)
    assert any(jobs == {"job1", "job2"} for jobs in by_route.values())


def test_a_backend_that_asks_only_for_the_node_gets_only_the_node():
    """🔴 One keyword at a time. A backend that wanted the node before the job half
    existed is called with the node and nothing else -- handing it an argument it never
    declared would be a `TypeError` rather than an extension."""
    factory, built = _capture(NodeOnly)
    runner = RollingRunner(_jobs("job1", "job2"), SIMPLE_ENV, backend_factory=factory,
                           random_seed=0)
    runner.run()
    assert not runner.failed

    backend = built["backend"]
    assert backend.processing
    assert all(node is not None for _p, node in backend.processing)


def test_a_backend_that_asks_for_neither_is_driven_unchanged():
    factory, built = _capture(Neither)
    runner = RollingRunner(_jobs("job1", "job2"), SIMPLE_ENV, backend_factory=factory,
                           random_seed=0)
    runner.run()
    assert not runner.failed
    assert built["backend"].dispatched > 0
