"""ofplang.run -- runner for Object-flow Programming Language v0.

The runner consumes an execution plan (the artefact produced by
`ofplang.schedule`, spec §6) and drives its activities to completion against an
execution backend, emitting an execution status/document as it progresses.

Two layers are planned and will land incrementally:

* ``ofplang.run.simulator`` -- a simulated execution backend used to test the
  runner end to end without real hardware (built first).
* ``ofplang.run.runner`` -- the runner itself, which walks a plan and dispatches
  its activities to a backend (built on top of the simulator).

Nothing is implemented yet; this package is a scaffold. Public exports will be
added here as each layer lands.
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__: list[str] = []
