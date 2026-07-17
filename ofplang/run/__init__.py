"""ofplang.run -- runner for Object-flow Programming Language v0.

Drives ofplang v0 work to completion against an execution backend, emitting an
execution status/document (spec §6/§7) as it progresses. Two layers:

* ``ofplang.run.simulator`` -- a physical-only simulated backend that stands in
  for real hardware, so the runner can be driven end to end without a lab.
* ``ofplang.run.runner`` -- the runner. `RollingRunner` drives a workflow by
  replanning each tick via `ofplang.schedule` (re-routing on device faults,
  event-boundary or fixed-interval polling, optional duration variance);
  `Runner` replays a pre-made execution plan (§6) with no replanning.

The `ofp-run` CLI (`ofplang.run.cli`) exposes both: `run` (rolling-horizon) and
`replay`. Design rationale is recorded in dev-notes (D9-D23).
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__: list[str] = []
