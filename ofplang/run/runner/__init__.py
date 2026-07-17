"""The runner: drive an execution plan to completion (spec §6/§7).

Built on top of the simulator. Two entry points:

* `Runner` (milestone 2a) -- replay a given execution plan (§6) with no
  replanning; see `runner.Runner`.
* `RollingRunner` (rolling-horizon) -- drive a workflow to completion by
  replanning each tick via `ofplang.schedule`, with device-down re-routing,
  event-boundary or fixed-interval polling, and optional duration variance
  (dev-notes design.md D9/D20-D23); see `rolling.RollingRunner`.
"""

from __future__ import annotations

from .loader import load_document, serialize_document
from .rolling import RollingRunner
from .runner import Runner, RunnerError

__all__ = [
    "Runner",
    "RollingRunner",
    "RunnerError",
    "load_document",
    "serialize_document",
]
