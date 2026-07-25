"""The runner: drive an execution plan to completion (spec §6/§7).

Section references: bare `§N` cite the execution-plan spec
(`ofplang-schedule/docs/SPECIFICATIONS.md`), which is this package's primary
input. The v0 language spec (`ofplang-spec/SPECIFICATION.md`) is cited as
`v0 §N`, because the two documents number their §3-§9 differently (e.g. plan
`§6` is the execution document, language `v0 §6` is phases).

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
