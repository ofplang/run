"""The runner: drive an execution plan to completion (spec §6/§7).

Built on top of the simulator. This layer reads an execution plan (§6), replays
its activities against an execution backend (the simulator, or later real
hardware), and emits an execution status/document reflecting the run.

Milestone 2a (implemented): replay a given plan with no replanning -- see
`runner.Runner`. Milestone 2b (planned): the rolling-horizon replanning loop
(dev-notes design.md D19).
"""

from __future__ import annotations

from .loader import load_document, serialize_document
from .runner import Runner, RunnerError

__all__ = [
    "Runner",
    "RunnerError",
    "load_document",
    "serialize_document",
]
