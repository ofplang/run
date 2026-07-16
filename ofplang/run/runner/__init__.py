"""The runner: drive an execution plan to completion (spec §6/§7).

Planned second, on top of the simulator. This layer reads an execution plan,
walks its activities in dependency/time order, dispatches each to an execution
backend (the simulator, or later real hardware), and emits an execution
status/document reflecting progress.

Not implemented yet -- this module is a placeholder marking where the runner
will live.
"""

from __future__ import annotations

__all__: list[str] = []
