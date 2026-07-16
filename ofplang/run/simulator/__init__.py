"""Simulated execution backend for the runner (spec §6/§7).

Planned first. This layer stands in for real hardware so the runner can be
exercised end to end: it accepts the same dispatch calls a real backend would,
advances a virtual clock, and reports activity outcomes, letting tests drive a
full plan deterministically.

Not implemented yet -- this module is a placeholder marking where the simulator
will live.
"""

from __future__ import annotations

__all__: list[str] = []
