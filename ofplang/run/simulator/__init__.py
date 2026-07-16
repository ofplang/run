"""Simulated execution backend for the runner (spec §6/§7; dev-notes D10-D18).

This layer stands in for real hardware so the runner can be exercised end to end:
it accepts the same physical dispatch calls a real backend would, advances a
virtual clock, and reports operation outcomes, letting tests drive a full plan
deterministically. It knows only the physical world (devices, spots,
transporters, opaque timed operations) -- never workflows or plans (D10).

See `core.Simulator` for the contract. This first cut models correct behaviour
only (no variance, no device faults; D13/D17).
"""

from __future__ import annotations

from .core import Event, Simulator
from .environment import (
    Environment,
    Mode,
    Process,
    device_of,
    environment_from_dict,
    load_environment,
)
from .errors import (
    ClockError,
    MissingObject,
    RelayNotSupported,
    ResourceBusy,
    SimulatorError,
    SpotConflict,
    UnknownReference,
)

__all__ = [
    "Simulator",
    "Event",
    "Environment",
    "Mode",
    "Process",
    "environment_from_dict",
    "load_environment",
    "device_of",
    "SimulatorError",
    "UnknownReference",
    "ResourceBusy",
    "SpotConflict",
    "MissingObject",
    "RelayNotSupported",
    "ClockError",
]
