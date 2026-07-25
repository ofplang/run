"""Simulated execution backend for the runner (spec §6/§7; dev-notes D10-D27).

Section references: bare `§N` cite the execution-plan spec
(`ofplang-schedule/docs/SPECIFICATIONS.md`); the v0 language spec
(`ofplang-spec/SPECIFICATION.md`) is cited as `v0 §N` where the two collide.

This layer stands in for real hardware so the runner can be exercised end to end:
it accepts the same physical dispatch calls a real backend would, advances a
virtual clock, and reports operation outcomes, letting tests drive a full plan
deterministically. Its physical core knows only the physical world (devices,
spots, transporters, opaque timed operations) and never interprets workflows or
plans (D10); on top of it sits the value seam (D26/D27), where a device model
turns input values into output view values.

See `core.Simulator` for the contract. It models correct physical behaviour plus
timed device up/down (a down device blocks new processing only, D21) and injected
operation failure per capability (a failing (process, mode) / (transporter, route)
ends `failed` with no material effect, D25). Duration variance is injected
externally via a dispatch `duration` (D13). Output view values are produced by an
injected device model or the built-in `script_device_model` (D31), which runs a
`python_script_processes` script (v0 §22) and otherwise falls back to
`default_device_model` (D27); non-script outputs are typed but still dummy.
"""

from __future__ import annotations

from .core import Event, Simulator, default_device_model
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
    DeviceDown,
    MissingObject,
    RelayNotSupported,
    ResourceBusy,
    SimulatorError,
    SpotConflict,
    UnknownReference,
)
from .script import DeviceComputationError, run_python_script, script_device_model

__all__ = [
    "Simulator",
    "Event",
    "default_device_model",
    "script_device_model",
    "run_python_script",
    "DeviceComputationError",
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
    "DeviceDown",
    "ClockError",
]
