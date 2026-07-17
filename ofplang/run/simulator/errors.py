"""Error types raised by the simulator.

The simulator acts as a *validating oracle* (dev-notes design.md D16): besides
the resource-conflict errors D15 requires, it also checks the physical
preconditions of every dispatch (input spots hold material, output / destination
spots are free). A valid plan never trips these; they exist to catch a runner
that drives an inconsistent plan. Each failure mode gets its own exception so
tests (and the runner) can distinguish them.
"""

from __future__ import annotations


class SimulatorError(Exception):
    """Base class for every error the simulator raises."""


class UnknownReference(SimulatorError):
    """A dispatch / query named something the environment does not define.

    e.g. an unknown process, mode, spot, device, transporter, or activity id, or
    a transport move with no route in the environment's transport table.
    """


class ResourceBusy(SimulatorError):
    """A dispatch would occupy a device or transporter already in use.

    Devices and transporters are exclusive resources (spec §4.4 / §4.6): only one
    activity may access them at a time.
    """


class SpotConflict(SimulatorError):
    """A spot that must be free is occupied.

    Raised when an output / destination spot is already holding material at
    dispatch (D16), or when material is placed onto an occupied spot.
    """


class MissingObject(SimulatorError):
    """A spot that must hold material is empty.

    Raised when a processing input spot or a transport source spot is empty at
    dispatch (D16), or when removing material from an empty spot.
    """


class DeviceDown(SimulatorError):
    """A processing operation was dispatched to a device that is down.

    A down device (spec §7 re-routing) cannot run processes -- but transports to
    and from it, and operations already running on it, are unaffected (dev-notes
    D21). Fully-unavailable devices that strand material are out of scope (D13).
    """


class RelayNotSupported(SimulatorError):
    """A relay was dispatched.

    A relay (spec §4.5 / §6.4.1) is an instantaneous scheduling junction, not a
    physical operation, so the backend has nothing to execute for it (D14). The
    runner keeps relays as bookkeeping and never dispatches them.
    """


class ClockError(SimulatorError):
    """The virtual clock was asked to move backwards (`advance(until)` with
    `until` earlier than the current time)."""
