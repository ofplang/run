"""Why something stopped (D36).

Its own module because both a run and a *job* record one, and the job
(`job.py`) is built before the runner that drives it -- so the two cannot both
reach for it where it used to live, inside `rolling.py`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Failure:
    """Why a run -- or one job of it (SPEC §6.11) -- stopped: a machine-readable
    `kind` (reason code), a human-readable `detail`, the `subject` that failed (a
    node path label, `main`, or an activity), and the virtual time `now` at which it
    was detected.

    A job records the failure that stopped *it*; the runner records the first
    failure of the run, which is that job's when only one job failed. The CLI prints
    them and the final status echoes neither -- it must stay a valid §6 document.
    """

    kind: str
    detail: str
    subject: str
    now: int
