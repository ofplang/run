"""A real-execution backend: run each op's script out-of-process, paced to a wall
clock, with completion discovered by polling (dev-notes: labcode 06).

`VirtualTimeSimulator` computes a script's outputs *in-process, instantly* at the
op's virtual end; `RealTimeSimulator` keeps that but paces the clock. Neither
actually *runs* work asynchronously -- a long computation blocks the whole loop.
`SubprocessBackend` is the real-execution counterpart: at dispatch it *starts* the
op's script in its own process and returns immediately; the op stays ``running``
until a later poll finds the process finished, exactly the `Backend` contract
(completion is discovered, `duration` is advisory). So a computation that takes
minutes never blocks the runner -- the rolling loop keeps polling and replanning
around it.

Design (reuses the `Simulator` physical oracle, so only time + completion differ):

* It **subclasses `Simulator`**, inheriting the entire physical/value oracle --
  dispatch preconditions, spot/device occupancy, and `_complete`'s material moves.
* **Time** is wall-clock, like `RealTimeSimulator`: `advance(until)` sleeps out the
  real time the step represents, then adopts the tick the clock actually reached
  (`monotonic` / `sleep` are injectable for tests). It is intentionally *not* a
  `RealTimeSimulator` subclass -- that timing helper may later be retired in favour
  of this backend, so the small pacing math is kept local.
* **Completion is driven by the work, not the clock**: a processing op whose code a
  `resolver` returns runs in a child process (via an injectable `spawn`); it
  completes only when that process exits. An op with no code (a transport, or a
  process with no script) is *timed* -- it completes when the clock passes its end,
  exactly as the simulator would, with the built-in `default_device_model` filling
  its outputs. The unifying rule: **code -> subprocess-async; no code -> timed**.
* **Output values** come from the child (its `{"outputs": ...}`), fed to the
  inherited `_complete` via the device-model seam; a child `{"error": ...}` (or a
  crashed child) is raised as `DeviceComputationError`, so it lands on the exact
  same graceful-failure path as an in-process script error (v0 §22.2 / D25).

Scope (0.1.4): the default `resolver` runs only the workflow's own
`python_script_processes` (v0 §22); transports are always timed. A dialect (e.g.
labcode) injects a richer `resolver` (and, later, transport execution) on top.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from typing import Any, Protocol

from .core import Event, Simulator, default_device_model
from .script import DeviceComputationError

# Marker used for the op currently being settled: `_TIMED` means "no child ran, fill
# outputs with the built-in default model" (a transport, or a script-less process).
_TIMED = object()


class _ProcessLike(Protocol):
    """The minimal handle surface the backend polls; `subprocess.Popen` matches it
    structurally, and a test injects a tiny fake with the same shape."""

    returncode: int | None
    stdin: Any
    stderr: Any

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


def default_code_resolver(process, mode, inputs, definition):
    """The default `resolver`: an op's code is its workflow `python_script_processes`
    script (v0 §22), or `None` when it has none (then the op is *timed*).

    A dialect resolver wraps / extends this -- e.g. also sourcing code from an
    environment extension -- but must still return the §22 script for a plain script
    process so a workflow with script processes runs unchanged."""
    script = (definition or {}).get("script")
    if script and script.get("language") == "python":
        return script.get("code") or ""
    return None


def _default_spawn(job: dict) -> _ProcessLike:
    """Launch the child harness (`python -m ofplang.run.simulator._child`), feeding
    the job JSON on its stdin, and return the `Popen` handle immediately.

    The child writes its outcome to ``job["result_path"]`` (not stdout), so the user
    script's own stdout is free; stderr is captured so a harness-level crash can be
    folded into the failure reason. The handle exposes the minimal seam the backend
    polls: ``poll()`` (None while running), ``returncode``, and ``stderr``."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "ofplang.run.simulator._child"],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Inputs are small (view values), so a single write + close does not deadlock; the
    # child reads stdin to EOF before running anything.
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(job))
    proc.stdin.close()
    return proc


class SubprocessBackend(Simulator):
    """A `Simulator` that runs op scripts out-of-process on a wall clock (see module
    docstring). Inherits the physical/value oracle; overrides only time (`advance`)
    and how completion + outputs are obtained (subprocess for coded ops, timed for
    the rest)."""

    def __init__(
        self,
        environment,
        *,
        resolver: Callable = default_code_resolver,
        spawn: Callable[[dict], _ProcessLike] = _default_spawn,
        seconds_per_tick: float = 1.0,
        speed: float = 1.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        # The device-model seam is how `_complete` gets an op's outputs; we route it
        # to `_resolve_model`, which returns the child's result for a coded op or the
        # built-in default for a timed one (see `_pending`).
        super().__init__(environment, device_model=self._resolve_model)
        if not (seconds_per_tick > 0 and speed > 0):
            raise ValueError(
                f"seconds_per_tick and speed must both be > 0, got {seconds_per_tick} and {speed}"
            )
        self._resolver = resolver
        self._spawn = spawn
        self._seconds_per_tick = seconds_per_tick / speed
        self._monotonic = monotonic
        self._sleep = sleep
        self._epoch: float | None = None
        # Live child handles and their result-file paths, keyed by op id.
        self._procs: dict[str, _ProcessLike] = {}
        self._result_paths: dict[str, str] = {}
        # The result for the op currently being settled by `_complete` (set just
        # before the call): `_TIMED`, or the child's ``{"outputs"|"error": ...}``.
        self._pending: Any = _TIMED

    # -- value seam: feed `_complete` the child's outputs (or the default) ---------

    def _resolve_model(self, process, mode, inputs, output_schema, definition):
        """The device model `_complete` calls at completion. Returns the child's
        outputs for a coded op; raises `DeviceComputationError` for a child error (so
        `_complete` fails the op with a reason, D25); falls back to the built-in
        default model for a *timed* op (no child ran)."""
        pending = self._pending
        if pending is _TIMED:
            return default_device_model(process, mode, inputs, output_schema, definition)
        if "error" in pending:
            err = pending["error"]
            raise DeviceComputationError(
                err.get("message", "child failed"), code=err.get("code", "child_error")
            )
        return pending.get("outputs") or {}

    # -- dispatch: start a child for a coded op; leave a script-less op timed --------

    def dispatch_processing(
        self, process, mode, duration=None, output_schema=None, inputs=None, definition=None
    ) -> str:
        # The inherited dispatch runs all preconditions, occupies the devices, and
        # registers the running op (its virtual `end` is advisory for a coded op).
        uuid = super().dispatch_processing(
            process, mode, duration=duration, output_schema=output_schema,
            inputs=inputs, definition=definition,
        )
        code = self._resolver(process, str(mode), inputs, definition)
        if code is not None:
            fd, result_path = tempfile.mkstemp(suffix=".json", prefix="ofp-run-")
            os.close(fd)  # the child opens it by path
            job = {
                "code": code,
                "inputs": inputs or {},
                "output_schema": output_schema or {},
                "process": process,
                "language": "python",
                "result_path": result_path,
            }
            self._result_paths[uuid] = result_path
            self._procs[uuid] = self._spawn(job)
        return uuid

    # -- time: pace the wall clock, then settle whatever finished --------------------

    def advance(self, until: int) -> int:
        """Block until the wall clock reaches `until`'s due time (like
        `RealTimeSimulator`), adopt the tick real time actually reached, then settle:
        complete every child that has finished and every timed op whose end has
        passed. Returns the reached tick (>= `until`)."""
        if self._epoch is None:
            self._epoch = self._monotonic() - self.now * self._seconds_per_tick
        due = self._epoch + until * self._seconds_per_tick
        remaining = due - self._monotonic()
        if remaining > 0:
            self._sleep(remaining)
        elapsed = self._monotonic() - self._epoch
        reached = int(elapsed / self._seconds_per_tick)
        reached = max(reached, until, self.now)
        self._clock = reached
        self._settle_reached(reached)
        return self._clock

    def _settle_reached(self, reached: int) -> None:
        """Complete each running op that has finished by `reached`: a coded op when its
        child has exited (polled), a timed op when its virtual end has passed. Ties
        are broken by dispatch order (`seq`), matching the base simulator. Each
        completion is applied via the inherited `_complete` / `_fail`, and recorded in
        the history channel (times live there, D18)."""
        running = [op for op in self._ops.values() if op.status == "running"]
        due = []
        for op in running:
            handle = self._procs.get(op.uuid)
            if handle is not None:
                if handle.poll() is not None:  # child exited -> op has finished
                    due.append(op)
            elif op.end <= reached:  # timed op: completes when the clock passes its end
                due.append(op)
        for op in sorted(due, key=lambda o: o.seq):
            if op.should_fail:  # an injected D25 failure (unused by real runs, kept faithful)
                self._fail(op)
            else:
                self._pending = self._collect(op) if op.uuid in self._procs else _TIMED
                try:
                    self._complete(op)
                finally:
                    self._pending = _TIMED
            self._history_events.append(
                Event(time=reached, uuid=op.uuid, kind=op.kind, status=op.status)
            )
            self._cleanup(op.uuid)

    def _collect(self, op) -> dict:
        """Read a finished child's outcome: its ``{"outputs"|"error": ...}`` from the
        result file on a clean (rc 0) exit, else a child-level error folding the
        captured stderr (a harness crash / unreadable result)."""
        handle = self._procs[op.uuid]
        path = self._result_paths.get(op.uuid)
        rc = getattr(handle, "returncode", None)
        if rc == 0 and path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError) as exc:
                return {"error": {"code": "child_error", "message": f"unreadable result: {exc}"}}
        stderr = ""
        with contextlib.suppress(OSError, ValueError):
            if getattr(handle, "stderr", None) is not None:
                stderr = handle.stderr.read() or ""
        detail = f"child exited {rc}: {stderr[-500:]}".strip()
        return {"error": {"code": "child_error", "message": detail}}

    def _cleanup(self, uuid: str) -> None:
        """Drop a finished op's child artifacts: delete its result temp file and close
        the handle's pipes."""
        path = self._result_paths.pop(uuid, None)
        if path:
            with contextlib.suppress(OSError):
                os.unlink(path)
        handle = self._procs.pop(uuid, None)
        if handle is not None:
            for pipe in (getattr(handle, "stdin", None), getattr(handle, "stderr", None)):
                if pipe is not None:
                    with contextlib.suppress(OSError, ValueError):
                        pipe.close()

    # -- lifecycle -------------------------------------------------------------------

    def close(self) -> None:
        """Terminate any still-running children and clean up their artifacts. The
        runner drains running ops on a normal finish, so this only bites on an early
        stop / exception; `run_workflow` calls it in a `finally`."""
        for uuid in list(self._procs):
            handle = self._procs.get(uuid)
            with contextlib.suppress(OSError, ValueError):
                if handle is not None and handle.poll() is None:
                    handle.terminate()
            self._cleanup(uuid)

    def __enter__(self) -> SubprocessBackend:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def subprocess_backend_factory(
    *,
    resolver: Callable = default_code_resolver,
    spawn: Callable[[dict], _ProcessLike] = _default_spawn,
    seconds_per_tick: float = 1.0,
    speed: float = 1.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[dict], SubprocessBackend]:
    """Build a `backend_factory(environment) -> SubprocessBackend` for the runner.

    Pass as `RollingRunner(..., backend_factory=subprocess_backend_factory(...))` (or
    via `run_workflow(..., backend_factory=...)`) to run scripts out-of-process on a
    wall clock. See `SubprocessBackend` for the parameters; `resolver` selects each
    op's code (default: the workflow §22 script), `spawn` launches the child (default:
    a real subprocess), and `monotonic` / `sleep` are injectable for tests."""

    def factory(environment: dict) -> SubprocessBackend:
        return SubprocessBackend(
            environment, resolver=resolver, spawn=spawn,
            seconds_per_tick=seconds_per_tick, speed=speed,
            monotonic=monotonic, sleep=sleep,
        )

    return factory
