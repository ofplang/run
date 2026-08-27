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
* **A hang is bounded, not silent** (D40): an optional `op_timeout` gives every coded
  op a deadline, in **real seconds**, measured from the moment its child starts. A
  child still running when the deadline passes is stopped and its op is `failed` with
  reason ``op_timeout`` -- so an instrument that stops answering ends the run the way
  any other failure does (status document, reason, exit 1) instead of blocking it
  forever. Nothing else in the stack bounds an operation: `duration` is an estimate
  for scheduling, not a deadline, and the runner's `max_ticks` counts iterations, not
  time. The default is `None` (no deadline, the behaviour before this existed); a
  dialect that knows its lab -- labcode -- sets one. Stopping the child is **not** a
  cancel: whatever it started outside this process (an instrument command) keeps
  running, and the state that leaves behind is the operator's to restore.

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

# How often `_stop_child` looks to see whether a terminated child has gone, in real
# seconds: short enough that a child which exits at once is barely waited on, long
# enough that the grace period is not a spin.
_KILL_POLL = 0.2


def _op_label(op) -> str:
    """Name an operation in a failure message: what it was doing, not its uuid (which
    means nothing to the operator reading the line)."""
    if op.kind == "transport":
        return f"transport {op.from_spot} -> {op.to_spot}"
    if op.kind == "replenishment":
        return f"replenishment of {op.devices[0] if op.devices else '?'} by {op.replenisher}"
    return f"process {op.process!r} (mode {op.mode!r})"


class _ProcessLike(Protocol):
    """The minimal handle surface the backend polls; `subprocess.Popen` matches it
    structurally, and a test injects a tiny fake with the same shape.

    ``kill()`` is deliberately *not* required: it is used when stopping a child that
    ignored `terminate` (see `_stop_child`), and is called only if the handle has one,
    so a fake or a custom `spawn` written against this protocol keeps working."""

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
        op_timeout: float | None = None,
        kill_grace: float = 5.0,
    ):
        # The device-model seam is how `_complete` gets an op's outputs; we route it
        # to `_resolve_model`, which returns the child's result for a coded op or the
        # built-in default for a timed one (see `_pending`).
        super().__init__(environment, device_model=self._resolve_model)
        if not (seconds_per_tick > 0 and speed > 0):
            raise ValueError(
                f"seconds_per_tick and speed must both be > 0, got {seconds_per_tick} and {speed}"
            )
        if op_timeout is not None and not op_timeout > 0:
            raise ValueError(f"op_timeout must be > 0 or None (no deadline), got {op_timeout}")
        if kill_grace < 0:
            raise ValueError(f"kill_grace must be >= 0, got {kill_grace}")
        self._resolver = resolver
        self._spawn = spawn
        self._seconds_per_tick = seconds_per_tick / speed
        self._monotonic = monotonic
        self._sleep = sleep
        self._epoch: float | None = None
        # How long a coded op may run before its child is stopped and the op failed,
        # in **real seconds** -- unrelated to `seconds_per_tick` (which maps environment
        # time to the wall clock) and never rescaled by it: a deadline is about how long
        # a real instrument may stay silent, not about how the plan's time is paced.
        # None disables it entirely (no deadlines are recorded, nothing is checked).
        self._op_timeout = op_timeout
        # How long a stopped child gets to exit after `terminate` before it is killed.
        self._kill_grace = kill_grace
        # Live child handles and their result-file paths, keyed by op id.
        self._procs: dict[str, _ProcessLike] = {}
        self._result_paths: dict[str, str] = {}
        # When each running child's op must be finished by (`_monotonic` reading), for
        # the ops that have a deadline at all.
        self._deadlines: dict[str, float] = {}
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
        self, process, mode, duration=None, output_schema=None, inputs=None,
        definition=None, node=None,
    ) -> str:
        # The inherited dispatch runs all preconditions, occupies the devices, and
        # registers the running op (its virtual `end` is advisory for a coded op).
        uuid = super().dispatch_processing(
            process, mode, duration=duration, output_schema=output_schema,
            inputs=inputs, definition=definition, node=node,
        )
        code = self._resolver(process, str(mode), inputs, definition)
        if code is not None:
            self._start_child_op(
                uuid, code=code, kind="process", inputs=inputs or {},
                output_schema=output_schema or {}, process=process,
            )
        return uuid

    def _start_child_op(
        self, uuid, *, code, kind, inputs, output_schema=None, process=None
    ) -> None:
        """Start a child process running `code` for op `uuid`, and register its handle so
        the settle loop discovers its completion. Shared by `dispatch_processing` and a
        subclass's transport dispatch (``kind="transport"``), so a coded op -- whatever
        its kind -- is launched and tracked the same way.

        `output_schema` / `process` are carried in the job only for a value-producing op
        (a process, so the child can verify its outputs); a transport passes neither and
        ``kind="transport"`` tells the child there is nothing to verify (side-effect only)."""
        fd, result_path = tempfile.mkstemp(suffix=".json", prefix="ofp-run-")
        os.close(fd)  # the child opens it by path
        job: dict = {
            "code": code,
            "kind": kind,
            "inputs": inputs or {},
            "language": "python",
            "result_path": result_path,
        }
        if output_schema is not None:
            job["output_schema"] = output_schema
        if process is not None:
            job["process"] = process
        self._result_paths[uuid] = result_path
        self._procs[uuid] = self._spawn(job)
        # The deadline runs from the moment the child starts, and covers everything the
        # op does (connecting, every command it issues, its own waiting) -- so it must
        # be looser than any per-command wait inside the script, which knows far more
        # about what it is waiting for.
        if self._op_timeout is not None:
            self._deadlines[uuid] = self._monotonic() + self._op_timeout

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
        the history channel (times live there, D18).

        A coded op whose child is *still running* past its `op_timeout` deadline is
        failed here instead (D40). This is the only place that check needs to live: the
        runner polls on a fixed interval, so it comes back through `advance` -- and
        keeps doing so while it drains after a failure -- for as long as anything runs."""
        running = [op for op in self._ops.values() if op.status == "running"]
        due = []
        overdue = []
        now = self._monotonic() if self._deadlines else 0.0
        for op in running:
            handle = self._procs.get(op.uuid)
            if handle is not None:
                if handle.poll() is not None:  # child exited -> op has finished
                    due.append(op)
                elif now >= self._deadlines.get(op.uuid, float("inf")):
                    overdue.append(op)
            elif op.end <= reached:  # timed op: completes when the clock passes its end
                due.append(op)
        for op in sorted(due, key=lambda o: o.seq):
            if op.should_fail:  # an injected D25 failure (unused by real runs, kept faithful)
                self._fail(op)
            else:
                self._pending = self._collect(op) if op.uuid in self._procs else _TIMED
                pending = self._pending
                if op.output_schema is None and isinstance(pending, dict) and pending.get("error"):
                    # A coded op with no value signature (a transport, output_schema None):
                    # `_complete` applies its material move and skips the value model, so a
                    # child failure would be silently completed. Fail it here with the
                    # reason instead (D25: no material effect). A processing op keeps
                    # failing via its value model (output_schema is not None), unchanged.
                    err = pending["error"]
                    op.reason = (err.get("code", "child_error"), err.get("message", ""))
                    self._fail(op)
                    self._pending = _TIMED
                else:
                    try:
                        self._complete(op)
                    finally:
                        self._pending = _TIMED
            self._history_events.append(
                Event(time=reached, uuid=op.uuid, kind=op.kind, status=op.status)
            )
            self._cleanup(op.uuid)
        # Overdue ops are failed *after* the finished ones: a child that exited on the
        # same pass genuinely did its work, and its real outcome is worth more than the
        # deadline that was about to catch it.
        for op in sorted(overdue, key=lambda o: o.seq):
            self._fail_overdue(op, reached)

    def _fail_overdue(self, op, reached: int) -> None:
        """Stop op `op`'s child and fail the op with the ``op_timeout`` reason (D40).

        The failure is the ordinary graceful one (`_fail`: resources freed, no material
        effect), so it reaches the runner through the path every other operation failure
        takes -- the run stops, the status document is written, the reason is reported.

        The message says the two things its reader needs and cannot get elsewhere: that
        the deadline, not the instrument, ended the wait, and that the instrument was
        never told to stop. A dialect CLI that sets `op_timeout` should say in its own
        words how to raise or lift it (the code, ``op_timeout``, is what it keys on)."""
        self._stop_child(op.uuid)
        limit = self._op_timeout or 0.0
        op.reason = (
            "op_timeout",
            f"{_op_label(op)} did not finish within {limit:.0f}s, so its child process "
            f"was stopped. The work it started outside this process was not cancelled -- "
            f"nothing here can cancel it -- so the instrument may still be running, and "
            f"whatever state that leaves is the operator's to restore. If this operation "
            f"legitimately takes longer, raise op_timeout (or set it to None to wait "
            f"forever).",
        )
        self._fail(op)
        self._history_events.append(
            Event(time=reached, uuid=op.uuid, kind=op.kind, status=op.status)
        )
        self._cleanup(op.uuid)

    def _stop_child(self, uuid: str) -> None:
        """Stop op `uuid`'s child process: ask it to exit, and kill it if it will not.

        `terminate` is a request, and a child wedged in a call that ignores it would
        outlive the run and keep talking to an instrument nobody is watching any more.
        So it gets `kill_grace` real seconds -- polled, not slept through, so a child
        that goes at once costs almost nothing -- and is then killed outright. `kill` is
        used only if the handle has one (`_ProcessLike` does not require it).

        Blocking here is deliberate: the caller is ending this op either way, and a run
        that is on its way out should not leave a process behind to be tidy about it."""
        handle = self._procs.get(uuid)
        if handle is None:
            return
        with contextlib.suppress(OSError, ValueError):
            if handle.poll() is not None:  # already gone
                return
            handle.terminate()
            deadline = self._monotonic() + self._kill_grace
            while handle.poll() is None and self._monotonic() < deadline:
                self._sleep(_KILL_POLL)
            if handle.poll() is None:
                kill = getattr(handle, "kill", None)
                if callable(kill):
                    kill()

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
        self._deadlines.pop(uuid, None)
        handle = self._procs.pop(uuid, None)
        if handle is not None:
            for pipe in (getattr(handle, "stdin", None), getattr(handle, "stderr", None)):
                if pipe is not None:
                    with contextlib.suppress(OSError, ValueError):
                        pipe.close()

    # -- lifecycle -------------------------------------------------------------------

    def close(self) -> None:
        """Stop any still-running children and clean up their artifacts. The runner
        drains running ops on a normal finish, so this only bites on an early stop /
        exception; `run_workflow` calls it in a `finally`.

        It uses the same `_stop_child` as a timed-out op, so a child that ignores
        `terminate` is killed rather than left behind -- which can cost up to
        `kill_grace` per stubborn child, on a run that is already over."""
        for uuid in list(self._procs):
            self._stop_child(uuid)
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
    op_timeout: float | None = None,
    kill_grace: float = 5.0,
) -> Callable[[dict], SubprocessBackend]:
    """Build a `backend_factory(environment) -> SubprocessBackend` for the runner.

    Pass as `RollingRunner(..., backend_factory=subprocess_backend_factory(...))` (or
    via `run_workflow(..., backend_factory=...)`) to run scripts out-of-process on a
    wall clock. See `SubprocessBackend` for the parameters; `resolver` selects each
    op's code (default: the workflow §22 script), `spawn` launches the child (default:
    a real subprocess), `op_timeout` bounds how long an op may run (default: no bound),
    and `monotonic` / `sleep` are injectable for tests."""

    def factory(environment: dict) -> SubprocessBackend:
        return SubprocessBackend(
            environment, resolver=resolver, spawn=spawn,
            seconds_per_tick=seconds_per_tick, speed=speed,
            monotonic=monotonic, sleep=sleep,
            op_timeout=op_timeout, kill_grace=kill_grace,
        )

    return factory
