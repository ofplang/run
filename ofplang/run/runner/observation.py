"""The observation document: a run-produced record of completed activities' I/O
view values (dev-notes design.md D38; spec `docs/OBSERVATION.md`).

The plan / status documents (schedule §6) carry an activity's structure, schedule,
and status but not the concrete view values that flowed through its ports. The
observation document fills that gap: it is the *value-layer sibling* of the status
document (`observation : CommitLog :: status : CommitLog`), keyed by the same
provenance, and it records **only completed activities** (values are undetermined
until completion, D27).

This module is the pure, backend-independent projection: it builds the header /
entry / trailer documents from the runner's own value layer (assembled inputs,
recorded outputs, a transport's moved view) and streams them to a file one document
at a time, so the file can be appended to in O(1) per completion and O(n) over a
run -- never re-serialising the whole document. The runner owns the lifecycle (it
calls `ObservationRecorder.record` per completion and `finish` at run end); this
module owns the shape, the value snapshotting, and the file I/O.

The file is a YAML multi-document stream (`---`-separated), read with
`yaml.safe_load_all()`: one header document, then one document per completed
activity in completion order, then one trailer. See `docs/OBSERVATION.md`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TextIO

import yaml

SCHEMA = "ofplang-observation/v0"


# -- document builders (pure) ---------------------------------------------------


def header_doc(
    *, time_section: dict | None = None, interface: dict | None = None, ref: dict | None = None
) -> dict:
    """The stream's first document: run-invariant fields only. `now` is deliberately
    absent (unknown at run start); the final time is in the trailer."""
    doc: dict[str, Any] = {"schema": SCHEMA}
    if time_section:
        doc["time"] = time_section
    if interface:
        doc["interface"] = interface
    if ref:
        doc["ref"] = ref
    return doc


def entry_doc(
    committed, *, inputs: dict | None = None, outputs: dict | None = None, moved: Any = None
) -> dict:
    """One activity entry: the plan/status activity's structural fields echoed
    verbatim (provenance + assignment), the observed `start`/`end`, and the value
    fields. `status` is omitted (every entry is completed by the scope rule).

    Value fields are **deep-copied** here so a later in-place mutation by a device
    model cannot rewrite an already-recorded value (the assembled inputs / recorded
    outputs are shared references into the runner's value store)."""
    activity = committed.activity
    entry: dict[str, Any] = {"kind": committed.kind}
    # Structural echo, in a readable order; skip `status` / `start` / `end` (the
    # record's observed times win, added below).
    for key in (
        "process",
        "mode",
        "node",
        "devices",
        "input_spots",
        "output_spots",
        "from_spot",
        "to_spot",
        "transporter",
        "arc",
        "seq",
    ):
        if key in activity:
            entry[key] = activity[key]
    entry["start"] = committed.start
    entry["end"] = committed.end
    if committed.kind == "transport":
        entry["moved"] = {"view": deepcopy(moved)}
    elif committed.kind == "processing":
        entry["inputs"] = _view_block(inputs)
        entry["outputs"] = _view_block(outputs)
    else:
        # Named rather than left to an `else`, so a kind with nothing to observe
        # cannot arrive here and be written up as a processing with empty ports. The
        # runner decides what is recorded (a refill is skipped: it has no ports and
        # no views, and what it did is a level, which is derived rather than
        # observed); this refuses to invent an entry for anything else.
        raise ValueError(f"an activity of kind {committed.kind!r} has nothing to observe")
    return entry


def trailer_doc(now: int, outcome: str) -> dict:
    """The stream's final document: marks the run complete and records the final
    time and outcome (`completed` | `failed`). A stream with no trailer is an
    unfinished (crashed / killed) run."""
    return {"final": True, "now": now, "outcome": outcome}


def _view_block(values: dict | None) -> dict:
    """`{port: view_value}` -> `{port: {view: <deep copy>}}` (the on-disk shape)."""
    return {port: {"view": deepcopy(value)} for port, value in (values or {}).items()}


# -- serialisation --------------------------------------------------------------


def _dump(doc: dict) -> str:
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def write_stream(
    path: str,
    entries: list[dict],
    *,
    time_section: dict | None = None,
    interface: dict | None = None,
    ref: dict | None = None,
    now: int,
    outcome: str,
) -> None:
    """Write a complete observation stream (header + entries + trailer) in one shot,
    from an in-memory entry list. The runner streams incrementally instead (see
    `ObservationRecorder`); this batch form is for consumers that already hold the
    full list (tests, the render scripts)."""
    docs = [header_doc(time_section=time_section, interface=interface, ref=ref)]
    docs.extend(entries)
    docs.append(trailer_doc(now, outcome))
    text = "---\n".join(_dump(doc) for doc in docs)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


# -- human-readable rendering ---------------------------------------------------


def format_text(entries: list[dict]) -> str:
    """Render the value flow (per **processing** activity: assembled inputs ->
    produced outputs) as human-readable text -- the value-flow section of an example
    `*.trace.txt`, in place of reaching into runner internals. Transports and times
    are omitted: the section is about value transformation (producer -> consumer),
    and a transport carries its view unchanged. Two-space indent, matching the
    render scripts' `activities` section."""
    lines: list[str] = []
    for entry in entries:
        if entry.get("kind") != "processing":
            continue
        node = "/".join(entry.get("node") or ()) or "main"
        lines.append(f"  {node} [{entry.get('process')}]")
        lines.append(f"      in : {_ports_repr(entry.get('inputs'))}")
        lines.append(f"      out: {_ports_repr(entry.get('outputs'))}")
    return "\n".join(lines) + ("\n" if lines else "")


def _ports_repr(block: dict | None) -> str:
    """`{port: {view: v}}` -> `{'port': v, ...}` (or `(none)` when empty), matching
    the plain `{port: view}` shape the render scripts printed."""
    if not block:
        return "(none)"
    return repr({port: cell.get("view") for port, cell in block.items()})


# -- incremental streaming + accumulation ---------------------------------------


class _Stream:
    """Incremental YAML multi-document writer: tail-able and crash-durable.

    The file is opened lazily and the header written just before the first document
    (an entry or the trailer), so the header's `time` section is available (it is
    unknown until the first replan). Each document is flushed on write, so a reader
    can `tail` the stream and a process crash loses at most the final document."""

    def __init__(
        self, path: str, *, interface: dict | None = None, ref: dict | None = None
    ) -> None:
        self._path = path
        self._interface = interface
        self._ref = ref
        self._file: TextIO | None = None
        self._header_written = False

    def _write_header(self, time_section: dict | None) -> None:
        if self._header_written:
            return
        if self._file is None:
            self._file = open(self._path, "w", encoding="utf-8")  # noqa: SIM115 (long-lived handle)
        self._file.write(
            _dump(header_doc(time_section=time_section, interface=self._interface, ref=self._ref))
        )
        self._header_written = True
        self._file.flush()

    def append(self, doc: dict, *, time_section: dict | None = None) -> None:
        self._write_header(time_section)
        assert self._file is not None
        self._file.write("---\n" + _dump(doc))
        self._file.flush()

    def finish(self, now: int, outcome: str, *, time_section: dict | None = None) -> None:
        self._write_header(time_section)
        assert self._file is not None
        self._file.write("---\n" + _dump(trailer_doc(now, outcome)))
        self._file.flush()
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class ObservationRecorder:
    """The runner's observation sink (D38). Accumulates entries in memory (exposed as
    `RollingRunner.observations`) and, when a path is given, also streams them to a
    file. Off entirely when the runner does not enable observation, so a run that
    does not want it pays nothing."""

    def __init__(
        self, *, path: str | None = None, interface: dict | None = None, ref: dict | None = None
    ) -> None:
        self.entries: list[dict] = []
        self._stream = _Stream(path, interface=interface, ref=ref) if path else None

    def record(
        self,
        committed,
        *,
        inputs: dict | None = None,
        outputs: dict | None = None,
        moved: Any = None,
        time_section: dict | None = None,
    ) -> None:
        """Record one completed activity (called after its `ensures` passes)."""
        entry = entry_doc(committed, inputs=inputs, outputs=outputs, moved=moved)
        self.entries.append(entry)
        if self._stream is not None:
            self._stream.append(entry, time_section=time_section)

    def finish(self, now: int, outcome: str, *, time_section: dict | None = None) -> None:
        """Append the trailer and close the stream (run reached a normal end)."""
        if self._stream is not None:
            self._stream.finish(now, outcome, time_section=time_section)

    def close(self) -> None:
        """Close the file if still open, without a trailer (run aborted); a no-op if
        `finish` already ran. Called from the runner's `finally`."""
        if self._stream is not None:
            self._stream.close()
