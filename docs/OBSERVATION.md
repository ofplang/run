# Observation document (v0)

The **observation document** is a run-produced record of the concrete input/output
values that completed activities actually had during a run. The plan / status
documents (schedule spec §6) record an activity's *structure and schedule* (kind,
provenance, assignment, times, status) but **not** the concrete values that flowed
through its ports. The observation document fills that gap.

It is the *value-layer sibling* of the status document: both are projections of the
runner's committed history, keyed the same way, but the status document answers
"what ran and when" while the observation document answers "with what values".

- Owner: **ofplang-run** only. The scheduler never produces or consumes it, and it
  is never fed back into planning (the scheduler is value-independent — dev-notes
  D9 / D26). The observation document is a leaf output, and has **no conformance
  validator** in v0 (nothing consumes it as a contract). This feature is dev-note
  **D38**.
- Relationship to `boundary` (see `runner/boundary.py`): the boundary document
  already captures the *interface* input/output views; the observation document is
  the generalization of that idea to **every** completed activity, internal ones
  included. Overlap with the boundary and status documents is intentional and
  allowed — the documents have different roles.

## 1. Scope

- **Only `completed` activities are recorded.** Values are undetermined until an
  activity completes (D27), so only completed activities have finalized values.
  Pending / running / failed / cancelled activities produce **no** observation
  entry at all.
- **View values only.** Each recorded value is a `.view` projection (spec §7.4).
  Physical Object identity (the simulator's opaque per-spot `obj_id`, D15) is **not**
  recorded in v0 — it is an unstable per-spot occupancy token, regenerated on every
  processing step, not a scenario-wide identity, so it would be misleading to expose.
  Object continuity is already expressed structurally by `arc` + `seq` provenance.
- Relays (schedule spec §6.4.1) produce no entry, mirroring the status document
  (the scheduler regenerates them from committed transport legs).
- **Replenishments produce no entry.** A refill has no ports and no views, so there
  is nothing this document is about; what it did is a stock *level*, and a level is
  derived from the status (the starting levels plus the history, schedule spec
  §4.7.2) rather than observed. An empty entry would say only "a refill happened",
  which the status already says, with times.
- **Same-spot no-op transports are not recorded** (`from_spot == to_spot`: a physical
  no-op that moves nothing and changes no value). **Value-less completed processing**
  (an activity with no output ports / no signature) *is* recorded, with empty
  `inputs` / `outputs` maps.
- **Times mirror the status document.** An entry's `start` / `end` are the runner's
  committed times — in fixed-interval polling `end` is the poll at which completion
  was first seen (an upper bound), identical to the status activity's, **not** the
  backend's true finish.

## 2. File form

- One file per run, conventionally `*.observation.yaml`, written via the CLI
  `--observation-out <path>`.
- The file is a **YAML multi-document stream** (`---`-separated documents), read with
  `yaml.safe_load_all()`. This format is chosen so the file can be **appended to**
  one document at a time as activities complete, in O(1) per completion and O(n) over
  a run — never re-serializing the whole document (which would be O(n²) and dominate
  runtime for large plans). A crashed run leaves a valid stream whose only possible
  damage is a truncated final document.

Document order in the stream:

1. **Header** (exactly one, written once at run start) — run-invariant fields only.
2. **Activity entries** (zero or more, one appended as each activity completes) — in
   *completion order*. Each entry carries `start`/`end`, so a consumer can sort to
   time order.
3. **Trailer** (exactly one, appended once at run end) — marks the stream complete
   and records the final `now`.

A stream without a trailer denotes a run that did not finish (crashed, killed, or
still in progress).

## 3. Header document

```yaml
schema: ofplang-observation/v0     # required, identifies the format + version
time: {unit: second}               # the run's time unit (mirrors the plan)
interface: {...}                   # optional: the §6.8 boundary constraint, carried verbatim
ref:                               # optional: filenames this observation accompanies
  plan: plan.yaml
  status: status.yaml
```

`now` is **not** in the header — it is unknown at run start. The final `now` is in
the trailer.

## 4. Activity entry documents

An entry echoes the plan/status activity's **structural fields verbatim** (the same
fields the scheduler emitted, so the entry is self-describing and pairs 1:1 with the
plan/status activity by provenance), plus the observed `start`/`end`, plus the
**value fields**. `status` is omitted: every entry is `completed` by the scope rule
(§1). Provenance keys: processing is identified by `node`; transport by `arc` + `seq`.
(A replenishment is identified by its `id` -- it has no workflow provenance, the solver
having placed it rather than the workflow asking for it -- but it is never recorded
here, so no entry carries one.)

### 4.1 Processing

```yaml
kind: processing
process: heat_sample
mode: fast
node: [heat]
devices: [incubator_0]                       # echoed if the mode has devices
input_spots:  {plate: incubator_0.slot_0}    # echoed if present
output_spots: {plate: incubator_0.slot_0}    # echoed if present
start: 0
end: 60
inputs:                                       # value: one entry per input port
  plate: {view: {barcode: ABC}}
outputs:                                      # value: one entry per output port
  plate: {view: {barcode: ABC, temperature: 37}}
```

- `inputs.<port>.view` is the value actually consumed (as assembled by the runner —
  connected producer, literal, or typed default; `runner/values.py:assemble_inputs`).
- `outputs.<port>.view` is the value the runner records at completion from the
  backend's `state()` report (normalized and contract-checked; backend-independent —
  see §9), not a simulator internal.
- Both Object-bearing and Pure Data ports appear. A **Pure-Data-only** processing
  activity simply has no `devices` / `input_spots` / `output_spots` (they are omitted
  by the plan), and its `inputs`/`outputs` carry ordinary Pure Data views.

### 4.2 Transport

```yaml
kind: transport
from_spot: incubator_0.slot_0
to_spot: reader_0.stage
transporter: arm_0                            # always present (same-spot no-ops are not recorded — §1)
arc:
  from: {node: [heat],  port: plate}
  to:   {node: [assay], port: plate}
seq: 0                                         # present only on a multi-leg move
start: 60
end: 80
moved:                                         # value: the moved Object's view
  view: {barcode: ABC}
```

- `moved.view` is the `.view` of the Object carried by this leg.
- A **boundary** transport (spec §6.8) uses an empty node path `[]` on the interface
  side of `arc`, exactly as in the plan.

### 4.3 Empty views

An Object-bearing port whose type declares no view fields is recorded with an empty
view record:

```yaml
outputs:
  plate: {view: {}}
```

The presence of the port (with `view: {}`) records that the Object flowed through it;
there is simply no contract-visible projection to report.

## 5. Trailer document

```yaml
final: true
now: 130               # the run's final time (last observed completion, or the stop time)
outcome: completed     # completed | failed — a run stopped by a failure (D25) is `failed`
```

## 6. Full example

```yaml
schema: ofplang-observation/v0
time: {unit: second}
ref: {plan: plan.yaml, status: status.yaml}
---
kind: processing
process: heat_sample
mode: fast
node: [heat]
devices: [incubator_0]
input_spots:  {plate: incubator_0.slot_0}
output_spots: {plate: incubator_0.slot_0}
start: 0
end: 60
inputs:  {plate: {view: {barcode: ABC}}}
outputs: {plate: {view: {barcode: ABC, temperature: 37}}}
---
kind: transport
from_spot: incubator_0.slot_0
to_spot: reader_0.stage
transporter: arm_0
arc:
  from: {node: [heat],  port: plate}
  to:   {node: [assay], port: plate}
start: 60
end: 80
moved: {view: {barcode: ABC}}
---
kind: processing            # a Pure-Data-only step: no devices / spots
process: compute_mean
mode: default
node: [analyze]
start: 80
end: 80
inputs:  {samples: {view: {length: 3}}}
outputs: {mean: {view: 12.7}}
---
final: true
now: 130
outcome: completed
```

## 7. Notes for consumers

- Entries are in **completion order**, not start-time order. Sort by `(start, end)`
  to recover the plan's ordering.
- A missing trailer means the run did not complete; entries seen so far are still
  valid and final (a completed activity is terminal and never revised, even across
  replanning rounds).
- This document is **not** the `*.trace.txt` render-script logs (those are
  human-readable diagnostics). The observation document is a structured data product.

## 8. Relationship to the example trace scripts

The `examples/render_*.py` scripts emit human-readable `*.trace.txt` files. Those
scripts stay as they are and keep emitting their own, example-specific content
(entry-input listings, static-literal provenance, declared contracts + runtime
verdict, whole-workflow `returns` with producer provenance) — none of which is the
observation document's concern.

What they should stop doing is hand-extracting the **value flow** (per-activity
assembled inputs → produced outputs) by reaching into runner internals
(`runner.values`, `assemble_inputs`, `ValueStore.snapshot`). That is exactly what
this document captures, so the value-flow section of a trace is produced by the
observation projection instead. To make that reuse possible, the runner accumulates
the entries in memory and `runner/observation.py` exposes them as data plus two
renderers, not just a file writer:

- `RollingRunner(observe=True)` accumulates the entries (built once, at completion,
  from the runner's value layer) and exposes them as `RollingRunner.observations` —
  the single value-extraction path.
- `observation.write_stream(path, entries, *, ...)` — serializes a full YAML
  multi-document stream from an entry list (§2), for consumers that hold the whole
  list (tests / batch use). The runner itself streams incrementally instead.
- `observation.format_text(entries)` — pretty-prints entries as human-readable text,
  which the render scripts call for their value-flow section.

So there is one extraction (`RollingRunner.observations`) and two renderings (YAML
for machines, text for humans). The observation document stays lean — entry inputs,
`returns`, literals, and contracts are **not** added to it; the trace scripts compose
those from the boundary document, the workflow, and the contract-observer channel
(D36).

## 9. Implementation notes (runner)

These are the runner-side facts that make the capture correct and cheap. They are
not part of the file format, but they constrain step 1.

- **Enablement / overhead.** Observation is off by default and pays nothing when off.
  Two knobs: `observe: bool` turns on in-memory accumulation (the render scripts set
  this without a file to use `build_entries`/`format_text`); `observation_out: <path>`
  additionally writes the stream file and implies `observe`. When off, no inputs are
  stashed, no entries are built, no file is opened. When on, the accumulated entries
  are also exposed on the runner as `RollingRunner.observations` for programmatic use.
- **Input stash.** A processing activity's inputs are assembled at dispatch
  (`assemble_inputs`, already computed for the `requires` check and the backend send —
  no extra computation). They are **not** retained on the `Committed` record (that
  record lives in the whole-run `CommitLog` that feeds replan status, which must stay
  value-free). Instead stash them in a transient uuid-keyed map populated when the
  running record is created and **popped at emit**, so retained memory is bounded by
  in-flight activities. A transport stashes its `moved` view (`_transported_view`,
  computed at dispatch) the same way. No stash for `requires`-failed (never
  dispatched) or same-spot no-op (not emitted) activities.
- **Snapshot on capture.** Assembled inputs and recorded outputs are the same value
  objects held in the `ValueStore` (shared references). The stream file is safe (an
  entry is serialized to YAML immediately on append), but the in-memory entries that
  `build_entries` returns must hold **deep copies** taken at emit, so a later in-place
  mutation by a device model cannot rewrite an already-recorded value.
- **Emit point and idempotency.** Append exactly once, on the running→completed
  transition, **after** the `ensures` postcondition check passes (a tentatively
  completed op can still flip to `failed` and have its output discarded). Two
  completion sites: `_poll` for real ops, and `_commit_start` for the same-spot no-op
  (which is filtered out, so in practice only `_poll` emits). Never re-emit on a
  replan re-render (unlike `build_status`, which re-renders every committed record
  each round). Replanning does not duplicate committed records (completed/running are
  fixed and never re-dispatched), so one completion is one entry.
- **File handling.** Open once at run start, write the header, flush. Append one
  document per emit and flush (so the stream is tail-able and survives a process
  crash; fsync is not required). Append the trailer and close at run end; close in a
  `finally` so an aborted run leaves a valid, trailer-less (= incomplete) stream.
- **Memory.** The in-memory entries list is O(n) — inherent to the product — and
  overlaps the `ValueStore`, so an enabled run holds roughly twice the view-value
  data. This is why observation is gated (§9, Enablement). CPU is O(n) total / O(1)
  per activity; there is no full-document re-render.

Fidelity note: the capture uses the **runner's own value layer** (backend
independent — the same values for the built-in simulator and a real backend such as
labcode). Outputs are the values the runner records at completion from the backend's
`state()` report (normalized and contract-checked; `record_outputs`). Inputs are the
values the runner assembled at **dispatch** (`assemble_inputs`), stashed at dispatch
in a transient uuid-keyed map (`_pending_capture`, popped at emit) since the
`Committed` record does not retain them. This is captured **per completion**, whereas
the old trace scripts recomputed inputs post-hoc against the final `ValueStore`. For
straight-line runs the two agree; for re-executed nodes (reroute / replan / loops) the
per-occurrence observation projection is the more faithful record. A transport's
`moved.view` is the value the runner resolves for the moved Object at dispatch
(`_transported_view`).
