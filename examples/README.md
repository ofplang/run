# Examples

Runnable demonstrations of what the runner does with a workflow: the value layer
routing and computing view values, and the rolling-horizon loop reacting to a lab
that does not behave as planned. These are not tests (those live under `tests/`);
they are complete scenarios that print or draw what happened.

They are Python scripts rather than `ofp-run` invocations because the interesting
part of each one is *injected*: a device model that computes, a device that goes
down mid-run, a polling interval coarse enough to lose time. Those are backend and
scenario concerns a caller supplies in code — deliberately not CLI flags. Each
script takes no arguments and needs the sibling `ofplang-schedule` installed (the
runner replans through it, and its visualizer draws the charts):

```sh
python examples/render_job_run.py
```

Generated artifacts live under `outputs/` and are committed, so an example can be
read without being run: a text trace of the value flow (`<name>.trace.txt`), the
observation document the run streamed (`<name>.observation.yaml` — what
`ofp-run run --observation-out FILE` writes), the result boundary where the example
supplies one (`<name>.boundary.yaml` — what `--boundary-out FILE` writes), and
rendered SVG Gantt charts. Re-running an example reproduces its committed files
byte for byte.

## `job_run` — supplied inputs, computed outputs

- `count_chain.workflow.yaml` — a `Count` (view `{value: Int}`) enters at the run
  boundary, passes through two device-less `inc` steps, and is returned as the
  whole-workflow output.
- `count_chain.env.yaml` — the environment for it.
- `render_job_run.py` — supplies `{start: {value: 42}}` at the boundary and injects
  a device model in which `inc` adds one, so the result boundary reads
  `{result: {value: 44}}`.

The whole value story in one run: the caller supplies the entry views, a device
model computes each step's outputs from its inputs, and the runner assembles the
whole-workflow outputs and echoes them into a result boundary of the same schema.
The value is *transformed* down the chain rather than merely carried.

The same workflow runs from the CLI — it is also the input `tests/test_cli.py`
drives:

```sh
ofp-run run examples/count_chain.workflow.yaml --env examples/count_chain.env.yaml \
    --boundary-out /tmp/result.yaml
```

With no boundary supplied the entry views default, and with no device model
injected `inc` computes nothing, so `result` comes back `{value: 0}` instead of
`{value: 44}`. That gap is exactly why this example is a script: the computation
lives in the model a real backend would provide.

## `plate_chain` — Pure Data and an Object, carried together

- `plate_chain.workflow.yaml` — each `step` carries two values at once: a Pure Data
  `Int` counter and an Object-bearing `Plate`, the same plate in and out (an
  in-place transform).
- `plate_chain.env.yaml` — the plate is loaded on `loader`, processed by `step` on
  `worker`, and delivered to `unloader`; `arm` moves it along. The Int ports are
  Pure Data and occupy nothing.
- `render_plate_chain.py` — the boundary supplies `start: 42` and
  `sample: {barcode: "ABC"}` on the loader; the injected model increments the Int
  and passes the plate through unchanged, yielding
  `{result: 44, plate_final: {barcode: "ABC"}}` at the unloader.

Both halves of a boundary port appear here in one document — `spot`, where the
Object sits, and `view`, its value. The Int is transformed down the chain (42 → 43
→ 44) while the Plate's view is carried through untouched, its identity tracked
physically by the simulator as it is loaded, processed, and delivered.

## `data_flow` — routing views across a composite boundary

- `data_flow.workflow.yaml` — an Object plate is measured (emitting a Pure Data
  `Reading`), a nested `Analyzer` composite turns the reading into a `Score`, and
  that score is both a gate for the final Object step and the workflow's returned
  output.
- `data_flow.env.yaml` — `loader`, `reader` and `printer`, with `arm` between them.
- `render_data_flow.py` — traces each activity's assembled inputs and produced
  outputs, then the whole-workflow outputs.

This is the value layer with *no* computation injected, which is what makes the
routing visible on its own: the backend fills each output port with the typed
default its view schema calls for (§7.4 — a `Reading` becomes `{mean: 0.0, n: 0}`,
a `Score` `{value: 0.0, ok: false}`), and the runner carries those along the
workflow's arcs, across the composite boundary and out through `returns`.

## `script_literal` — a literal, a script, and contracts checked at runtime

- `script_literal.workflow.yaml` — a measured `raw` from the run boundary and a
  `threshold` embedded as a static literal (`value: 60`, §11) both feed a `score`
  script process (§22) computing `margin` and `passed`; a `report` script turns
  those into a summary string. `score` declares a `requires` precondition and
  `ensures` postconditions, `report` an `ensures` (§9).
- `script_literal.env.yaml` — both are device-less Pure-Data-only script processes,
  so each `duration` is the scheduler's estimate of a compute cost rather than an
  instrument's time.
- `render_script_literal.py` — traces which inputs were seeded at the boundary,
  which came from the literal, the computed outputs per activity, and the contracts
  as they are checked.

Unlike `data_flow`, the outputs here are genuinely computed and the contracts bite
on real values — the first point in the stack where the value layer is doing
arithmetic rather than carrying defaults.

## `reroute` — routing around a device that goes down

- `reroute.workflow.yaml` / `reroute.env.yaml` — `target` can run on `station_1`
  (cheap) or `station_2`, so the scheduler's initial plan sends it to `station_1`.
- `render_reroute.py` — takes `station_1` down just after the sample has been
  delivered to it, by calling `sim.schedule_device_down` directly. The
  rolling-horizon runner re-routes `target` to `station_2` via a relay and a second
  transport.

It writes two charts so the decision is visible side by side:
`outputs/reroute.initial.svg` (the plan as first proposed) and
`outputs/reroute.final.svg` (what the run actually did). Machine up/down is a
scenario concern driven from Python; there is no CLI knob for it.

## `poll_drift` — the cost of looking at fixed intervals

- `render_poll_drift.py` — reuses the `reroute` workflow and environment with *no*
  fault injected, and drives it twice: once observing each completion the instant
  it happens, once polling every `D = 3`.

Polling only sees a completion at the next poll, so each activity is recorded as
finishing there — an upper bound — and its successors slip: source 0–2, transport
2–3, target 3–5 (makespan 5) becomes source 0–3, transport 3–6, target 6–9
(makespan 9). The backend's history still holds the true finishes (2, 4, 8) while
the polled schedule reports 3, 6, 9. Both are rendered
(`outputs/poll_drift.exact.svg`, `outputs/poll_drift.polled.svg`) so the drift can
be read off the two charts.

## `shared_refill` — two jobs run together, and the refill only the pair needs

- `shared_refill.workflow.yaml` — one plate: made, assayed, discarded. It says
  nothing about consumables anywhere.
- `shared_refill.env.yaml` — one `reader` holding at most 6 units of `reagent`;
  each assay draws 2, and a `dispenser` can top it up.
- `shared_refill.run.yaml` — the run document: two jobs, both running that
  workflow, plus what the reader holds at the start of the run (2 units).
- `render_shared_refill.py` — runs one job, then the two together, and prints both
  schedules side by side.

One job needs 2 units and has 2, so it is planned with no replenishment at all. Two
jobs need 4, so exactly **one** refill appears — covering both. Neither workflow
mentions a resource and neither asks to be refilled: the stock belongs to the
*device* (SPEC §4.7), so running the two jobs against one laboratory is what puts
them on one stock, and one visit tops it up for whichever assays follow, from either
job. That refill carries no `job` in the status: the scheduler decided to run it, and
it serves both.

Unlike the examples above, nothing here is injected — the two-job run is exactly
what the CLI does:

```sh
ofp-run run --jobs examples/shared_refill.run.yaml \
    --env examples/shared_refill.env.yaml
```

The script exists for the *contrast*: one run cannot show what the other job
changed. It writes `outputs/shared_refill.txt`.
