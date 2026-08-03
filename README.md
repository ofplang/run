# ofplang run

[![CI](https://github.com/ofplang/run/actions/workflows/ci.yml/badge.svg)](https://github.com/ofplang/run/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ofplang-run.svg)](https://pypi.org/project/ofplang-run/)

A runner for **Object-flow Programming Language v0** — a YAML-based dataflow
workflow IR with linear Object tracking. The language is defined in the
[ofplang/spec](https://github.com/ofplang/spec) repository.

The runner drives an ofplang v0 workflow to completion against an execution
backend, emitting an execution status document (spec §6/§7) as it progresses and
routing typed **view values** through the workflow. It runs on a **simulator** — a
simulated physical backend — so a full workflow can be exercised end to end
without real hardware; the same dispatch contract targets real hardware later.

> **Status:** the simulator, the runner, and a typed (dummy) value layer are
> implemented.
>
> - **Simulator** (`ofplang.run.simulator`) — a physical backend: devices, spots,
>   transporters, and timed operations advanced on a clock. It validates every
>   dispatch (an inconsistent plan is rejected), models timed up/down for a device
>   or a transporter and injected operation failure, and at completion produces each
>   operation's output
>   view values via a **device model** (the built-in `default_device_model` fills
>   type defaults and carries Object outputs through from their `objects.map`
>   inputs; a custom / real model computes them). `Simulator` is an abstract base
>   over the runner's `Backend` contract; the concrete `VirtualTimeSimulator`
>   advances time instantly (deterministic; the default) and `RealTimeSimulator`
>   paces it to a wall clock (a hardware-free stand-in for a real backend).
> - **Runner** (`ofplang.run.runner`) — two ways to drive a backend:
>   - **`replay`** runs a given execution plan (spec §6) on the backend verbatim.
>   - **`run`** is a rolling-horizon loop: it calls
>     [`ofplang.schedule`](https://github.com/ofplang/schedule) each tick,
>     dispatches the work that can start now, advances the clock, and polls —
>     replanning from the committed history as it goes. It re-routes around a
>     downed machine — a device (its process modes and, by default, its spots'
>     transports) or a transporter (the transports it carries) — polls at a fixed
>     interval with completion-time estimation, absorbs duration variance (an
>     operation running longer or shorter than planned), and stops the whole run if
>     any activity fails (marking the abandoned work cancelled). Machine up/down,
>     operation failure, and duration variance are scenario concerns injected from
>     Python (not CLI flags).
> - **Value layer** — the runner resolves each port's type and view schema (§7),
>   routes typed view values along the workflow's arcs (producer output → consumer
>   input, across nested composites), contract-checks them, and assembles the
>   whole-workflow outputs. A caller supplies the whole-workflow I/O as a single
>   **run boundary** (`--boundary`): one document with a per-port `{spot, view}`
>   descriptor — `spot` places a boundary Object (§6.8), `view` supplies an input
>   value — for the workflow's entry inputs and final outputs. Unsupplied entry
>   views default. A workflow-embedded static literal (`bind: {port: {value: …}}`,
>   §11) is seeded as that consumer input's value in place of a default. At run end
>   the produced output views are echoed back into a result boundary of the same
>   schema (`--boundary-out`). Non-script values are typed but still dummy — a real
>   device backend plugs into the same seam later.
> - **Python script processes** (spec §22, `python_script_processes`) — an atomic
>   Pure-Data process may carry a `script: {language: python, code: …}` section.
>   The built-in device model runs it: the input port values are bound as locals,
>   the code returns a mapping of the declared outputs, and those become the
>   operation's computed output values — the first genuine (non-dummy) computation
>   in the value layer. A script that raises, returns the wrong output names, or
>   returns a non-conformant value fails runtime verification (§22.2) and stops the
>   run gracefully like any other activity failure (the failed activity is marked
>   `failed`, its unstarted successors `cancelled`, exit 1). The script runs inline
>   in real time but advances no simulation time; its outputs appear when the
>   operation completes at its `end`, and the environment mode's `duration` is the
>   scheduler's estimate of the compute cost. Scripts run with full Python builtins
>   and no import restriction (§22.3 permits an implementation to restrict these;
>   this one does not).
> - **Contracts** (spec §9) — an atomic process may declare `contracts` with
>   `requires` (preconditions over `inputs.*.view`) and `ensures` (postconditions
>   over `inputs.*.view` and `outputs.*.view`). The runner evaluates them at
>   runtime against the actual view values: `requires` before the operation runs
>   (a violation stops it before dispatch), `ensures` after it completes. An atomic
>   `requires` that references only run/graph-phase inputs (§5.6) is knowable at run
>   start, so it is checked there as a *preflight* — before any work is dispatched —
>   rather than waiting for that (possibly late) operation. A
>   violation — or a runtime evaluation error — stops the run gracefully like any
>   activity failure (`failed` + downstream `cancelled`, exit 1). Static /
>   graph-time contract checking (a fully-constant contract that is statically
>   false, type errors, bad references) is `ofplang-validate`'s job and is not
>   repeated here. Contracts are checked on atomic processes and on the top-level
>   entry composite (`main`): the entry composite's `requires` is evaluated at run
>   start over the whole-workflow inputs (a violation stops the run before any work
>   runs) and its `ensures` at run end over the whole-workflow inputs and outputs (a
>   violation marks the completed run failed). Nested composite contracts are checked
>   too, at each composite invocation's value boundary: its `requires` once its inputs
>   are available (for an input fed by an upstream process this is mid-run, before the
>   composite's body runs) and its `ensures` once its outputs are, a violation
>   stopping the run gracefully at the composite boundary (its not-yet-run body
>   cancelled). Because non-script process outputs are still typed defaults, `ensures`
>   bites mainly on script processes and boundary-supplied inputs until a real device
>   backend computes physical outputs.
> - **Failure observability** — when a run stops (an injected activity failure, a
>   script error, or a contract violation), the reason is exposed as a structured
>   `RollingRunner.failure` (a machine-readable `kind` code, e.g. `contract_requires`
>   / `script_error`, plus a human-readable `detail`, the `subject`, and the time),
>   and the CLI prints it to stderr. The status document itself stays a valid §6
>   document (the reason is out of band). An optional `contract_observer` callback is
>   invoked for every contract check (held or violated) — a trace hook for debugging.
> - **Static view values** (spec §7.4) — a type whose view field declares a `value:`
>   fixes that field to a constant for every value of the type. The runner projects it
>   onto every view record it routes, so a Python script reading the field and a
>   contract referencing it both see the static value (not a runtime default). A value
>   that carries a conflicting one is forced to the static value.

## Install

```sh
pip install ofplang-run
```

Requires Python 3.10+. Runtime dependencies (pulled in automatically) are PyYAML,
the sibling [`ofplang-schedule`](https://pypi.org/project/ofplang-schedule/) that
`run` (rolling-horizon) calls each tick, and
[`ofplang-validate`](https://pypi.org/project/ofplang-validate/) used by the CLI
front door. The runner *library* never imports validate (and the per-tick replans
never re-validate), so it stays a one-shot CLI front door.

For development, install editable with the test extra from a clone:

```sh
pip install -e ".[test]"
```

## Command line

```sh
ofp-run run <workflow> --env <env>
    [--boundary DOC] [--boundary-out FILE] [--observation-out FILE]
    [--poll-interval D] [--margin M] [--seed N] [-o OUT]
ofp-run replay <plan> --env <env> [-o OUT]
```

`run` drives a v0 workflow to completion by replanning as it goes: each tick it
renders the committed history as a status, calls the scheduler, and dispatches the
newly-runnable work. `--boundary` supplies the whole-workflow I/O as one document —
a `boundary:` mapping with a `{spot, view}` descriptor per entry input / final
output port. `spot` places a boundary Object on an environment spot (spec §6.8;
Object ports only); `view` supplies an input's view value (unsupplied entry views
default). The runner projects it into the scheduler's interface (spots only, so the
scheduler stays value-independent) and the seeded input values. `--boundary-out`
writes the result boundary — the same schema with each produced output's `view`
filled in — a run-local artifact, separate from the value-free status document. On
completion each pinned Object output is checked to have reached its declared spot.
`--observation-out` streams the **observation document** (see `docs/OBSERVATION.md`):
a YAML multi-document stream recording each *completed* activity's concrete input /
output view values (a transport's moved view), appended as each activity finishes —
the value-layer companion to the status document, also run-local.
`--poll-interval` sets the fixed polling interval (default 1). `replay` runs a plan
produced by `ofp-schedule` verbatim on the simulator (no value layer). Both write
the final execution status as YAML (`-o`, else stdout). Exit codes: `0` success,
`1` execution failed (an activity failed, or a replan is infeasible), `2`
usage/input error.

This tool is also intended to be exposed as the `run` subcommand of the umbrella
`ofp` CLI (a separate repository in the `ofplang` organization).

The package lives under the `ofplang` PEP 420 namespace (`ofplang.run`), shared
across the organization's tools.

## Tests

```sh
pytest
```
