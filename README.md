# ofplang run

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
>   transporters, and timed operations advanced on a virtual clock. It validates
>   every dispatch (an inconsistent plan is rejected), models timed device up/down
>   and injected operation failure, and at completion produces each operation's
>   output view values via a **device model** (the built-in `default_device_model`
>   fills type defaults and carries Object outputs through from their `objects.map`
>   inputs; a custom / real model computes them).
> - **Runner** (`ofplang.run.runner`) — two ways to drive a backend:
>   - **`replay`** runs a given execution plan (spec §6) on the backend verbatim.
>   - **`run`** is a rolling-horizon loop: it calls
>     [`ofplang.schedule`](https://github.com/ofplang/schedule) each tick,
>     dispatches the work that can start now, advances the clock, and polls —
>     replanning from the committed history as it goes. It re-routes around a
>     downed device, polls at a fixed interval with completion-time estimation, and
>     stops the whole run if any activity fails (marking the abandoned work
>     cancelled).
> - **Value layer** — the runner resolves each port's type and view schema (§7),
>   routes typed view values along the workflow's arcs (producer output → consumer
>   input, across nested composites), contract-checks them, and assembles the
>   whole-workflow outputs. A caller supplies the entry values as a **job**
>   (`--job`); unsupplied entries default. Values are typed but still dummy — a real
>   device backend plugs into the same seam later.

## Install

```sh
pip install -e ".[test]"
```

Requires Python 3.10+. `replay` needs only PyYAML. `run` (rolling-horizon) calls
the sibling [`ofplang-schedule`](https://github.com/ofplang/schedule); install it
editable alongside this repo:

```sh
pip install -e ../ofplang-schedule
```

## Command line

```sh
ofp-run run <workflow> --env <env>
    [--interface DOC] [--job DOC] [--outputs FILE]
    [--poll-interval D] [--margin M] [--seed N] [-o OUT]
ofp-run replay <plan> --env <env> [-o OUT]
```

`run` drives a v0 workflow to completion by replanning as it goes: each tick it
renders the committed history as a status, calls the scheduler, and dispatches the
newly-runnable work. `--interface` supplies the boundary spots for a workflow with
Object-bearing entry inputs (spec §6.8); `--job` supplies the whole-workflow input
values (`{entry_port: view value}`, contract-checked; unsupplied entries default);
`--outputs` writes the whole-workflow output values to a YAML file (a run-local
artifact, separate from the status document); `--poll-interval` sets the fixed
polling interval (default 1). `replay` runs a plan produced by `ofp-schedule`
verbatim on the simulator (no value layer). Both write the final execution status
as YAML (`-o`, else stdout). Exit codes: `0` success, `1` execution failed (an
activity failed, or a replan is infeasible), `2` usage/input error.

This tool is also intended to be exposed as the `run` subcommand of the umbrella
`ofp` CLI (a separate repository in the `ofplang` organization).

The package lives under the `ofplang` PEP 420 namespace (`ofplang.run`), shared
across the organization's tools.

## Tests

```sh
pytest
```
