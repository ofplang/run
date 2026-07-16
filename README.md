# ofplang run

A runner for **Object-flow Programming Language v0** — a YAML-based dataflow
workflow IR with linear Object tracking. The language is defined in the
[ofplang/spec](https://github.com/ofplang/spec) repository.

The runner takes an execution plan — the artefact produced by
[`ofplang.schedule`](https://github.com/ofplang) (spec §6) — and drives its
activities to completion against an execution backend, emitting an execution
status/document as it progresses.

> **Status:** scaffold. Only the package layout and the `ofp-run` CLI skeleton
> exist; no runner logic is implemented yet. It is being built in two steps:
>
> 1. **Simulator** (`ofplang.run.simulator`) — a simulated execution backend so
>    the runner can be exercised end to end without real hardware.
> 2. **Runner** (`ofplang.run.runner`) — walks a plan and dispatches its
>    activities to a backend (the simulator first, real hardware later).

## Install

```sh
pip install -e ".[test]"
```

Requires Python 3.10+. The only runtime dependency so far is PyYAML.

## Command line

```sh
ofp-run run <plan>      # drive an execution plan to completion (not implemented yet)
```

`run` will consume an execution plan produced by `ofp-schedule` and run it to
completion. Exit codes: `0` success, `1` execution failed, `2` usage/input
error.

This tool is also intended to be exposed as the `run` subcommand of the umbrella
`ofp` CLI (a separate repository in the `ofplang` organization).

The package lives under the `ofplang` PEP 420 namespace (`ofplang.run`), shared
across the organization's tools.

## Tests

```sh
pytest
```
