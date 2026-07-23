"""The run boundary document -- the single run-facing I/O format (dev-notes D28).

This unifies the three previously separate run boundaries into one document:

  - the interface spot placement (§6.8) -- where boundary Objects physically sit;
  - the whole-workflow input view values (formerly the ``--job``);
  - the whole-workflow output view values (formerly ``--outputs``).

The format is a ``boundary:`` document with one descriptor per boundary port:

    boundary:
      inputs:
        sample: { spot: loader.stage, view: { barcode: ABC } }
        start:  { view: 42 }
      outputs:
        plate_final: { spot: unloader.slot }   # view filled in at run end
        result: {}                             # view filled in at run end

A port descriptor has two orthogonal halves -- exactly the two halves of a v0
boundary value:

  - ``spot`` is the physical location / linear identity of an Object at the
    boundary (§6.8). Object-bearing ports only; a Pure Data port occupies no spot.
  - ``view`` is the port's view projection (§7.4) -- the observable value. Present
    for inputs (a supplied value; omitted means "use a typed default"); filled in
    by the runner at run end for outputs (the produced value).

The two keys mirror the type system a user already writes: a type's ``view:``
schema names the fields carried in the ``view``, while ``spot`` names an
environment ``<device>.<spot>``.

The runner projects this **run-local** document into the pieces its collaborators
need, and -- critically -- never sends view values to the scheduler:

  - ``interface`` (spots only) -> the §6.8 boundary constraint fed to the
    scheduler each replan, keeping the scheduler value-independent (D9/D26). The
    projection is one-way: view values never round-trip into a replan, so an
    unpinned output can never silently become a scheduling constraint.
  - ``job`` ({port: view}) -> the entry view values seeded into the value store.

At run end the produced output views are echoed back into a result document of
the *same* schema (written by ``--boundary-out``), so a boundary round-trips.

This module is pure: it parses / validates a boundary doc against the resolved
contracts (`contracts.py`) and projects it. Loading files and wiring the
projections into the run are the CLI's and the rolling runner's jobs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .contracts import is_object_bearing
from .runner import RunnerError

# The only keys a port descriptor may carry. Anything else is a typo (e.g. `viwe`,
# `sopt`) and is rejected rather than silently ignored.
_DESCRIPTOR_KEYS = frozenset({"spot", "view"})


@dataclass(frozen=True)
class Boundary:
    """A parsed, validated run boundary, projected for the runner's collaborators.

    `interface` is the §6.8 boundary constraint (spots only) handed to the
    scheduler; `job` is the entry view values to seed; `output_spots` are the
    Object outputs the user pinned to a delivery spot (checked at run end, P3).
    The original input / output descriptors are retained so the produced result
    can be echoed back in the same schema (`result`).
    """

    interface: dict  # {inputs?: {port: spot}, outputs?: {port: spot}} for the scheduler
    job: dict  # {port: view} entry view values to seed (contract-checked at seed time)
    output_spots: dict  # {port: spot} Object outputs with a declared delivery spot (P3)
    _inputs_doc: dict = field(default_factory=dict)  # input descriptors, echoed verbatim
    _outputs_doc: dict = field(default_factory=dict)  # output descriptors (spots kept)

    def result(self, produced: dict) -> dict:
        """Build the result boundary document (same schema) for `--boundary-out`.

        Inputs are echoed verbatim (what the user supplied). Every produced output
        view (`produced`, {port: view}) is echoed under `outputs`, merging in any
        delivery spot the user declared; a declared output that did not run has its
        spot echoed with no view. Ports keep the user's declared order, then any
        extra produced outputs the user did not list (so no produced value is lost).
        """
        out_inputs = copy.deepcopy(self._inputs_doc)
        out_outputs: dict = {}
        # Declared outputs first (in the user's order), then any produced output the
        # user did not list -- so the result never drops a produced value.
        ports = list(self._outputs_doc) + [p for p in produced if p not in self._outputs_doc]
        for port in ports:
            desc = dict(self._outputs_doc.get(port) or {})
            desc.pop("view", None)  # drop any stale input-side view; the run owns it now
            if port in produced:
                desc["view"] = produced[port]  # spot first, then view
            out_outputs[port] = desc
        return {"boundary": {"inputs": out_inputs, "outputs": out_outputs}}


def _descriptor(desc, port: str, side: str) -> tuple:
    """Validate one port descriptor and return `(spot, has_view, view)`.

    A descriptor must be a mapping carrying only `spot` / `view` (an unknown key is
    a typo, rejected). `has_view` distinguishes an omitted `view` (default) from a
    supplied one."""
    if not isinstance(desc, dict):
        raise RunnerError(f"boundary {side} {port!r} must be a mapping with spot / view")
    unknown = set(desc) - _DESCRIPTOR_KEYS
    if unknown:
        raise RunnerError(f"boundary {side} {port!r} has unknown key(s): {', '.join(sorted(unknown))}")
    return desc.get("spot"), ("view" in desc), desc.get("view")


def parse_boundary(doc, contracts) -> Boundary:
    """Parse and validate a boundary document against the resolved contracts.

    `doc` is the loaded YAML (a mapping with a `boundary:` root), or None for no
    boundary (all defaults). Returns a `Boundary` projecting the scheduler
    interface, the seed job, and the pinned output spots. Raises `RunnerError` on
    an unknown boundary port, an Object input with no spot, or a Pure Data port
    given a spot -- surfacing an authoring mistake up front rather than mid-run.

    The value carried in an input `view` is not contract-checked here; that happens
    when it is seeded (`values.seed_entry`), which owns the value-conformance check.
    An output descriptor's `view` (if any) is ignored -- outputs are produced, not
    supplied -- so a result document round-trips as an input document unchanged.
    """
    if doc is None:
        return Boundary({}, {}, {})
    if not isinstance(doc, dict):
        raise RunnerError("boundary document must be a mapping")
    boundary = doc.get("boundary")
    if boundary is None:
        boundary = {}
    if not isinstance(boundary, dict):
        raise RunnerError("boundary: must be a mapping")
    inputs_doc = boundary.get("inputs") or {}
    outputs_doc = boundary.get("outputs") or {}
    if not isinstance(inputs_doc, dict) or not isinstance(outputs_doc, dict):
        raise RunnerError("boundary inputs / outputs must be mappings")

    # The entry composite's declared ports are the boundary ports; their resolved
    # types classify each as Object-bearing (needs a spot) or Pure Data (no spot).
    entry = contracts.entry
    entry_inputs = contracts.processes[entry].inputs if entry is not None else {}
    entry_outputs = contracts.processes[entry].outputs if entry is not None else {}

    # Inputs: an Object input must name a spot; a Pure Data input must not. A
    # supplied `view` becomes a job value (checked later at seed time).
    interface_inputs: dict = {}
    job: dict = {}
    for port, desc in inputs_doc.items():
        if port not in entry_inputs:
            raise RunnerError(f"boundary input {port!r} is not an entry input of the workflow")
        spot, has_view, view = _descriptor(desc, port, "input")
        obj = is_object_bearing(entry_inputs[port])
        if obj and spot is None:
            raise RunnerError(f"boundary input {port!r} is Object-bearing and must name a spot")
        if not obj and spot is not None:
            raise RunnerError(f"boundary input {port!r} is Pure Data and occupies no spot")
        if spot is not None:
            interface_inputs[port] = spot
        if has_view:
            job[port] = view

    # Outputs: a delivery spot is optional (an unpinned Object output stays where it
    # is produced), but a Pure Data output can carry none. A supplied `view` here is
    # ignored -- the run produces it.
    interface_outputs: dict = {}
    for port, desc in outputs_doc.items():
        if port not in entry_outputs:
            raise RunnerError(f"boundary output {port!r} is not a final output of the workflow")
        spot, _has_view, _view = _descriptor(desc, port, "output")
        obj = is_object_bearing(entry_outputs[port])
        if not obj and spot is not None:
            raise RunnerError(f"boundary output {port!r} is Pure Data and occupies no spot")
        if spot is not None:
            interface_outputs[port] = spot

    # The scheduler interface (§6.8) carries only the non-empty spot maps; an empty
    # interface is omitted entirely (a workflow with no boundary Objects), matching
    # the pre-boundary behaviour where no interface was supplied.
    interface: dict = {}
    if interface_inputs:
        interface["inputs"] = interface_inputs
    if interface_outputs:
        interface["outputs"] = interface_outputs

    # `output_spots` == the pinned Object outputs (Pure Data + spot already errored),
    # kept for the run-end delivery check (P3).
    return Boundary(interface, job, dict(interface_outputs), dict(inputs_doc), dict(outputs_doc))
