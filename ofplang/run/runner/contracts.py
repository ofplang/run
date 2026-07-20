"""Resolved type / view contracts for the value layer (dev-notes design.md D27, F1).

The value layer needs to know, for each process port, its resolved type: whether
it is Object-bearing (§5.2), and -- for typed value generation and contract
checking in later stages -- its view schema (§7.4), the contract-visible Pure
Data projection of the value. This module resolves that from the workflow.

Independence (D27): the runner does NOT depend on ofplang-validate. This is a
small, self-contained resolver specialised to what the runner needs (resolve a
port type, its domain / Object-bearing-ness, and its flat view schema). It works
on plain dicts (like the scheduler's reader), pulls in no validate internals, and
is deliberately promotable to a shared `ofplang-types` later. It assumes valid v0
input -- shape / reference / phase checking is validate's job, not duplicated here.

Scope (D27, F1): concrete types only -- primitives (Bool/Int/Float/String),
`Array<T>`, and nominal data / object types with a flat view schema. Generics
(§6), traits (§7.3), phase (§5.6), and `$import` (§5.7) are out of scope; a
generic or unknown type raises. A port's runtime "value" is its type's view
projection: a scalar for a primitive, a list for an Array, a field dict for a
nominal (empty when the type declares no view). Object-bearing ports carry a view
projection too; their linear identity is tracked separately by the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from .runner import RunnerError

# v0 built-in primitive Data types (§7.1). They have no view fields; a primitive
# value is its own contract-visible projection.
PRIMITIVES = frozenset({"Bool", "Int", "Float", "String"})

# The default value per primitive, used to synthesise a runner-side typed value
# (an unsupplied entry input, an unconnected input). Mirrors the backend's own
# defaults in the simulator (they must agree; a test checks conformance).
_PRIMITIVE_DEFAULTS = {"Bool": False, "Int": 0, "Float": 0.0, "String": ""}


# -- resolved type model -----------------------------------------------------
#
# A resolved type is one of Primitive / ArrayType / Nominal. `ResolvedType` is
# their union (used only for annotation / documentation).


@dataclass(frozen=True)
class Primitive:
    """A built-in primitive Data type (§7.1); its value is a scalar of this kind."""

    name: str  # one of PRIMITIVES


@dataclass(frozen=True)
class ArrayType:
    """`Array<T>` (§7.1); its value is a list of `element`-shaped values. Object-
    bearing iff its element is (§7.1 / §5.2)."""

    element: "ResolvedType"


@dataclass(frozen=True)
class Nominal:
    """A user-defined nominal type (§7.2): `domain` is "data" or "object", and
    `view` is its flat view schema (§7.4) -- field name -> resolved field type,
    each a Primitive or an Array (recursively) of primitives. A nominal with no
    declared view has an empty `view`. Its value is a dict of the view fields; an
    object nominal additionally has a linear identity, tracked by the simulator,
    not carried here."""

    name: str
    domain: str  # "data" | "object"
    view: dict = field(default_factory=dict)  # field name -> ResolvedType


ResolvedType = "Primitive | ArrayType | Nominal"


def is_object_bearing(resolved) -> bool:
    """Whether a value of this resolved type carries an Object slot (§5.2): an
    object nominal, or an Array (recursively) whose element is Object-bearing.
    Primitives and data nominals are Pure Data."""
    if isinstance(resolved, Primitive):
        return False
    if isinstance(resolved, ArrayType):
        return is_object_bearing(resolved.element)
    return resolved.domain == "object"  # Nominal


def conforms(value, resolved) -> bool:
    """Whether `value` conforms to `resolved` as a view value (D27, F3).

    A value is the type's view projection, so conformance is checked structurally:
      - a primitive is a Python scalar of its kind -- Bool is `bool`, Int is `int`
        (not `bool`), Float is `int`/`float` (not `bool`, JSON-lenient), String is `str`;
      - an Array is a list whose every element conforms to the element type;
      - a nominal is a dict with *exactly* the view field names, each conforming to
        its field type (an empty view requires `{}`). Object nominals are checked the
        same way -- only the view projection; their identity is the simulator's concern.
    This is the checker primitive; F4 wires it where external values (a supplied job,
    routed inputs) can actually be non-conformant. The runner's own generated defaults
    (F2) always conform."""
    if isinstance(resolved, Primitive):
        name = resolved.name
        if name == "Bool":
            return isinstance(value, bool)
        if name == "Int":
            return isinstance(value, int) and not isinstance(value, bool)
        if name == "Float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if name == "String":
            return isinstance(value, str)
        return False  # not a v0 primitive
    if isinstance(resolved, ArrayType):
        return isinstance(value, list) and all(conforms(item, resolved.element) for item in value)
    # Nominal: the value is its view record -- exactly the view fields, each conforming.
    if not isinstance(value, dict):
        return False
    if set(value) != set(resolved.view):
        return False
    return all(conforms(value[field], field_type) for field, field_type in resolved.view.items())


def default_value(resolved):
    """A typed default value for `resolved`, mirroring the backend's generator
    (D27 F2): a primitive's default, an empty array, or a record of its view
    fields' defaults. Used runner-side to synthesise a value the runner is
    responsible for -- an entry input the job did not supply, or an unconnected
    input -- as a conformant view value (F4). The result always `conforms`."""
    if isinstance(resolved, Primitive):
        return _PRIMITIVE_DEFAULTS[resolved.name]
    if isinstance(resolved, ArrayType):
        return []
    return {field: default_value(field_type) for field, field_type in resolved.view.items()}


# -- resolution --------------------------------------------------------------


def _parse(expr: str, registry: dict) -> ResolvedType:
    """Resolve one v0 type expression string against the nominal `registry`.

    Concrete types only (D27): a primitive, `Array<...>` (possibly nested), or a
    known nominal. Anything else -- a generic type parameter, an unknown name --
    raises, since the runner assumes valid, concrete v0 input."""
    expr = expr.strip()
    if expr.startswith("Array<") and expr.endswith(">"):
        return ArrayType(_parse(expr[len("Array<"):-1], registry))
    if expr in PRIMITIVES:
        return Primitive(expr)
    nominal = registry.get(expr)
    if nominal is None:
        raise RunnerError(f"unknown or unsupported type expression: {expr!r}")
    return nominal


def _build_registry(types_section: dict) -> dict:
    """Resolve the document's `types:` section (§7.2) into a name -> Nominal map.

    Two passes so a view field can reference any declared type by name: first
    create each nominal with an empty view (recording its domain), then resolve
    each nominal's view fields against the now-complete registry. (In valid v0 a
    view field is a primitive or Array of primitives, §7.4, so this never cycles.)"""
    registry: dict = {}
    for name, spec in (types_section or {}).items():
        spec = spec or {}
        registry[name] = Nominal(name, spec.get("domain"), view={})
    for name, spec in (types_section or {}).items():
        view_raw = (spec or {}).get("view") or {}
        view = {f: _parse((fs or {}).get("type", ""), registry) for f, fs in view_raw.items()}
        registry[name] = replace(registry[name], view=view)
    return registry


@dataclass(frozen=True)
class ProcessContract:
    """One process's resolved port signature: port name -> resolved type, for its
    declared inputs and outputs."""

    inputs: dict
    outputs: dict


class Contracts:
    """The workflow's resolved port contracts (D27, F1).

    `processes` maps each process name (atomic and composite alike) to its
    `ProcessContract`; `types` is the resolved nominal registry. Callers marry this
    to the flattened graph via the dataflow adapter's `process_of` (node -> process)
    -- keeping types process-keyed (they are declared per process) rather than
    duplicating them per node.
    """

    def __init__(self, processes: dict, types: dict, entry: str | None = None) -> None:
        self.processes = processes
        self.types = types
        # The entry (top composite) process name; its declared inputs are the
        # workflow's boundary input ports (their types drive the job / seed values).
        self.entry = entry

    @classmethod
    def from_workflow(cls, workflow_path) -> "Contracts":
        """Resolve the contracts of the workflow at `workflow_path` (single file;
        `$import` is out of F1 scope). Assumes valid v0 input."""
        data = yaml.safe_load(Path(workflow_path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RunnerError("workflow must be a mapping")
        registry = _build_registry(data.get("types") or {})
        raw_processes = data.get("processes") or {}
        processes: dict = {}
        for name, spec in raw_processes.items():
            spec = spec or {}
            inputs = {p: _parse((ps or {}).get("type", ""), registry) for p, ps in (spec.get("inputs") or {}).items()}
            outputs = {p: _parse((ps or {}).get("type", ""), registry) for p, ps in (spec.get("outputs") or {}).items()}
            processes[name] = ProcessContract(inputs, outputs)
        entry = data.get("entry") or ("main" if "main" in raw_processes else None)
        return cls(processes, registry, entry)

    def entry_input_type(self, port: str) -> ResolvedType:
        """The resolved type of an entry (boundary) input port."""
        return self.processes[self.entry].inputs[port]

    def input_type(self, process: str, port: str) -> ResolvedType:
        return self.processes[process].inputs[port]

    def output_type(self, process: str, port: str) -> ResolvedType:
        return self.processes[process].outputs[port]


# -- value-shape descriptor (D27, F2) ----------------------------------------
#
# The seam between the runner (which resolves types) and the backend (which
# generates values, per D26 principle B). Rather than share this module's type
# model with the simulator, the runner converts a resolved type into a neutral,
# serialisable descriptor and passes it in the dispatch signature; the backend
# walks the descriptor to generate a value, importing nothing from here. The
# descriptor is the extensible wire (F4 will describe inputs the same way).


def to_descriptor(resolved) -> dict:
    """Convert a resolved type into a neutral value-shape descriptor. Shapes:

        {"kind": "primitive", "name": "Bool" | "Int" | "Float" | "String"}
        {"kind": "array", "element": <descriptor>}
        {"kind": "record", "fields": {name: <descriptor>, ...}}

    A nominal's value is its view record -- a record of its view fields, empty when
    it declares no view. The domain (data / object) does not affect the value
    shape: an object port carries a view record like a data one, its identity being
    tracked separately by the simulator (D27)."""
    if isinstance(resolved, Primitive):
        return {"kind": "primitive", "name": resolved.name}
    if isinstance(resolved, ArrayType):
        return {"kind": "array", "element": to_descriptor(resolved.element)}
    # Nominal -> a record of its (already resolved) view fields.
    return {"kind": "record", "fields": {f: to_descriptor(t) for f, t in resolved.view.items()}}
