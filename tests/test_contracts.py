"""Unit tests for the resolved type / view contracts (dev-notes design.md D27, F1).

These pin the resolver the value layer's later stages build on: type-expression
parsing, domain / Object-bearing resolution (§5.2), and flat view schemas (§7.4),
for data and object types alike -- all self-contained, no ofplang-validate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ofplang.run.runner.contracts import (
    ArrayType,
    Contracts,
    Nominal,
    Primitive,
    is_object_bearing,
    to_descriptor,
)
from ofplang.run.runner.runner import RunnerError

FIXTURES = Path(__file__).parent / "fixtures"

# A workflow exercising primitives, arrays (incl. of an object type), a data type
# with a view, an object type with a view, and an object type with no view.
_WORKFLOW = """\
spec_version: "0.0"
types:
  Reading: {domain: data, view: {mean: {type: Float}, n: {type: Int}}}
  Plate96: {domain: object, view: {well_count: {type: Int}, barcode: {type: String}}}
  Tube:    {domain: object}
processes:
  p:
    kind: atomic
    inputs:
      r:     {type: Reading}
      plate: {type: Plate96}
      flags: {type: Array<Bool>}
    outputs:
      score:  {type: Float}
      plates: {type: Array<Plate96>}
      tube:   {type: Tube}
"""


def _contracts(tmp_path):
    doc = tmp_path / "wf.yaml"
    doc.write_text(_WORKFLOW, encoding="utf-8")
    return Contracts.from_workflow(doc)


def test_primitive_and_array_ports(tmp_path):
    c = _contracts(tmp_path)
    assert c.output_type("p", "score") == Primitive("Float")
    assert c.input_type("p", "flags") == ArrayType(Primitive("Bool"))
    # Nested arrays parse recursively (used later, but resolve now).
    from ofplang.run.runner.contracts import _parse

    assert _parse("Array<Array<Int>>", {}) == ArrayType(ArrayType(Primitive("Int")))


def test_data_nominal_resolves_its_view_schema(tmp_path):
    c = _contracts(tmp_path)
    reading = c.input_type("p", "r")
    assert reading == Nominal("Reading", "data", {"mean": Primitive("Float"), "n": Primitive("Int")})
    assert not is_object_bearing(reading)


def test_object_nominal_is_object_bearing_and_keeps_its_view(tmp_path):
    c = _contracts(tmp_path)
    plate = c.input_type("p", "plate")
    assert plate.domain == "object"
    assert plate.view == {"well_count": Primitive("Int"), "barcode": Primitive("String")}
    assert is_object_bearing(plate)  # an object nominal carries an Object slot


def test_object_type_without_view_has_empty_view(tmp_path):
    c = _contracts(tmp_path)
    tube = c.output_type("p", "tube")
    assert tube == Nominal("Tube", "object", {})
    assert is_object_bearing(tube)


def test_array_of_object_type_is_object_bearing(tmp_path):
    c = _contracts(tmp_path)
    plates = c.output_type("p", "plates")
    assert plates == ArrayType(Nominal("Plate96", "object", {
        "well_count": Primitive("Int"), "barcode": Primitive("String"),
    }))
    assert is_object_bearing(plates)  # Array<T> is Object-bearing iff T is


def test_object_bearing_of_every_shape():
    assert not is_object_bearing(Primitive("Int"))
    assert not is_object_bearing(ArrayType(Primitive("String")))
    assert not is_object_bearing(Nominal("D", "data", {}))
    assert is_object_bearing(Nominal("O", "object", {}))
    assert is_object_bearing(ArrayType(ArrayType(Nominal("O", "object", {}))))


def test_unknown_type_raises(tmp_path):
    # A generic type parameter / unknown name is out of F1 scope -> error.
    doc = tmp_path / "bad.yaml"
    doc.write_text(
        "processes:\n"
        "  p: {kind: atomic, inputs: {x: {type: T}}}\n",
        encoding="utf-8",
    )
    with pytest.raises(RunnerError):
        Contracts.from_workflow(doc)


def test_to_descriptor_produces_neutral_value_shape(tmp_path):
    # The dispatch-signature wire (D27 F2): a resolved type -> a neutral, serialisable
    # value-shape descriptor the backend can walk without the type model.
    c = _contracts(tmp_path)
    assert to_descriptor(c.output_type("p", "score")) == {"kind": "primitive", "name": "Float"}
    assert to_descriptor(c.input_type("p", "flags")) == {
        "kind": "array",
        "element": {"kind": "primitive", "name": "Bool"},
    }
    # A nominal -> a record of its view fields (an object nominal too; domain does
    # not affect the value shape).
    assert to_descriptor(c.input_type("p", "r")) == {
        "kind": "record",
        "fields": {"mean": {"kind": "primitive", "name": "Float"}, "n": {"kind": "primitive", "name": "Int"}},
    }
    assert to_descriptor(c.output_type("p", "tube")) == {"kind": "record", "fields": {}}  # no view


def test_resolves_a_real_fixture(tmp_path):
    # The nested_returns fixture: nominal types with no view fields resolve to empty
    # views, with the right domains.
    c = Contracts.from_workflow(FIXTURES / "nested_returns.workflow.yaml")
    assert c.output_type("measure", "plate_out") == Nominal("Plate", "object", {})
    assert c.output_type("measure", "reading") == Nominal("Reading", "data", {})
    assert c.output_type("analyze", "score") == Nominal("Score", "data", {})
    # main is a composite; its declared ports resolve too (used for boundary later).
    assert c.input_type("main", "sample") == Nominal("Plate", "object", {})
    assert c.output_type("main", "final_score") == Nominal("Score", "data", {})
