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
    conforms,
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


# -- conformance (F3) --------------------------------------------------------


def test_conforms_primitives_with_bool_int_float_distinctions():
    assert conforms(True, Primitive("Bool"))
    assert not conforms(1, Primitive("Bool"))          # int is not Bool
    assert conforms(3, Primitive("Int"))
    assert not conforms(True, Primitive("Int"))        # bool is not Int
    assert not conforms(3.0, Primitive("Int"))         # float is not Int
    assert conforms(3.0, Primitive("Float"))
    assert conforms(3, Primitive("Float"))             # int accepted for Float (JSON-lenient)
    assert not conforms(True, Primitive("Float"))      # bool is not Float
    assert conforms("x", Primitive("String"))
    assert not conforms(1, Primitive("String"))


def test_conforms_arrays_check_elements():
    t = ArrayType(Primitive("Int"))
    assert conforms([], t)                 # empty conforms
    assert conforms([1, 2, 3], t)
    assert not conforms([1, "x"], t)       # a bad element
    assert not conforms("nope", t)         # not a list
    # Nested arrays.
    assert conforms([[1], [2, 3]], ArrayType(ArrayType(Primitive("Int"))))


def test_conforms_records_need_exactly_the_view_fields():
    reading = Nominal("Reading", "data", {"mean": Primitive("Float"), "n": Primitive("Int")})
    assert conforms({"mean": 0.0, "n": 0}, reading)
    assert conforms({"mean": 1, "n": 2}, reading)        # int ok for Float
    assert not conforms({"mean": 0.0}, reading)          # missing field
    assert not conforms({"mean": 0.0, "n": 0, "x": 1}, reading)  # extra field
    assert not conforms({"mean": "bad", "n": 0}, reading)        # bad field value
    assert not conforms(["mean", "n"], reading)          # not a dict


def test_conforms_empty_view_and_object_nominal():
    # A no-view nominal requires an empty dict; object nominals are checked the same
    # way (view only -- identity is not part of the value).
    assert conforms({}, Nominal("Tube", "object", {}))
    assert not conforms({"x": 1}, Nominal("Tube", "object", {}))
    plate = Nominal("Plate96", "object", {"well_count": Primitive("Int")})
    assert conforms({"well_count": 96}, plate)
    assert not conforms({"well_count": 96.0}, plate)     # float is not Int


def test_conforms_nested_record_with_array_field():
    t = Nominal("Batch", "data", {"labels": ArrayType(Primitive("String")), "size": Primitive("Int")})
    assert conforms({"labels": ["a", "b"], "size": 2}, t)
    assert not conforms({"labels": [1], "size": 2}, t)


def test_generated_defaults_conform(tmp_path):
    # The values F2's backend generates (schema defaults) satisfy the checker -- the
    # generator and checker agree.
    c = _contracts(tmp_path)
    assert conforms(0.0, c.output_type("p", "score"))
    assert conforms([], c.input_type("p", "flags"))
    assert conforms({"mean": 0.0, "n": 0}, c.input_type("p", "r"))
    assert conforms({}, c.output_type("p", "tube"))


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
