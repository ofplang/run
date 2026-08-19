"""The front door over an in-memory workflow document (R8-1).

Since S2/R6 the recommended route for an embedding caller is to hand the scheduler and
the runner documents rather than paths -- but the front door took only a path, so that
route was the one nobody checked: `run_workflow(dict, validate=True)` refused outright,
and under `validate=False` a malformed document reached the runner and failed deep.
These pin that the two routes now agree, and how the two things a document can be that
a file cannot -- unexpanded, or holding a value YAML cannot spell -- are answered.
"""

from __future__ import annotations

import copy
import datetime
from pathlib import Path

import pytest
import yaml

from ofplang.run.app import UNEXPANDED_IMPORT, front_door_check

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = FIXTURES / "simple.workflow.yaml"


def _document() -> dict:
    return yaml.safe_load(SIMPLE_WF.read_text(encoding="utf-8"))


@pytest.mark.parametrize("validate", [True, False])
def test_a_valid_document_passes(validate: bool) -> None:
    fd = front_door_check(_document(), validate=validate)
    assert fd.ok, [d.code for d in fd.diagnostics]
    assert fd.document == _document()


def test_the_document_route_and_the_file_route_agree() -> None:
    """The point of R8-1: the same workflow gets the same verdict either way."""
    from_file = front_door_check(str(SIMPLE_WF))
    from_doc = front_door_check(_document())
    assert from_doc.ok == from_file.ok
    assert [d.code for d in from_doc.diagnostics] == [
        d.code for d in from_file.diagnostics
    ]
    assert from_doc.document == from_file.document


def test_a_malformed_document_is_rejected_with_diagnostics() -> None:
    doc = _document()
    doc["types"]["Sample"] = {"domain": "bogus"}
    fd = front_door_check(doc)
    assert not fd.ok
    assert fd.diagnostics
    # There is no file to point into, so a finding locates itself by logical path.
    assert all(d.file is None and d.line is None for d in fd.diagnostics)
    assert any(d.path for d in fd.diagnostics)


def test_the_capability_gate_still_fires_on_a_document() -> None:
    doc = _document()
    doc["processes"]["gen"] = {
        "kind": "atomic",
        "type_params": {"O": {"domain": "object"}},
        "inputs": {},
        "outputs": {},
    }
    fd = front_door_check(doc, validate=False)  # isolate the gate
    assert not fd.ok
    assert fd.unsupported is not None and "gen" in fd.unsupported


@pytest.mark.parametrize("validate", [True, False])
def test_an_unexpanded_document_is_unsupported_either_way(validate: bool) -> None:
    """A relative `$import` cannot be resolved without a base directory, so it is
    refused -- and refused the same way whether or not the validate pass runs, even
    though validate itself raises on it."""
    doc = _document()
    doc["types"] = {"$import": "types.yaml"}
    fd = front_door_check(doc, validate=validate)
    assert not fd.ok
    assert fd.unsupported == UNEXPANDED_IMPORT
    assert not fd.diagnostics


def test_a_value_yaml_cannot_spell_is_a_caller_error() -> None:
    """Not a finding: the front door cannot call a document valid while checking a
    stringified stand-in for it. validate names the position; the error propagates."""
    doc = _document()
    doc["when"] = datetime.date(2026, 8, 20)
    with pytest.raises(ValueError, match="when"):
        front_door_check(doc)


def test_the_document_is_not_mutated() -> None:
    doc = _document()
    before = copy.deepcopy(doc)
    front_door_check(doc)
    front_door_check(doc, validate=False)
    assert doc == before


def test_validate_false_hands_back_the_given_document() -> None:
    """Nothing to expand, so nothing is copied: the caller's document is what runs
    (the D30 convention -- input read-only). Under `validate=True` the document is
    validate's own plain copy instead, which is why this is asserted, not assumed."""
    doc = _document()
    assert front_door_check(doc, validate=False).document is doc
    assert front_door_check(doc, validate=True).document is not doc
