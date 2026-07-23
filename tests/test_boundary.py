"""Unit tests for the run boundary parser / projector (dev-notes design.md D28).

These are pure: the boundary module and the contracts it validates against read
the workflow directly and pull in no scheduler, so no `importorskip` is needed.
They pin the projection (a `boundary:` doc -> the scheduler interface + the seed
job + the pinned output spots) and the result echo, and the validation errors that
surface an authoring mistake up front.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ofplang.run.runner.boundary import Boundary, parse_boundary
from ofplang.run.runner.contracts import Contracts
from ofplang.run.runner.runner import RunnerError

FIXTURES = Path(__file__).parent / "fixtures"

# typed_returns: Object input `sample` (Plate, view {barcode}); Pure Data output
# `final_score` (Score). count_chain: Pure Data input `start` + output `result`
# (Count, view {value}); no Objects. interface_load: Object input `sample` and
# Object output `result` (Plate, no view).
TYPED = str(FIXTURES / "typed_returns.workflow.yaml")
COUNT = str(FIXTURES / "count_chain.workflow.yaml")
LOAD = str(FIXTURES / "interface_load.workflow.yaml")


def _contracts(path):
    return Contracts.from_workflow(path)


# -- projection --------------------------------------------------------------


def test_object_input_projects_spot_and_view():
    """An Object input's spot goes to the scheduler interface; its view to the job."""
    doc = {"boundary": {"inputs": {"sample": {"spot": "loader.stage", "view": {"barcode": "ABC"}}}}}
    b = parse_boundary(doc, _contracts(TYPED))
    assert b.interface == {"inputs": {"sample": "loader.stage"}}
    assert b.job == {"sample": {"barcode": "ABC"}}
    assert b.output_spots == {}


def test_pure_data_input_projects_view_only():
    """A Pure Data input carries no spot: nothing reaches the interface, only the job."""
    doc = {"boundary": {"inputs": {"start": {"view": {"value": 42}}}}}
    b = parse_boundary(doc, _contracts(COUNT))
    assert b.interface == {}  # no Objects -> no interface at all
    assert b.job == {"start": {"value": 42}}


def test_input_view_omitted_is_not_seeded():
    """An input with a spot but no view supplies no job value (it defaults later)."""
    doc = {"boundary": {"inputs": {"sample": {"spot": "loader.stage"}}}}
    b = parse_boundary(doc, _contracts(TYPED))
    assert b.interface == {"inputs": {"sample": "loader.stage"}}
    assert b.job == {}  # view omitted -> seed_entry will default it


def test_object_output_projects_spot():
    """An Object output's delivery spot goes to the interface and to `output_spots`
    (the run-end delivery check, P3)."""
    doc = {"boundary": {"outputs": {"result": {"spot": "unloader.slot"}}}}
    b = parse_boundary(doc, _contracts(LOAD))
    assert b.interface == {"outputs": {"result": "unloader.slot"}}
    assert b.output_spots == {"result": "unloader.slot"}


def test_output_view_on_input_is_ignored():
    """A `view` on an input-side output descriptor is ignored (outputs are produced),
    so a result document round-trips as an input document."""
    doc = {"boundary": {"outputs": {"result": {"spot": "unloader.slot", "view": {"barcode": "STALE"}}}}}
    b = parse_boundary(doc, _contracts(LOAD))
    assert b.output_spots == {"result": "unloader.slot"}
    assert b.job == {}  # nothing from the output view leaks into the seed


def test_unpinned_object_output_is_allowed():
    """An Object output need not be pinned (it stays where produced); no interface,
    no P3 check for it."""
    doc = {"boundary": {"outputs": {"result": {}}}}
    b = parse_boundary(doc, _contracts(LOAD))
    assert b.interface == {}
    assert b.output_spots == {}


def test_empty_and_absent_boundary():
    """None, an empty document, and a document without a `boundary:` key all yield an
    empty boundary (all defaults)."""
    for doc in (None, {}, {"boundary": {}}, {"boundary": {"inputs": {}, "outputs": {}}}):
        b = parse_boundary(doc, _contracts(COUNT))
        assert b.interface == {} and b.job == {} and b.output_spots == {}


# -- validation --------------------------------------------------------------


def test_object_input_without_spot_errors():
    doc = {"boundary": {"inputs": {"sample": {"view": {"barcode": "ABC"}}}}}
    with pytest.raises(RunnerError, match="Object-bearing and must name a spot"):
        parse_boundary(doc, _contracts(TYPED))


def test_pure_data_input_with_spot_errors():
    doc = {"boundary": {"inputs": {"start": {"spot": "somewhere.slot"}}}}
    with pytest.raises(RunnerError, match="Pure Data and occupies no spot"):
        parse_boundary(doc, _contracts(COUNT))


def test_pure_data_output_with_spot_errors():
    doc = {"boundary": {"outputs": {"final_score": {"spot": "somewhere.slot"}}}}
    with pytest.raises(RunnerError, match="Pure Data and occupies no spot"):
        parse_boundary(doc, _contracts(TYPED))


def test_unknown_input_port_errors():
    doc = {"boundary": {"inputs": {"nope": {"view": {"value": 1}}}}}
    with pytest.raises(RunnerError, match="not an entry input"):
        parse_boundary(doc, _contracts(COUNT))


def test_unknown_output_port_errors():
    doc = {"boundary": {"outputs": {"nope": {}}}}
    with pytest.raises(RunnerError, match="not a final output"):
        parse_boundary(doc, _contracts(COUNT))


def test_unknown_descriptor_key_errors():
    doc = {"boundary": {"inputs": {"start": {"vieww": {"value": 1}}}}}
    with pytest.raises(RunnerError, match="unknown key"):
        parse_boundary(doc, _contracts(COUNT))


def test_non_mapping_descriptor_errors():
    doc = {"boundary": {"inputs": {"start": 42}}}
    with pytest.raises(RunnerError, match="must be a mapping"):
        parse_boundary(doc, _contracts(COUNT))


# -- result echo -------------------------------------------------------------


def test_result_echoes_inputs_and_fills_output_views():
    """`result` echoes inputs verbatim and fills each produced output view, merging a
    declared delivery spot."""
    doc = {
        "boundary": {
            "inputs": {"sample": {"spot": "loader.stage", "view": {"barcode": "ABC"}}},
            "outputs": {"result": {"spot": "unloader.slot"}},
        }
    }
    b = parse_boundary(doc, _contracts(LOAD))
    result = b.result({"result": {"barcode": "ABC"}})
    assert result == {
        "boundary": {
            "inputs": {"sample": {"spot": "loader.stage", "view": {"barcode": "ABC"}}},
            "outputs": {"result": {"spot": "unloader.slot", "view": {"barcode": "ABC"}}},
        }
    }


def test_result_includes_undeclared_produced_outputs():
    """A produced output the user did not list is still echoed (no value is lost)."""
    b = parse_boundary({"boundary": {}}, _contracts(COUNT))
    result = b.result({"result": {"value": 44}})
    assert result["boundary"]["outputs"] == {"result": {"view": {"value": 44}}}


def test_result_declared_output_that_did_not_run_keeps_spot_without_view():
    """A declared Object output that never produced a value keeps its spot, no view."""
    doc = {"boundary": {"outputs": {"result": {"spot": "unloader.slot"}}}}
    b = parse_boundary(doc, _contracts(LOAD))
    result = b.result({})  # nothing produced (e.g. the run failed before delivery)
    assert result["boundary"]["outputs"] == {"result": {"spot": "unloader.slot"}}


def test_result_does_not_mutate_input_descriptors():
    """Echoing must not mutate the caller's parsed descriptors (deep-copied)."""
    doc = {"boundary": {"inputs": {"sample": {"spot": "loader.stage", "view": {"barcode": "ABC"}}}}}
    b = parse_boundary(doc, _contracts(TYPED))
    first = b.result({})
    first["boundary"]["inputs"]["sample"]["view"]["barcode"] = "MUTATED"
    second = b.result({})
    assert second["boundary"]["inputs"]["sample"]["view"]["barcode"] == "ABC"


def test_empty_boundary_result_is_wellformed():
    """An empty boundary still yields a well-formed result skeleton."""
    b = parse_boundary(None, _contracts(COUNT))
    assert b.result({}) == {"boundary": {"inputs": {}, "outputs": {}}}
