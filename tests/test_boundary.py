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

from ofplang.run.runner.boundary import parse_boundary
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
    assert b.entry_values == {"sample": {"barcode": "ABC"}}
    assert b.output_spots == {}


def test_pure_data_input_projects_view_only():
    """A Pure Data input carries no spot: nothing reaches the interface, only the job."""
    doc = {"boundary": {"inputs": {"start": {"view": {"value": 42}}}}}
    b = parse_boundary(doc, _contracts(COUNT))
    assert b.interface == {}  # no Objects -> no interface at all
    assert b.entry_values == {"start": {"value": 42}}


def test_input_view_omitted_is_not_seeded():
    """An input with a spot but no view supplies no job value (it defaults later)."""
    doc = {"boundary": {"inputs": {"sample": {"spot": "loader.stage"}}}}
    b = parse_boundary(doc, _contracts(TYPED))
    assert b.interface == {"inputs": {"sample": "loader.stage"}}
    assert b.entry_values == {}  # view omitted -> seed_entry will default it


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
    doc = {
        "boundary": {
            "outputs": {"result": {"spot": "unloader.slot", "view": {"barcode": "STALE"}}}
        }
    }
    b = parse_boundary(doc, _contracts(LOAD))
    assert b.output_spots == {"result": "unloader.slot"}
    assert b.entry_values == {}  # nothing from the output view leaks into the seed


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
        assert b.interface == {} and b.entry_values == {} and b.output_spots == {}


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


# -- inventories (§6.10) -----------------------------------------------------


def test_inventories_pass_through_unchanged():
    """The section mirrors §6.10 exactly, so the translation into the status is an
    identity copy -- there is no projection to get wrong."""
    doc = {"boundary": {"inventories": {"levels": {"reader": {"reagent": 6, "tips": 0}}}}}
    b = parse_boundary(doc, _contracts(COUNT))
    assert b.inventories == {"levels": {"reader": {"reagent": 6, "tips": 0}}}


def test_inventories_absent_stays_absent():
    """"Every stock starts empty" (`levels: {}`) and "the run does not say" are
    different claims; only the scheduler can tell whether the second is an error."""
    assert parse_boundary({"boundary": {}}, _contracts(COUNT)).inventories == {}
    assert parse_boundary(None, _contracts(COUNT)).inventories == {}


def test_inventories_with_empty_levels_is_a_claim_not_an_omission():
    doc = {"boundary": {"inventories": {"levels": {}}}}
    assert parse_boundary(doc, _contracts(COUNT)).inventories == {"levels": {}}


def test_inventories_are_not_echoed_into_the_result():
    """A result boundary is fed back in; the starting stock must not travel with it."""
    doc = {"boundary": {"inventories": {"levels": {"reader": {"reagent": 6}}}}}
    b = parse_boundary(doc, _contracts(COUNT))
    assert "inventories" not in b.result({"result": {"value": 1}})["boundary"]


def test_unknown_key_under_boundary_is_rejected():
    """`boundary:` is closed for the same reason a descriptor is: silently ignoring
    `inventores:` would report `missing_inventories` for a run that did supply it."""
    with pytest.raises(RunnerError, match="unknown key"):
        parse_boundary({"boundary": {"inventores": {}}}, _contracts(COUNT))


def test_unknown_key_under_inventories_is_rejected():
    with pytest.raises(RunnerError, match="unknown key"):
        parse_boundary({"boundary": {"inventories": {"initial": {}}}}, _contracts(COUNT))


@pytest.mark.parametrize(
    "levels",
    [
        {"reader": {"reagent": -1}},  # a stock cannot hold less than nothing
        {"reader": {"reagent": "two"}},  # a level is an integer, not a word
        {"reader": {"reagent": 1.5}},  # nor a fraction of a unit
        {"reader": {"reagent": True}},  # `bool` is an `int` in Python; this is a typo
        {"reader": 6},  # a device holds named stocks, not one number
    ],
)
def test_a_malformed_level_is_rejected_here(levels):
    """Shape only, but shape early: a level of `"two"` would otherwise travel all the
    way into the solver before anything said so."""
    with pytest.raises(RunnerError, match="inventories"):
        parse_boundary({"boundary": {"inventories": {"levels": levels}}}, _contracts(COUNT))


def test_which_devices_and_resources_exist_is_not_checked_here():
    """Answering that needs the environment definition, which this module does not
    read; the scheduler reports `unknown_device` / `unknown_resource` /
    `inventory_exceeds_capacity` against the document it is given."""
    doc = {"boundary": {"inventories": {"levels": {"no_such_device": {"no_such": 999}}}}}
    assert parse_boundary(doc, _contracts(COUNT)).inventories["levels"]["no_such_device"]
