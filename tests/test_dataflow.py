"""Unit tests for the value-routing layer (dev-notes design.md D26-1).

These are where the producer -> consumer *wiring* correctness is pinned (the
v0-lite rolling loop only smoke-tests integration): the dataflow adapter must
resolve every input port to the right source, including Pure Data arcs spliced
across a nested composite boundary, and the value primitives must route and
collect accordingly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ofplang.schedule.scheduler.model import SourceLiteral, SourceRef, SourceSeq

from ofplang.run.runner.contracts import Contracts
from ofplang.run.runner.dataflow import from_workflow
from ofplang.run.runner.values import (
    ValueStore,
    assemble_inputs,
    collect_outputs,
    record_outputs,
    resolve,
    seed_entry,
    unproduced_inputs,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _write(tmp_path, text):
    doc = tmp_path / "wf.yaml"
    doc.write_text(text, encoding="utf-8")
    return from_workflow(doc)

# A workflow exercising every routing case: an Object entry input (sample), an
# Object arc (M.plate_out -> F.plate_in), a Pure Data arc spliced across a nested
# composite boundary (M.reading -> Az/A.reading, via the analyzer's a_in), a Pure
# Data arc from inside the composite out to a sibling (Az/A.score -> F.go), and a
# non-empty return (result <- F.done).
_WORKFLOW = """\
spec_version: "0.0"
types:
  Sample: {domain: object, view: {id: {type: String}}}
  Reading: {domain: data, view: {v: {type: String}}}
processes:
  measure:
    kind: atomic
    inputs: {plate: {type: Sample, phase: data}}
    outputs:
      plate_out: {type: Sample, phase: data}
      reading: {type: Reading, phase: data}
    objects: {map: {outputs.plate_out: inputs.plate}}
  analyze:
    kind: atomic
    inputs: {reading: {type: Reading, phase: data}}
    outputs: {score: {type: Reading, phase: data}}
  finish:
    kind: atomic
    inputs:
      plate_in: {type: Sample, phase: data}
      go: {type: Reading, phase: data}
    outputs: {done: {type: Sample, phase: data}}
    objects: {map: {outputs.done: inputs.plate_in}}
  analyzer:
    kind: composite
    inputs: {a_in: {type: Reading, phase: data}}
    outputs: {a_out: {type: Reading, phase: data}}
    body:
      nodes:
        - {id: A, process: analyze, bind: {reading: {from: inputs.a_in}}}
      returns: {a_out: {from: A.score}}
  main:
    kind: composite
    inputs: {sample: {type: Sample, phase: data}}
    outputs: {result: {type: Sample, phase: data}}
    body:
      nodes:
        - {id: M, process: measure, state: {plate: {from: inputs.sample}}}
        - {id: Az, process: analyzer, bind: {a_in: {from: M.reading}}}
        - id: F
          process: finish
          state: {plate_in: {from: M.plate_out}}
          bind: {go: {from: Az.a_out}}
      returns: {result: {from: F.done}}
entry: main
"""


def _dataflow(tmp_path):
    doc = tmp_path / "wf.yaml"
    doc.write_text(_WORKFLOW, encoding="utf-8")
    return from_workflow(doc)


def _dataflow_and_contracts(tmp_path):
    doc = tmp_path / "wf.yaml"
    doc.write_text(_WORKFLOW, encoding="utf-8")
    return from_workflow(doc), Contracts.from_workflow(doc)


def test_ports_and_activities(tmp_path):
    df = _dataflow(tmp_path)
    assert df.process_of == {("M",): "measure", ("Az", "A"): "analyze", ("F",): "finish"}
    assert df.in_ports[("M",)] == ("plate",)
    assert df.in_ports[("Az", "A")] == ("reading",)
    assert set(df.in_ports[("F",)]) == {"plate_in", "go"}
    assert df.out_ports[("M",)] == ("plate_out", "reading")
    assert df.out_ports[("Az", "A")] == ("score",)
    assert df.out_ports[("F",)] == ("done",)


def test_input_sources_resolve_across_arcs_and_boundary(tmp_path):
    df = _dataflow(tmp_path)
    # Object arc, Pure Data arc (boundary-spliced), Pure Data arc out of the
    # composite, and the Object entry input all resolve to the right source.
    assert df.sources[(("F",), "plate_in")] == SourceRef(("M",), "plate_out")
    assert df.sources[(("Az", "A"), "reading")] == SourceRef(("M",), "reading")
    assert df.sources[(("F",), "go")] == SourceRef(("Az", "A"), "score")
    assert df.sources[(("M",), "plate")] == SourceRef((), "sample")
    # Boundary + returns.
    assert df.entry_ports == ("sample",)
    assert df.returns == {"result": SourceRef(("F",), "done")}


def test_pure_data_entry_input_is_a_boundary_source(tmp_path):
    # A workflow whose entry input is Pure Data and consumed via `bind` records the
    # boundary source too (via data_entry_inputs, D26-0), not just Object entries.
    doc = tmp_path / "wf2.yaml"
    doc.write_text(
        "spec_version: \"0.0\"\n"
        "types: {Reading: {domain: data}}\n"
        "processes:\n"
        "  gen: {kind: atomic, outputs: {out: {type: Reading, phase: data}}}\n"
        "  use: {kind: atomic, inputs: "
        "{a: {type: Reading, phase: data}, b: {type: Reading, phase: data}}}\n"
        "  main:\n"
        "    kind: composite\n"
        "    inputs: {cfg: {type: Reading, phase: data}}\n"
        "    body:\n"
        "      nodes:\n"
        "        - {id: G, process: gen}\n"
        "        - {id: U, process: use, bind: {a: {from: G.out}, b: {from: inputs.cfg}}}\n"
        "      returns: {}\n"
        "entry: main\n",
        encoding="utf-8",
    )
    df = from_workflow(doc)
    assert df.sources[(("U",), "a")] == SourceRef(("G",), "out")   # Pure Data arc
    assert df.sources[(("U",), "b")] == SourceRef((), "cfg")       # Pure Data entry input
    assert df.entry_ports == ("cfg",)


def test_structured_node_workflow_is_rejected(tmp_path):
    from ofplang.run.runner.runner import RunnerError

    doc = tmp_path / "bad.yaml"
    doc.write_text(
        "types: {Cup: {domain: object}}\n"
        "processes:\n"
        "  make: {kind: atomic, outputs: {cup: {type: Cup, phase: data}}, "
        "objects: {create: [outputs.cup]}}\n"
        "  main:\n"
        "    kind: composite\n"
        "    body: {nodes: [{id: m, kind: do_while, process: make}]}\n"
        "entry: main\n",
        encoding="utf-8",
    )
    try:
        from_workflow(doc)
    except RunnerError:
        return
    raise AssertionError("expected RunnerError for a structured node workflow")


# -- value primitives --------------------------------------------------------


def test_seed_assemble_record_collect_route_values(tmp_path):
    df, contracts = _dataflow_and_contracts(tmp_path)
    store = ValueStore()

    # Seed the boundary: with no job, the entry input gets a typed default (Sample's
    # one view field, defaulted).
    seed_entry(df, contracts, store)
    assert store.get((), "sample") == {"id": ""}

    # M consumes the entry input; run each node, recording distinct sentinel outputs,
    # and check each consumer assembles the right upstream value (routing).
    P, R, S, D = ({"id": "P"}, {"v": "R"}, {"v": "S"}, {"id": "D"})
    assert assemble_inputs(df, contracts, store, ("M",)) == {"plate": {"id": ""}}
    record_outputs(store, ("M",), {"plate_out": P, "reading": R})

    assert assemble_inputs(df, contracts, store, ("Az", "A")) == {"reading": R}
    record_outputs(store, ("Az", "A"), {"score": S})

    assert assemble_inputs(df, contracts, store, ("F",)) == {"plate_in": P, "go": S}
    record_outputs(store, ("F",), {"done": D})

    # The whole-workflow output follows the return back to F.done.
    assert collect_outputs(df, store) == {"result": D}


def test_seed_entry_uses_job_values_and_checks_them(tmp_path):
    df, contracts = _dataflow_and_contracts(tmp_path)
    from ofplang.run.runner.runner import RunnerError

    # A supplied job value is used verbatim.
    store = ValueStore()
    seed_entry(df, contracts, store, job={"sample": {"id": "S1"}})
    assert store.get((), "sample") == {"id": "S1"}

    # A non-conformant job value is rejected, as is an unknown entry port.
    for bad in ({"sample": {"unexpected": 1}}, {"nope": {}}):
        try:
            seed_entry(df, contracts, ValueStore(), job=bad)
        except RunnerError:
            continue
        raise AssertionError(f"expected RunnerError for job {bad}")


def test_assemble_refuses_an_input_whose_producer_has_not_run(tmp_path):
    from ofplang.run.runner.runner import RunnerError

    df, contracts = _dataflow_and_contracts(tmp_path)
    store = ValueStore()
    # Before any producer has run, F's inputs have no value. They used to fall back
    # to a typed default -- a stand-in the run would then compute on as though it
    # were the workflow's value. Now nothing is assembled at all: a caller asks
    # `unproduced_inputs` (below) first, and reaching this is a runner bug.
    with pytest.raises(RunnerError, match="no value recorded for M.plate_out"):
        assemble_inputs(df, contracts, store, ("F",))


def test_unproduced_inputs_reports_only_connected_ports_without_a_value(tmp_path):
    df, contracts = _dataflow_and_contracts(tmp_path)
    store = ValueStore()

    # Both of F's inputs are connected to producers that have not run.
    assert unproduced_inputs(df, store, ("F",)) == ["plate_in", "go"]

    # A boundary-fed input is *not* reported once seeded -- and seeding happens at
    # run start, before anything is dispatched, so it never is in practice.
    assert unproduced_inputs(df, store, ("M",)) == ["plate"]
    seed_entry(df, contracts, store)
    assert unproduced_inputs(df, store, ("M",)) == []

    # As each producer records its output, its consumer's port stops being reported.
    record_outputs(store, ("M",), {"plate_out": {"id": "P"}, "reading": {"v": "R"}})
    assert unproduced_inputs(df, store, ("F",)) == ["go"]
    record_outputs(store, ("Az", "A"), {"score": {"v": "S"}})
    assert unproduced_inputs(df, store, ("F",)) == []


def test_unproduced_inputs_ignores_a_literal_bound_port(tmp_path):
    # A static literal is the workflow's own value, so a port bound to one is never
    # "unproduced" -- there is no producer to wait for.
    doc = tmp_path / "wf.yaml"
    doc.write_text(_LITERAL_WF, encoding="utf-8")
    df = from_workflow(doc)
    assert unproduced_inputs(df, ValueStore(), ("C",)) == []


# -- static literals (`bind: {port: {value: ...}}`, §11 / D30) ---------------

_LITERAL_WF = """\
spec_version: "0.0"
processes:
  compute:
    kind: atomic
    inputs: {cfg: {type: Int, phase: data}}
    outputs: {out: {type: Int, phase: data}}
  main:
    kind: composite
    inputs: {}
    body:
      nodes:
        - {id: C, process: compute, bind: {cfg: {value: 5}}}
      returns: {}
entry: main
"""


def test_literal_is_recorded_and_assembled(tmp_path):
    doc = tmp_path / "wf.yaml"
    doc.write_text(_LITERAL_WF, encoding="utf-8")
    df = from_workflow(doc)
    contracts = Contracts.from_workflow(doc)

    # The adapter surfaces the literal as the consuming (node, port)'s source.
    assert df.sources == {(("C",), "cfg"): SourceLiteral(5)}
    # assemble_inputs seeds it as the port's value (not a typed default of 0).
    assert assemble_inputs(df, contracts, ValueStore(), ("C",)) == {"cfg": 5}


def test_nonconformant_literal_is_rejected(tmp_path):
    from ofplang.run.runner.runner import RunnerError

    doc = tmp_path / "wf.yaml"
    doc.write_text(_LITERAL_WF.replace("value: 5", 'value: "not-an-int"'), encoding="utf-8")
    df = from_workflow(doc)
    contracts = Contracts.from_workflow(doc)
    with pytest.raises(RunnerError, match="does not conform"):
        assemble_inputs(df, contracts, ValueStore(), ("C",))


_LITERAL_NESTED_WF = """\
spec_version: "0.0"
processes:
  compute:
    kind: atomic
    inputs: {cfg: {type: Int, phase: data}}
    outputs: {out: {type: Int, phase: data}}
  wrap:
    kind: composite
    inputs: {w: {type: Int, phase: data}}
    outputs: {wo: {type: Int, phase: data}}
    body:
      nodes:
        - {id: C, process: compute, bind: {cfg: {from: inputs.w}}}
      returns: {wo: {from: C.out}}
  main:
    kind: composite
    inputs: {}
    body:
      nodes:
        - {id: W, process: wrap, bind: {w: {value: 9}}}
      returns: {}
entry: main
"""


def test_literal_spliced_across_composite_boundary(tmp_path):
    # A literal supplied to a composite input reaches the inner atomic that consumes
    # it, at its qualified node path (the flattener propagates it, D30).
    doc = tmp_path / "wf.yaml"
    doc.write_text(_LITERAL_NESTED_WF, encoding="utf-8")
    df = from_workflow(doc)
    contracts = Contracts.from_workflow(doc)
    assert df.sources == {(("W", "C"), "cfg"): SourceLiteral(9)}
    assert assemble_inputs(df, contracts, ValueStore(), ("W", "C")) == {"cfg": 9}


# -- dataflow adapter edge cases --------------------------------------------


def test_internal_fan_out_one_output_feeds_many_consumers(tmp_path):
    # A producer output bound into two consumers resolves for both (arcs are a
    # list, so fan-out is not lost -- unlike a boundary entry, see below).
    df = _write(
        tmp_path,
        "spec_version: \"0.0\"\n"
        "types: {R: {domain: data}}\n"
        "processes:\n"
        "  g: {kind: atomic, outputs: {o: {type: R, phase: data}}}\n"
        "  u: {kind: atomic, inputs: {a: {type: R, phase: data}}}\n"
        "  main:\n"
        "    kind: composite\n"
        "    body:\n"
        "      nodes:\n"
        "        - {id: G, process: g}\n"
        "        - {id: U1, process: u, bind: {a: {from: G.o}}}\n"
        "        - {id: U2, process: u, bind: {a: {from: G.o}}}\n"
        "      returns: {}\n"
        "entry: main\n",
    )
    assert df.sources[(("U1",), "a")] == SourceRef(("G",), "o")
    assert df.sources[(("U2",), "a")] == SourceRef(("G",), "o")


def test_deep_nesting_flattens_paths_and_splices_data_arc(tmp_path):
    # A composite nested two levels deep: the atomic gains a three-segment path and
    # a Pure Data arc splices straight from the top producer to the deep consumer.
    df = _write(
        tmp_path,
        "spec_version: \"0.0\"\n"
        "types: {R: {domain: data}}\n"
        "processes:\n"
        "  g: {kind: atomic, outputs: {o: {type: R, phase: data}}}\n"
        "  a: {kind: atomic, inputs: {i: {type: R, phase: data}}, "
        "outputs: {o: {type: R, phase: data}}}\n"
        "  inner:\n"
        "    kind: composite\n"
        "    inputs: {ii: {type: R, phase: data}}\n"
        "    outputs: {io: {type: R, phase: data}}\n"
        "    body: {nodes: [{id: A, process: a, bind: {i: {from: inputs.ii}}}], "
        "returns: {io: {from: A.o}}}\n"
        "  outer:\n"
        "    kind: composite\n"
        "    inputs: {oi: {type: R, phase: data}}\n"
        "    outputs: {oo: {type: R, phase: data}}\n"
        "    body: {nodes: [{id: In, process: inner, bind: {ii: {from: inputs.oi}}}], "
        "returns: {oo: {from: In.io}}}\n"
        "  main:\n"
        "    kind: composite\n"
        "    body:\n"
        "      nodes:\n"
        "        - {id: G, process: g}\n"
        "        - {id: Ou, process: outer, bind: {oi: {from: G.o}}}\n"
        "      returns: {}\n"
        "entry: main\n",
    )
    assert set(df.process_of) == {("G",), ("Ou", "In", "A")}
    assert df.sources[(("Ou", "In", "A"), "i")] == SourceRef(("G",), "o")


def test_pure_data_entry_fan_out_reaches_every_consumer(tmp_path):
    # A Pure Data entry input consumed by two nodes feeds both. This used to be a
    # known limitation (design.md D26 scope ledger): the boundary was recorded in a
    # dict keyed by port name, so only the last consumer got a source and the other
    # was silently given a typed default. The scheduler's flattener now records one
    # boundary data arc per consumer, with `()` as the source. (Object entries cannot
    # fan out -- linearity -- so only Pure Data entries ever hit this.)
    wf_text = (
        "spec_version: \"0.0\"\n"
        "processes:\n"
        "  u: {kind: atomic, inputs: {a: {type: Int, phase: data}}}\n"
        "  main:\n"
        "    kind: composite\n"
        "    inputs: {cfg: {type: Int, phase: data}}\n"
        "    body:\n"
        "      nodes:\n"
        "        - {id: U1, process: u, bind: {a: {from: inputs.cfg}}}\n"
        "        - {id: U2, process: u, bind: {a: {from: inputs.cfg}}}\n"
        "      returns: {}\n"
        "entry: main\n"
    )
    df = _write(tmp_path, wf_text)
    fed = {key for key, source in df.sources.items() if source == SourceRef((), "cfg")}
    assert fed == {(("U1",), "a"), (("U2",), "a")}
    assert df.entry_ports == ("cfg",)

    # And at the value layer: both consumers read the value that came in, not a default.
    contracts = Contracts.from_workflow(tmp_path / "wf.yaml")
    store = ValueStore()
    seed_entry(df, contracts, store, {"cfg": 7})
    assert assemble_inputs(df, contracts, store, ("U1",)) == {"a": 7}
    assert assemble_inputs(df, contracts, store, ("U2",)) == {"a": 7}


# Outputs that have no producing activity: `t_echo` returns the entry input verbatim,
# `t_wrapped` returns it through a composite that passes it straight back, and `k`
# returns the literal a nested composite was bound to. Until the scheduler's flattener
# recorded them, all three silently went missing from the run's outputs.
_NO_PRODUCER_RETURNS_WF = """\
spec_version: "0.0"
processes:
  echo:
    kind: composite
    inputs: {k: {type: Float, phase: run}}
    outputs: {k: {type: Float, phase: run}}
    body:
      nodes: []
      returns: {k: {from: inputs.k}}
  main:
    kind: composite
    inputs: {t: {type: Float, phase: run}}
    outputs:
      t_echo: {type: Float, phase: run}
      t_wrapped: {type: Float, phase: run}
      k: {type: Float, phase: run}
    body:
      nodes:
        - {id: E, process: echo, bind: {k: {from: inputs.t}}}
        - {id: C, process: echo, bind: {k: {value: 3.0}}}
      returns:
        t_echo: {from: inputs.t}
        t_wrapped: {from: E.k}
        k: {from: C.k}
entry: main
"""


def test_pass_through_and_literal_returns_are_collected(tmp_path):
    df = _write(tmp_path, _NO_PRODUCER_RETURNS_WF)
    # A pass-through is produced by the boundary, where the entry input is seeded; a
    # literal return has no producer at all.
    assert df.returns == {
        "t_echo": SourceRef((), "t"),
        "t_wrapped": SourceRef((), "t"),
        "k": SourceLiteral(3.0),
    }

    contracts = Contracts.from_workflow(tmp_path / "wf.yaml")
    store = ValueStore()
    seed_entry(df, contracts, store, {"t": 1.5})
    assert collect_outputs(df, store) == {"t_echo": 1.5, "t_wrapped": 1.5, "k": 3.0}


def test_create_process_has_no_inputs_and_no_returns():
    # simple.workflow: `source` creates (no inputs), `target` consumes (no outputs),
    # no entry input, empty returns -- the CREATE / no-boundary shape.
    df = from_workflow(FIXTURES / "simple.workflow.yaml")
    assert df.in_ports[("SampleSource",)] == ()
    assert df.out_ports[("SampleSource",)] == ("source_out",)
    assert df.out_ports[("SampleTarget",)] == ()
    assert df.entry_ports == ()
    assert df.returns == {}
    assert df.sources[(("SampleTarget",), "target_in")] == SourceRef(
        ("SampleSource",), "source_out"
    )


def test_object_entry_and_object_return():
    # interface_load.workflow: an Object entry input and an Object return.
    df = from_workflow(FIXTURES / "interface_load.workflow.yaml")
    assert df.entry_ports == ("sample",)
    assert df.returns == {"result": SourceRef(("Heat",), "out")}
    assert df.sources[(("Heat",), "plate")] == SourceRef((), "sample")


# -- value primitives edge cases --------------------------------------------


def test_value_store_normalises_list_and_tuple_keys():
    # The rolling loop keys by an activity's `node` (a list); tests key by tuple.
    # Both must address the same slot.
    store = ValueStore()
    store.put(["A", "B"], "p", "v")  # list in
    assert store.has(("A", "B"), "p") and store.get(("A", "B"), "p") == "v"
    record_outputs(store, ["C"], {"q": "w"})
    assert store.get(("C",), "q") == "w"


def test_fan_out_value_read_by_multiple_consumers(tmp_path):
    df = _write(
        tmp_path,
        "spec_version: \"0.0\"\n"
        "types: {R: {domain: data}}\n"
        "processes:\n"
        "  g: {kind: atomic, outputs: {o: {type: R, phase: data}}}\n"
        "  u: {kind: atomic, inputs: {a: {type: R, phase: data}}}\n"
        "  main:\n"
        "    kind: composite\n"
        "    body:\n"
        "      nodes:\n"
        "        - {id: G, process: g}\n"
        "        - {id: U1, process: u, bind: {a: {from: G.o}}}\n"
        "        - {id: U2, process: u, bind: {a: {from: G.o}}}\n"
        "      returns: {}\n"
        "entry: main\n",
    )
    contracts = Contracts.from_workflow(tmp_path / "wf.yaml")
    store = ValueStore()
    record_outputs(store, ("G",), {"o": {}})
    # Both consumers read the one produced value (values are not consumed in v0).
    assert assemble_inputs(df, contracts, store, ("U1",)) == {"a": {}}
    assert assemble_inputs(df, contracts, store, ("U2",)) == {"a": {}}


def test_collect_outputs_omits_unproduced_return(tmp_path):
    df = _dataflow(tmp_path)  # returns {result <- F.done}
    store = ValueStore()
    # F never produced -> its return is omitted, not defaulted.
    assert collect_outputs(df, store) == {}
    record_outputs(store, ("F",), {"done": {"id": "D"}})
    assert collect_outputs(df, store) == {"result": {"id": "D"}}


# -- Source trees (map / fold, D59) ---------------------------------------------


def test_resolve_reads_an_element_and_assembles_a_sequence():
    store = ValueStore()
    store.put((), "plates", [{"id": "a"}, {"id": "b"}])
    store.put(("Read", 0), "od", 0.5)
    store.put(("Read", 1), "od", 0.7)
    assert resolve(SourceRef((), "plates", (1,)), store) == {"id": "b"}
    gathered = SourceSeq((SourceRef(("Read", 0), "od"), SourceRef(("Read", 1), "od")))
    assert resolve(gathered, store) == [0.5, 0.7]
    assert resolve(SourceSeq(()), store) == []
    assert resolve(SourceLiteral(3), store) == 3


def test_resolve_refuses_an_element_past_the_end():
    from ofplang.run.runner.runner import RunnerError

    store = ValueStore()
    store.put((), "xs", [1.0])
    with pytest.raises(RunnerError, match=r"main\.xs has no element at \[1\]"):
        resolve(SourceRef((), "xs", (1,)), store)


def test_an_unbound_input_is_refused_before_anything_runs(tmp_path):
    # v0 binds every input (§11) and defines no default for one. A document that
    # leaves one unbound is invalid -- and run without validation it used to get a
    # typed default; now it is refused before anything runs, by the scheduler's
    # reader with validate's code for it (schedule D60).
    from ofplang.run.runner.runner import RunnerError

    doc = tmp_path / "wf.yaml"
    doc.write_text(_LITERAL_WF.replace("bind: {cfg: {value: 5}}", "bind: {}"), encoding="utf-8")
    with pytest.raises(RunnerError, match="data_indegree"):
        from_workflow(doc)


def test_seed_entry_shapes_an_array_of_objects_by_its_spots(tmp_path):
    from ofplang.run.runner.runner import RunnerError

    doc = tmp_path / "wf.yaml"
    doc.write_text(_ARRAY_WF, encoding="utf-8")
    interface = {"inputs": {"plates": ["hotel.a", "hotel.b", "hotel.c"]}}
    df = from_workflow(doc, interface=interface)
    contracts = Contracts.from_workflow(doc)
    spots = interface["inputs"]

    # No view supplied: one default view per spot, not `[]`.
    store = ValueStore()
    seed_entry(df, contracts, store, spots=spots)
    assert store.get((), "plates") == [{"id": ""}] * 3

    # A supplied view is one per spot, in order.
    views = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    store = ValueStore()
    seed_entry(df, contracts, store, {"plates": views}, spots=spots)
    assert assemble_inputs(df, contracts, store, ("Heat", 1)) == {"plate": {"id": "y"}}

    # One that is not is refused.
    with pytest.raises(RunnerError, match="shape of its spots"):
        seed_entry(df, contracts, ValueStore(), {"plates": views[:2]}, spots=spots)


_ARRAY_WF = """\
spec_version: "0.4"
types:
  Plate: {domain: object, view: {id: {type: String}}}
processes:
  heat:
    kind: atomic
    inputs: {plate: {type: Plate, phase: data}}
    outputs: {plate: {type: Plate, phase: data}}
    objects: {map: {outputs.plate: inputs.plate}}
  main:
    kind: composite
    inputs: {plates: {type: "Array<Plate>", phase: data}}
    outputs: {plates: {type: "Array<Plate>", phase: data}}
    body:
      nodes:
        - {id: Heat, kind: map, process: heat, each: {plate: {from: inputs.plates}}}
      returns: {plates: {from: Heat.plate}}
entry: main
"""
