"""Unit tests for the value-routing layer (dev-notes design.md D26-1).

These are where the producer -> consumer *wiring* correctness is pinned (the
v0-lite rolling loop only smoke-tests integration): the dataflow adapter must
resolve every input port to the right source, including Pure Data arcs spliced
across a nested composite boundary, and the value primitives must route and
collect accordingly.
"""

from __future__ import annotations

from pathlib import Path

from ofplang.run.runner.dataflow import from_workflow
from ofplang.run.runner.values import (
    ValueStore,
    assemble_inputs,
    collect_outputs,
    dummy_value,
    record_outputs,
    seed_entry,
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
  Sample: {domain: object}
  Reading: {domain: data}
processes:
  measure:
    kind: atomic
    inputs: {plate: {type: Sample, phase: data}}
    outputs:
      plate_out: {type: Sample, phase: data}
      reading: {type: Reading, phase: data}
    objects: {transform: [inputs.plate, outputs.plate_out]}
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
    objects: {transform: [inputs.plate_in, outputs.done]}
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
    assert df.input_source[(("F",), "plate_in")] == (("M",), "plate_out")
    assert df.input_source[(("Az", "A"), "reading")] == (("M",), "reading")
    assert df.input_source[(("F",), "go")] == (("Az", "A"), "score")
    assert df.input_source[(("M",), "plate")] == ((), "sample")
    # Boundary + returns.
    assert df.entry_ports == ("sample",)
    assert df.returns == {"result": (("F",), "done")}


def test_pure_data_entry_input_is_a_boundary_source(tmp_path):
    # A workflow whose entry input is Pure Data and consumed via `bind` records the
    # boundary source too (via data_entry_inputs, D26-0), not just Object entries.
    doc = tmp_path / "wf2.yaml"
    doc.write_text(
        "spec_version: \"0.0\"\n"
        "types: {Reading: {domain: data}}\n"
        "processes:\n"
        "  gen: {kind: atomic, outputs: {out: {type: Reading, phase: data}}}\n"
        "  use: {kind: atomic, inputs: {a: {type: Reading, phase: data}, b: {type: Reading, phase: data}}}\n"
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
    assert df.input_source[(("U",), "a")] == (("G",), "out")   # Pure Data arc
    assert df.input_source[(("U",), "b")] == ((), "cfg")       # Pure Data entry input
    assert df.entry_ports == ("cfg",)


def test_structured_node_workflow_is_rejected(tmp_path):
    from ofplang.run.runner.runner import RunnerError

    doc = tmp_path / "bad.yaml"
    doc.write_text(
        "types: {Cup: {domain: object}}\n"
        "processes:\n"
        "  make: {kind: atomic, outputs: {cup: {type: Cup, phase: data}}, objects: {create: [outputs.cup]}}\n"
        "  main:\n"
        "    kind: composite\n"
        "    body: {nodes: [{id: m, kind: map, process: make, each: {x: {from: inputs.xs}}}]}\n"
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
    df = _dataflow(tmp_path)
    store = ValueStore()

    # Seed the boundary: the entry input carries a dummy.
    seed_entry(df, store)
    assert store.get((), "sample") == dummy_value((), "sample")

    # M consumes the entry input; run each node, recording the backend's outputs,
    # and check each consumer assembles the right upstream value.
    assert assemble_inputs(df, store, ("M",)) == {"plate": dummy_value((), "sample")}
    record_outputs(store, ("M",), {"plate_out": "P", "reading": "R"})

    assert assemble_inputs(df, store, ("Az", "A")) == {"reading": "R"}
    record_outputs(store, ("Az", "A"), {"score": "S"})

    assert assemble_inputs(df, store, ("F",)) == {"plate_in": "P", "go": "S"}
    record_outputs(store, ("F",), {"done": "D"})

    # The whole-workflow output follows the return back to F.done.
    assert collect_outputs(df, store) == {"result": "D"}


def test_unconnected_input_falls_back_to_dummy(tmp_path):
    df = _dataflow(tmp_path)
    store = ValueStore()
    # Before any producer has run, F's inputs have no stored source value yet, so
    # they dummy-fill (an unconnected/not-yet-produced input, v0-lite).
    assert assemble_inputs(df, store, ("F",)) == {
        "plate_in": dummy_value(("F",), "plate_in"),
        "go": dummy_value(("F",), "go"),
    }


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
    assert df.input_source[(("U1",), "a")] == (("G",), "o")
    assert df.input_source[(("U2",), "a")] == (("G",), "o")


def test_deep_nesting_flattens_paths_and_splices_data_arc(tmp_path):
    # A composite nested two levels deep: the atomic gains a three-segment path and
    # a Pure Data arc splices straight from the top producer to the deep consumer.
    df = _write(
        tmp_path,
        "spec_version: \"0.0\"\n"
        "types: {R: {domain: data}}\n"
        "processes:\n"
        "  g: {kind: atomic, outputs: {o: {type: R, phase: data}}}\n"
        "  a: {kind: atomic, inputs: {i: {type: R, phase: data}}, outputs: {o: {type: R, phase: data}}}\n"
        "  inner:\n"
        "    kind: composite\n"
        "    inputs: {ii: {type: R, phase: data}}\n"
        "    outputs: {io: {type: R, phase: data}}\n"
        "    body: {nodes: [{id: A, process: a, bind: {i: {from: inputs.ii}}}], returns: {io: {from: A.o}}}\n"
        "  outer:\n"
        "    kind: composite\n"
        "    inputs: {oi: {type: R, phase: data}}\n"
        "    outputs: {oo: {type: R, phase: data}}\n"
        "    body: {nodes: [{id: In, process: inner, bind: {ii: {from: inputs.oi}}}], returns: {oo: {from: In.io}}}\n"
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
    assert df.input_source[(("Ou", "In", "A"), "i")] == (("G",), "o")


def test_pure_data_entry_fan_out_is_a_known_v0_lite_limitation(tmp_path):
    # KNOWN LIMITATION (design.md D26 scope ledger): a boundary entry input is
    # recorded in a dict keyed by port name, so a Pure Data entry consumed by two
    # nodes keeps only the last consumer as a boundary source; the other falls back
    # to a dummy at assemble time. (Object entries cannot fan out -- linearity --
    # so only Pure Data entries hit this.) Pinning current behavior so a v0-full fix
    # (list-valued boundaries) is a deliberate, noticed change.
    df = _write(
        tmp_path,
        "spec_version: \"0.0\"\n"
        "types: {R: {domain: data}}\n"
        "processes:\n"
        "  u: {kind: atomic, inputs: {a: {type: R, phase: data}}}\n"
        "  main:\n"
        "    kind: composite\n"
        "    inputs: {cfg: {type: R, phase: data}}\n"
        "    body:\n"
        "      nodes:\n"
        "        - {id: U1, process: u, bind: {a: {from: inputs.cfg}}}\n"
        "        - {id: U2, process: u, bind: {a: {from: inputs.cfg}}}\n"
        "      returns: {}\n"
        "entry: main\n",
    )
    fed = [key for key in df.input_source if df.input_source[key] == ((), "cfg")]
    assert len(fed) == 1  # only one of U1/U2 -- the limitation
    assert df.entry_ports == ("cfg",)


def test_create_process_has_no_inputs_and_no_returns():
    # simple.workflow: `source` creates (no inputs), `target` consumes (no outputs),
    # no entry input, empty returns -- the CREATE / no-boundary shape.
    df = from_workflow(FIXTURES / "simple.workflow.yaml")
    assert df.in_ports[("SampleSource",)] == ()
    assert df.out_ports[("SampleSource",)] == ("source_out",)
    assert df.out_ports[("SampleTarget",)] == ()
    assert df.entry_ports == ()
    assert df.returns == {}
    assert df.input_source[(("SampleTarget",), "target_in")] == (("SampleSource",), "source_out")


def test_object_entry_and_object_return():
    # interface_load.workflow: an Object entry input and an Object return.
    df = from_workflow(FIXTURES / "interface_load.workflow.yaml")
    assert df.entry_ports == ("sample",)
    assert df.returns == {"result": (("Heat",), "out")}
    assert df.input_source[(("Heat",), "plate")] == ((), "sample")


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
    store = ValueStore()
    record_outputs(store, ("G",), {"o": "shared"})
    # Both consumers read the one produced value (values are not consumed in v0-lite).
    assert assemble_inputs(df, store, ("U1",)) == {"a": "shared"}
    assert assemble_inputs(df, store, ("U2",)) == {"a": "shared"}


def test_collect_outputs_omits_unproduced_return(tmp_path):
    df = _dataflow(tmp_path)  # returns {result <- F.done}
    store = ValueStore()
    # F never produced -> its return is omitted, not dummied.
    assert collect_outputs(df, store) == {}
    record_outputs(store, ("F",), {"done": "D"})
    assert collect_outputs(df, store) == {"result": "D"}


def test_dummy_value_is_deterministic_and_identifiable():
    assert dummy_value(("A", "B"), "p") == dummy_value(("A", "B"), "p")
    assert dummy_value(("A",), "p") != dummy_value(("A",), "q")
    assert dummy_value(("A",), "p") != dummy_value(("B",), "p")
    assert dummy_value((), "sample") == "dummy:@entry/sample"  # boundary renders as @entry
