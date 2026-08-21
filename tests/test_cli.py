"""Smoke tests for the ofp-run CLI scaffold.

These pin the CLI's shape and exit-code contract while the library is built out;
they are intentionally light and will grow as `run` gains real behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ofplang.run.cli import EXIT_OK, EXIT_USAGE, main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FIXTURES = Path(__file__).parent / "fixtures"
_BROKEN_YAML = "a: [1, 2\nb: :::\n"


def test_help_exits_zero(capsys):
    # `--help` is handled by argparse and exits with code 0.
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "ofp-run" in capsys.readouterr().out


def test_missing_subcommand_is_usage_error():
    # A subcommand is required; omitting it is a usage error (argparse -> exit 2).
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == EXIT_USAGE


def test_run_requires_env():
    # `run` needs --env; omitting it is an argparse usage error (exit 2).
    with pytest.raises(SystemExit) as exc:
        main(["run", "workflow.yaml"])
    assert exc.value.code == EXIT_USAGE


def test_run_missing_workflow_is_usage_error(capsys):
    # A workflow path that does not exist is an input (usage) error, not a failure.
    assert main(["run", "does_not_exist.yaml", "--env", "nope.yaml"]) == EXIT_USAGE
    assert "workflow not found" in capsys.readouterr().err


def test_replay_missing_plan_is_usage_error(capsys):
    # `replay` reads a plan; a missing plan file is a usage error.
    assert main(["replay", "does_not_exist.yaml", "--env", "nope.yaml"]) == EXIT_USAGE
    assert "cannot read plan" in capsys.readouterr().err


def test_run_malformed_workflow_is_caught_by_front_door(tmp_path, capsys):
    # A malformed workflow is rejected by the ofplang-validate front door. For
    # `run`, anything that stops it before execution is a usage error (exit 2);
    # the failure is reported with its validate error code.
    bad = tmp_path / "broken.workflow.yaml"
    bad.write_text(_BROKEN_YAML, encoding="utf-8")
    code = main(["run", str(bad), "--env", str(EXAMPLES / "count_chain.env.yaml")])
    assert code == EXIT_USAGE
    assert "wrong_value_kind" in capsys.readouterr().err


def test_run_no_validate_bypasses_front_door_on_malformed(tmp_path, capsys):
    # --no-validate skips the validation pass, but `$import` expansion is a
    # structural step that still runs, so a malformed workflow fails there: still
    # a usage error (exit 2), reported as the structural load failure.
    bad = tmp_path / "broken.workflow.yaml"
    bad.write_text(_BROKEN_YAML, encoding="utf-8")
    code = main(
        ["run", str(bad), "--env", str(EXAMPLES / "count_chain.env.yaml"), "--no-validate"]
    )
    assert code == EXIT_USAGE
    assert "wrong_value_kind" in capsys.readouterr().err


def test_run_structured_node_is_refused_before_running(capsys):
    # A structured node is valid portable v0 (so validate has nothing to say) that this
    # runner cannot execute -- spec 4.1's "valid v0 but unsupported". The capability
    # gate refuses it up front, which makes it a usage error (exit 2) like any other
    # workflow that never ran, rather than the failed *run* (exit 1) it used to report
    # from inside the scheduler. The reason names the node and the v0 feature.
    code = main(
        [
            "run",
            str(FIXTURES / "structured_node.workflow.yaml"),
            "--env",
            str(EXAMPLES / "count_chain.env.yaml"),
        ]
    )
    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "unsupported" in err
    assert "make_cups" in err
    assert "node_map" in err


def test_run_malformed_contract_is_caught_by_front_door(tmp_path, capsys):
    # An unparsable contract expression is caught by the front door (exit 2),
    # reported with its validate code rather than reaching the runner.
    wf_text = (EXAMPLES / "count_chain.workflow.yaml").read_text(encoding="utf-8")
    wf_text = wf_text.replace(
        "  inc:\n    kind: atomic\n",
        (
            '  inc:\n    kind: atomic\n    contracts:\n'
            '      requires:\n        - expr: "inputs.x.view.value >>>"\n'
        ),
    )
    wf = tmp_path / "bad_contract.workflow.yaml"
    wf.write_text(wf_text, encoding="utf-8")
    code = main(["run", str(wf), "--env", str(EXAMPLES / "count_chain.env.yaml")])
    assert code == EXIT_USAGE
    assert "contract_parse_error" in capsys.readouterr().err


def _count_chain_with_bogus_key(tmp_path) -> Path:
    """A parseable count_chain workflow that is invalid v0 (unknown top-level key)
    but that the runner still tolerates — it reads only the keys it needs."""
    src = (EXAMPLES / "count_chain.workflow.yaml").read_text(encoding="utf-8")
    wf = tmp_path / "bogus.workflow.yaml"
    wf.write_text("bogus_key: 1\n" + src, encoding="utf-8")
    return wf


def test_run_invalid_v0_workflow_is_usage_error(tmp_path, capsys):
    # A parseable-but-invalid-v0 workflow (not just malformed YAML) is caught by
    # the front door with its specific validate code.
    wf = _count_chain_with_bogus_key(tmp_path)
    code = main(["run", str(wf), "--env", str(EXAMPLES / "count_chain.env.yaml")])
    assert code == EXIT_USAGE
    assert "unknown_key" in capsys.readouterr().err


_GENERIC_WORKFLOW = """\
spec_version: "0.0"
processes:
  wash:
    kind: atomic
    type_params: {O: {domain: object}}
    inputs: {item: {type: O, phase: data}}
    outputs: {item: {type: O, phase: data}}
    objects: {map: {outputs.item: inputs.item}}
  main: {kind: atomic, inputs: {}, outputs: {}}
entry: main
"""


def test_run_generic_workflow_is_unsupported(tmp_path, capsys):
    # Generics are valid v0 (front door passes) but the runner does not
    # instantiate them: the capability gate rejects them as a usage error.
    wf = tmp_path / "generic.workflow.yaml"
    wf.write_text(_GENERIC_WORKFLOW, encoding="utf-8")
    code = main(["run", str(wf), "--env", str(EXAMPLES / "count_chain.env.yaml")])
    assert code == EXIT_USAGE
    assert "unsupported" in capsys.readouterr().err


def test_run_no_validate_still_gates_generics(tmp_path, capsys):
    # The capability gate runs even under --no-validate.
    wf = tmp_path / "generic.workflow.yaml"
    wf.write_text(_GENERIC_WORKFLOW, encoding="utf-8")
    code = main(
        ["run", str(wf), "--env", str(EXAMPLES / "count_chain.env.yaml"), "--no-validate"]
    )
    assert code == EXIT_USAGE
    assert "unsupported" in capsys.readouterr().err


def test_run_expands_import_end_to_end(tmp_path):
    # A `$import` workflow now runs: the front door resolves the imported `Count`
    # type and drives the expanded workflow to completion (no gate rejection).
    src = (EXAMPLES / "count_chain.workflow.yaml").read_text(encoding="utf-8")
    body = src[src.index("processes:") :]
    (tmp_path / "count_types.yaml").write_text(
        "Count:\n  domain: data\n  view:\n    value: { type: Int }\n", encoding="utf-8"
    )
    wf = tmp_path / "main.workflow.yaml"
    wf.write_text(
        'spec_version: "0.0"\ntypes:\n  $import: ./count_types.yaml\n' + body,
        encoding="utf-8",
    )
    out = tmp_path / "status.yaml"
    code = main(
        ["run", str(wf), "--env", str(EXAMPLES / "count_chain.env.yaml"), "-o", str(out)]
    )
    assert code == EXIT_OK


def test_run_no_validate_runs_invalid_v0(tmp_path):
    # --no-validate lets an invalid-v0-but-runnable workflow through: the runner
    # ignores the unknown key and drives it to completion.
    wf = _count_chain_with_bogus_key(tmp_path)
    out = tmp_path / "status.yaml"
    code = main(
        [
            "run",
            str(wf),
            "--env",
            str(EXAMPLES / "count_chain.env.yaml"),
            "--no-validate",
            "-o",
            str(out),
        ]
    )
    assert code == EXIT_OK


def test_run_unwritable_output_is_usage_error(tmp_path, capsys):
    # The run completes, but `-o` points into a directory that does not exist. An
    # output path that cannot be written is an input error like one that cannot be
    # read (exit 2), not an execution failure, and it says which output it was.
    out = tmp_path / "no_such_dir" / "status.yaml"
    code = main(
        [
            "run",
            str(EXAMPLES / "count_chain.workflow.yaml"),
            "--env",
            str(EXAMPLES / "count_chain.env.yaml"),
            "-o",
            str(out),
        ]
    )
    assert code == EXIT_USAGE
    assert "cannot write status" in capsys.readouterr().err


def test_run_unwritable_boundary_out_still_emits_the_status(tmp_path, capsys):
    # A secondary output that cannot be written must not cost the completed run its
    # status document: the failure is reported and decides the exit code, but the
    # remaining outputs are still written.
    out = tmp_path / "status.yaml"
    code = main(
        [
            "run",
            str(EXAMPLES / "count_chain.workflow.yaml"),
            "--env",
            str(EXAMPLES / "count_chain.env.yaml"),
            "--boundary-out",
            str(tmp_path / "no_such_dir" / "boundary.yaml"),
            "-o",
            str(out),
        ]
    )
    assert code == EXIT_USAGE
    assert "cannot write result boundary" in capsys.readouterr().err
    assert out.is_file()
