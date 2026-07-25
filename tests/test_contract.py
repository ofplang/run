"""Runtime contract checking (spec §9; dev-notes design.md D32).

Two layers:

* the contract-expression evaluator in isolation -- `parse` + `evaluate` over a
  `resolve` callback: literals, operator precedence, numeric promotion, `.view`
  references (including `Array<T>.view.length`), and runtime evaluation errors;
* the runner end to end -- an atomic process's `requires` is checked before it
  runs and its `ensures` after it completes, over real script-computed values; a
  violation of either stops the run gracefully (D25), leaving downstream work
  cancelled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ofplang.run.runner.contract_eval import ContractSyntaxError, evaluate, parse

FIXTURES = Path(__file__).parent / "fixtures"
CONTRACT_WF = str(FIXTURES / "contract.workflow.yaml")
CONTRACT_ENV = str(FIXTURES / "contract.env.yaml")


# -- evaluator in isolation --------------------------------------------------


def _const(expr):
    """Evaluate a reference-free contract expression (no `resolve` needed)."""
    def resolve(scope, port, fields):  # pragma: no cover - not reached for constants
        raise AssertionError("no reference expected")

    return evaluate(parse(expr), resolve)


def _with(env):
    """A `resolve` over `env` = {scope: {port: view value}}; a 1-field path reads a
    record field, or `length` on a list."""
    def resolve(scope, port, fields):
        value = env[scope][port]
        if not fields:
            return value
        field = fields[0]
        return len(value) if field == "length" else value[field]

    return resolve


def test_arithmetic_precedence():
    assert _const("1 + 2 * 3") == 7
    assert _const("(1 + 2) * 3") == 9


def test_numeric_promotion_and_division():
    # Division always yields Float, even Int / Int (§9.2).
    assert _const("6 / 2") == 3.0 and isinstance(_const("6 / 2"), float)
    assert _const("7 / 2") == 3.5
    # + / - / * are Int only when both operands are Int.
    assert _const("1 + 2") == 3 and isinstance(_const("1 + 2"), int)
    assert isinstance(_const("1.0 + 2"), float)


def test_unary_and_boolean():
    assert _const("- 3 + 5") == 2          # unary minus binds tighter than +
    assert _const("not false") is True
    assert _const("true and false") is False
    assert _const("true or false") is True
    assert _const("not true == false") is True  # (not true) == false


def test_comparisons():
    assert _const("3 > 2") is True
    assert _const("2 >= 2") is True
    assert _const("1 < 0") is False


def test_references_primitive_record_array_and_string():
    env = {
        "inputs": {
            "x": 5,                       # primitive view = scalar
            "p": {"lo": 1, "hi": 9},      # nominal view record
            "xs": [10, 20, 30],           # Array view; .length below
            "s": "ok",
        },
        "outputs": {"y": 12},
    }
    resolve = _with(env)
    assert evaluate(parse("inputs.x.view >= 0"), resolve) is True
    assert evaluate(parse("inputs.p.view.hi - inputs.p.view.lo == 8"), resolve) is True
    assert evaluate(parse("inputs.xs.view.length == 3"), resolve) is True
    assert evaluate(parse('inputs.s.view == "ok"'), resolve) is True
    assert evaluate(parse("outputs.y.view == inputs.x.view + 7"), resolve) is True


def test_runtime_error_propagates():
    # A runtime evaluation error (division by zero on runtime values) propagates; the
    # runner treats it as a contract violation (§9.3).
    resolve = _with({"inputs": {"a": 1, "b": 0}, "outputs": {}})
    with pytest.raises(ZeroDivisionError):
        evaluate(parse("inputs.a.view / inputs.b.view > 0"), resolve)


def test_comparison_chaining_is_a_parse_error():
    # `a < b < c` is not valid v0 (non-associative comparisons, §9.2).
    with pytest.raises(ContractSyntaxError):
        parse("1 < 2 < 3")


def test_incomplete_expression_is_a_parse_error():
    with pytest.raises(ContractSyntaxError):
        parse("1 +")


# -- runner end to end -------------------------------------------------------

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402


def _boundary(raw):
    return {"boundary": {"inputs": {"raw": {"view": raw}}}}


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_contracts_hold_run_completes(poll_interval):
    # raw = 72: requires (raw >= 0) holds, and margin = 72 - 60 = 12 satisfies ensures.
    runner = RollingRunner(
        CONTRACT_WF,
        CONTRACT_ENV,
        _boundary(72),
        poll_interval=poll_interval,
        random_seed=0,
    )
    status = runner.run()

    assert not runner.failed
    assert all(a["status"] == "completed" for a in status["activities"])
    assert runner.outputs == {"margin": 12, "doubled": 24}


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_requires_violation_stops_before_dispatch(poll_interval):
    # raw = -5 violates `requires: inputs.raw.view >= 0`. `score` must not run: it is
    # recorded failed before dispatch and its downstream `report` is cancelled (D25).
    runner = RollingRunner(
        CONTRACT_WF,
        CONTRACT_ENV,
        _boundary(-5),
        poll_interval=poll_interval,
        random_seed=0,
    )
    status = runner.run()

    assert runner.failed
    statuses = {
        a.get("process"): a["status"]
        for a in status["activities"]
        if a.get("kind") == "processing"
    }
    assert statuses.get("score") == "failed"
    assert statuses.get("report") == "cancelled"
    # `score` never produced a value, so no whole-workflow output was assembled.
    assert runner.outputs == {}


def test_violated_exprs_treats_arithmetic_error_as_violation():
    # A runtime evaluation error over the view values (division by zero) is a
    # contract violation, not a crash (§9.2/§9.3): the wrapper returns the expr.
    runner = RollingRunner(CONTRACT_WF, CONTRACT_ENV, _boundary(72), random_seed=0)
    ast = parse("inputs.a.view / inputs.b.view >= 1")
    violated = runner._violated_exprs(
        "score", "requires", [("expr", ast)], {"a": 1, "b": 0}, {}, "x"
    )
    assert violated == "expr"


def test_violated_exprs_propagates_internal_lookup_error():
    # A structural error -- a referenced port missing from the resolver dict, i.e. a
    # runner bug -- must NOT be swallowed as a user-facing contract violation; it
    # propagates so the real fault is not hidden.
    runner = RollingRunner(CONTRACT_WF, CONTRACT_ENV, _boundary(72), random_seed=0)
    ast = parse("inputs.missing.view >= 0")
    with pytest.raises(KeyError):
        runner._violated_exprs("score", "requires", [("expr", ast)], {}, {}, "x")


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_ensures_violation_stops_at_completion(poll_interval, tmp_path):
    # A script whose `margin` breaks `ensures: margin == raw - threshold` (it adds
    # instead of subtracting). `score` completes physically, then its ensures fails at
    # the poll -> the run stops gracefully with `score` failed and `report` cancelled.
    wf = tmp_path / "bad_ensures.workflow.yaml"
    wf.write_text(
        Path(CONTRACT_WF).read_text(encoding="utf-8").replace(
            'return {"margin": raw - threshold}', 'return {"margin": raw + threshold}'
        ),
        encoding="utf-8",
    )
    runner = RollingRunner(
        str(wf),
        CONTRACT_ENV,
        _boundary(72),
        poll_interval=poll_interval,
        random_seed=0,
    )
    status = runner.run()

    assert runner.failed
    statuses = {
        a.get("process"): a["status"]
        for a in status["activities"]
        if a.get("kind") == "processing"
    }
    assert statuses.get("score") == "failed"
    assert statuses.get("report") == "cancelled"


@pytest.mark.parametrize("poll_interval", [None, 1])
def test_ensures_failed_output_is_withheld(poll_interval, tmp_path):
    # `Score.margin` feeds `returns.margin`, but on the ensures violation the value
    # that failed its postcondition must not surface as a produced workflow output
    # (review #5): it is withdrawn from the store, so self.outputs omits it rather
    # than echoing the tainted value.
    wf = tmp_path / "bad_ensures.workflow.yaml"
    wf.write_text(
        Path(CONTRACT_WF).read_text(encoding="utf-8").replace(
            'return {"margin": raw - threshold}', 'return {"margin": raw + threshold}'
        ),
        encoding="utf-8",
    )
    runner = RollingRunner(
        str(wf),
        CONTRACT_ENV,
        _boundary(72),
        poll_interval=poll_interval,
        random_seed=0,
    )
    runner.run()
    assert runner.failed
    assert runner.outputs == {}  # margin withheld (ensures-failed); doubled never produced
