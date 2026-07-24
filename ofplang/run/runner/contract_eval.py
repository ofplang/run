"""Runtime evaluator for v0 contract expressions (spec §9; dev-notes design.md D32).

Run-local, and deliberately promotable to a shared `ofplang-types` /
`ofplang-contracts` module later (the project's stated direction): this module
imports nothing else from the runner. It parses a contract `expr` string into a
tiny AST and evaluates it against runtime view values supplied through a `resolve`
callback, so it knows nothing about the runner's type model or value store.

Division of labour with `ofplang-validate` (spec §9.3): validate owns the
*graph-time* layer -- it parses, type-checks the expression to Bool, enforces
reference scope and view-field existence, and constant-folds a fully-literal
contract, flagging a statically-false one as an IR error. This module does the
complementary *runtime* layer: evaluate a reference-bearing contract against the
actual view values of one activity's invocation (D32). The runner assumes valid v0
input, so this evaluator trusts the expression is well-typed and re-derives none
of validate's static diagnostics.

Grammar (§9.2), precedence lowest -> highest: `or`; `and`; comparison (`==` `!=`
`<` `<=` `>` `>=`, non-associative); additive (`+` `-`); multiplicative (`*`
`/`); unary (`not`, `-`); primary (literal, `.view` reference, parenthesised). A
reference is `inputs.<port>.view[.<field>]` or `outputs.<port>.view[.<field>]`.
Numeric promotion (Int with Float) falls out of Python's own arithmetic, matching
the spec's contract-local rule (`Int / Int` yields a Float, etc.).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


class ContractSyntaxError(Exception):
    """A contract expression could not be parsed. In valid v0 this never fires
    (validate parses and type-checks first); it guards against a malformed
    expression reaching the runner regardless."""


# -- lexer -------------------------------------------------------------------
#
# Numbers follow the strict v0 forms (a Float needs digits on both sides of the
# dot, so it must be tried before Int). A dotted word is a reference; the bare
# words below are keywords; the rest are operators / parentheses.

_FLOAT_RE = re.compile(r"[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?")
_INT_RE = re.compile(r"0|[1-9][0-9]*")
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_PATH_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_OPS = ("==", "!=", "<=", ">=", "<", ">", "+", "-", "*", "/")
_KEYWORDS = {"and", "or", "not", "true", "false"}


@dataclass
class _Tok:
    kind: str  # 'int' | 'float' | 'str' | 'ref' | 'kw' | 'op' | 'lparen' | 'rparen'
    text: str


def _tokenize(expr: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i, n = 0, len(expr)
    while i < n:
        c = expr[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "(":
            toks.append(_Tok("lparen", "(")); i += 1; continue
        if c == ")":
            toks.append(_Tok("rparen", ")")); i += 1; continue
        if c == '"':
            m = _STRING_RE.match(expr, i)
            if not m:
                raise ContractSyntaxError(f"unterminated string in contract expression: {expr!r}")
            toks.append(_Tok("str", m.group())); i = m.end(); continue
        # Float before Int (a Float carries the '.').
        m = _FLOAT_RE.match(expr, i)
        if m:
            toks.append(_Tok("float", m.group())); i = m.end(); continue
        m = _INT_RE.match(expr, i)
        if m:
            toks.append(_Tok("int", m.group())); i = m.end(); continue
        # A dotted word: a keyword, else a reference path.
        m = _PATH_RE.match(expr, i)
        if m:
            text = m.group()
            toks.append(_Tok("kw" if text in _KEYWORDS else "ref", text)); i = m.end(); continue
        # An operator (try two-character forms before one-character).
        for op in _OPS:
            if expr.startswith(op, i):
                toks.append(_Tok("op", op)); i += len(op); break
        else:
            raise ContractSyntaxError(f"unexpected character {c!r} in contract expression: {expr!r}")
    return toks


# -- AST ---------------------------------------------------------------------


@dataclass
class _Lit:
    value: object  # bool / int / float / str


@dataclass
class _Ref:
    scope: str  # "inputs" | "outputs"
    port: str
    fields: tuple  # segments after ".view": () (bare view) or (field,)


@dataclass
class _Unary:
    op: str  # "not" | "-"
    operand: object


@dataclass
class _Binary:
    op: str  # and or == != < <= > >= + - * /
    left: object
    right: object


def _make_ref(path: str) -> _Ref:
    """Turn a dotted reference path into a `_Ref`. Every valid v0 contract
    reference is a `.view` reference (§9.1): `inputs|outputs.<port>.view[.<field>]`.
    """
    parts = path.split(".")
    if len(parts) < 3 or parts[0] not in ("inputs", "outputs") or parts[2] != "view":
        raise ContractSyntaxError(f"invalid contract reference: {path!r}")
    return _Ref(parts[0], parts[1], tuple(parts[3:]))


# -- parser (precedence-climbing recursive descent) --------------------------


class _Parser:
    def __init__(self, toks: list[_Tok]) -> None:
        self.toks = toks
        self.pos = 0

    def parse(self):
        node = self._or()
        if self.pos != len(self.toks):
            raise ContractSyntaxError("trailing tokens in contract expression")
        return node

    def _peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _next(self):
        tok = self._peek()
        self.pos += 1
        return tok

    def _is_kw(self, kw: str) -> bool:
        t = self._peek()
        return t is not None and t.kind == "kw" and t.text == kw

    def _is_op(self, *ops: str) -> bool:
        t = self._peek()
        return t is not None and t.kind == "op" and t.text in ops

    def _or(self):
        left = self._and()
        while self._is_kw("or"):
            self._next()
            left = _Binary("or", left, self._and())
        return left

    def _and(self):
        left = self._cmp()
        while self._is_kw("and"):
            self._next()
            left = _Binary("and", left, self._cmp())
        return left

    def _cmp(self):
        # Comparisons are non-associative (`a < b < c` is invalid v0), so at most one.
        left = self._add()
        if self._is_op("==", "!=", "<", "<=", ">", ">="):
            op = self._next().text
            return _Binary(op, left, self._add())
        return left

    def _add(self):
        left = self._mul()
        while self._is_op("+", "-"):
            op = self._next().text
            left = _Binary(op, left, self._mul())
        return left

    def _mul(self):
        left = self._unary()
        while self._is_op("*", "/"):
            op = self._next().text
            left = _Binary(op, left, self._unary())
        return left

    def _unary(self):
        if self._is_kw("not"):
            self._next()
            return _Unary("not", self._unary())
        if self._is_op("-"):
            self._next()
            return _Unary("-", self._unary())
        return self._primary()

    def _primary(self):
        t = self._next()
        if t is None:
            raise ContractSyntaxError("unexpected end of contract expression")
        if t.kind == "int":
            return _Lit(int(t.text))
        if t.kind == "float":
            return _Lit(float(t.text))
        if t.kind == "str":
            return _Lit(json.loads(t.text))  # JSON-style string escapes (§9.2)
        if t.kind == "kw" and t.text in ("true", "false"):
            return _Lit(t.text == "true")
        if t.kind == "ref":
            return _make_ref(t.text)
        if t.kind == "lparen":
            node = self._or()
            close = self._next()
            if close is None or close.kind != "rparen":
                raise ContractSyntaxError("missing ')' in contract expression")
            return node
        raise ContractSyntaxError(f"unexpected token {t.text!r} in contract expression")


def parse(expr: str):
    """Parse a v0 contract `expr` string into an AST (see `evaluate`)."""
    return _Parser(_tokenize(expr)).parse()


# -- evaluation --------------------------------------------------------------


def evaluate(ast, resolve):
    """Evaluate a parsed contract AST to a value (a Bool for a whole contract).

    `resolve(scope, port, fields)` supplies the runtime value of a `.view`
    reference: `scope` is "inputs" / "outputs", `port` the port name, and `fields`
    the segments after `.view` (empty for a bare `.view`). The caller wires it to
    the invocation's actual view values. Numeric promotion and division follow
    Python's arithmetic, which matches the spec's contract-local rule (§9.2). A
    runtime error (e.g. division by zero on runtime values) propagates to the
    caller, which treats it as a runtime contract violation (§9.3)."""
    if isinstance(ast, _Lit):
        return ast.value
    if isinstance(ast, _Ref):
        return resolve(ast.scope, ast.port, ast.fields)
    if isinstance(ast, _Unary):
        if ast.op == "not":
            return not evaluate(ast.operand, resolve)
        return -evaluate(ast.operand, resolve)
    # _Binary. `and` / `or` may short-circuit at runtime (§9.2 leaves this to the
    # implementation); the comparison / arithmetic operators evaluate both sides.
    op = ast.op
    if op == "and":
        return bool(evaluate(ast.left, resolve)) and bool(evaluate(ast.right, resolve))
    if op == "or":
        return bool(evaluate(ast.left, resolve)) or bool(evaluate(ast.right, resolve))
    a = evaluate(ast.left, resolve)
    b = evaluate(ast.right, resolve)
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
    raise ContractSyntaxError(f"unknown operator {op!r}")  # unreachable for a parsed AST
