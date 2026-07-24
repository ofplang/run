"""Python script execution for the value seam (spec §22; dev-notes design.md D31).

A v0 `python_script_processes` process (§22) is an atomic Pure-Data process
carrying a `script` section: inline Python that computes its Pure Data outputs
from its Pure Data inputs. This module is the backend's executor for such a
process -- the first *real* computation in the value seam (D26/D27), where until
now the built-in device model only synthesised typed defaults.

Where it fits (D26 principle B -- value generation is the backend's job):
`script_device_model` is a device model (the `(process, mode, inputs,
output_schema, definition)` callback the simulator calls at completion). When the
process definition carries a `script` section it runs the script; otherwise it
delegates to `default_device_model`, so non-script processes are unaffected. It is
the simulator's built-in default (see `core.Simulator`), so a workflow with script
processes computes real values straight from the CLI with no injected model.

Execution model (§22.2): the `code` is the body of an implementation-provided
Python function; each input port name is bound as a local, and the function must
return a mapping with *exactly* the declared output names, each value conforming
to its declared type. A name mismatch, a non-conformant value, or any exception
the script raises is a runtime verification failure (§22.2) -- not an IR error --
so it is signalled with `DeviceComputationError`, which the simulator turns into a
`failed` operation (the graceful stop of D25). A script process is Pure Data, so a
failed one has no material effect, exactly as D25 requires.

Time model (dev-notes design.md D31): the script computes in real (wall-clock)
time but advances no simulation time. It runs inline here, at the operation's
completion; the outputs become visible only when the op reaches its `end` and is
reported `completed`, like any other operation. `end` is immediate for a
duration-0 mode and later for a duration>0 one -- the mode's duration is the
scheduler's estimate of the compute cost, and an estimate that differs from the
real cost is absorbed exactly like physical duration variance (D23).

Sandboxing (§22.3): none in this implementation. The script sees the full Python
builtins and may import anything the host Python can. §22.3 explicitly allows an
implementation to restrict imports / builtins for sandboxing or determinism; that
is a future tightening (implementation-defined), not done here. Execution is
deliberately confined to `run_python_script` so it can later move behind a thread
/ subprocess / remote executor (a separate design discussion, for distributed /
cloud compute) without touching the seam.
"""

from __future__ import annotations

from typing import Any


class DeviceComputationError(Exception):
    """A device model failed to compute an operation's outputs -- a *runtime*
    failure, not a validating-oracle precondition error.

    Raised by a device model (e.g. `script_device_model` when a script raises,
    returns the wrong output names, or returns a non-conformant value, §22.2) and
    caught by the simulator, which ends the operation `failed` (D25) instead of
    `completed`. A custom device model may raise it to signal the same graceful
    failure. It is intended for Pure Data computation (no material effect on
    failure), consistent with D25.

    `code` is a machine-readable reason code (D36) the simulator surfaces and the
    runner maps to its failure reason; the message is the human-readable detail."""

    def __init__(self, message: str, code: str = "script_error") -> None:
        super().__init__(message)
        self.code = code


def run_python_script(code: str, inputs: dict) -> Any:
    """Execute a script process's Python `code` (§22.2) and return its raw result.

    The `code` is evaluated as the body of an implementation-provided function
    whose parameters are the input port names, so each input is bound as a local
    variable (§22.2). `inputs` maps each declared input port to its view value. No
    sandboxing (§22.3): the function sees the full builtins and may import. The raw
    return value is passed back as-is; the caller (`script_device_model`) verifies
    its shape against the declared outputs.

    Any failure -- a syntax error compiling the code, or an exception the script
    raises -- is wrapped in `DeviceComputationError`, so it becomes a graceful
    runtime failure (§22.2) rather than crashing the run."""
    # Build `def <fn>(<ports>): <indented code>`. Input port names are v0
    # identifiers (§8.1) and so are valid Python parameter names. Every code line is
    # indented one level to form the function body; an empty body falls back to
    # `pass` so a scriptless / empty body still compiles.
    params = ", ".join(inputs.keys())
    body = "\n".join("    " + line for line in code.splitlines()) or "    pass"
    source = f"def __ofp_script__({params}):\n{body}"
    try:
        # `exec` injects the real builtins into `namespace` (no restriction, §22.3),
        # so the compiled function may use them -- and import -- freely.
        namespace: dict = {}
        exec(source, namespace)  # noqa: S102 - executing the user script is the feature
        return namespace["__ofp_script__"](**inputs)
    except DeviceComputationError:
        raise
    except Exception as exc:  # a compile / runtime error in the script -> graceful failure
        raise DeviceComputationError(f"script raised {type(exc).__name__}: {exc}") from exc


def _conforms_to_descriptor(value: Any, descriptor: dict) -> bool:
    """Whether `value` conforms to a neutral value-shape `descriptor` (the seam
    contract produced by `contracts.to_descriptor`).

    Mirrors `contracts.conforms`, but the backend walks the descriptor rather than
    the runner's resolved-type model (the backend imports nothing from the runner,
    D26). A primitive checks its Python kind (Bool is not an Int / Float, matching
    the runner's JSON-lenient checker); an array is a list of conforming elements; a
    record is a dict with exactly the declared view fields, each conforming."""
    kind = descriptor["kind"]
    if kind == "primitive":
        name = descriptor["name"]
        if name == "Bool":
            return isinstance(value, bool)
        if name == "Int":
            return isinstance(value, int) and not isinstance(value, bool)
        if name == "Float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if name == "String":
            return isinstance(value, str)
        return False
    if kind == "array":
        return isinstance(value, list) and all(_conforms_to_descriptor(v, descriptor["element"]) for v in value)
    # record: exactly the declared view fields, each conforming.
    if not isinstance(value, dict):
        return False
    fields = descriptor["fields"]
    if set(value) != set(fields):
        return False
    return all(_conforms_to_descriptor(value[name], fdesc) for name, fdesc in fields.items())


def script_device_model(process, mode, inputs, output_schema, definition):
    """Built-in device model that runs `python_script_processes` (§22).

    If `definition` carries a `script` section, execute it (§22.2) and verify the
    result: it must be a mapping whose keys are *exactly* the declared output ports
    (`output_schema`), each value conforming to its declared type (its value-shape
    descriptor). Any violation -- a non-mapping result, a missing / extra output
    name, a non-conformant value, or an unsupported script language -- raises
    `DeviceComputationError` (a runtime verification failure, §22.2; the simulator
    ends the op `failed`).

    With no `script` section it delegates to `default_device_model` (typed defaults
    + `objects.map` object pass-through), so non-script processes behave exactly as
    before. This is the simulator's default model, and it composes: a custom model
    may call it to handle its own script processes."""
    script = (definition or {}).get("script")
    if not script:
        # Not a script process: fall back to the type-default model. Imported lazily
        # because `core` imports this module as its default model (avoids a cycle).
        from .core import default_device_model

        return default_device_model(process, mode, inputs, output_schema, definition)

    # `python` is the only v0 script language (§22); a workflow that reached the
    # runner is assumed valid v0, but a wrong language cannot be run as Python, so
    # fail it as a runtime verification error rather than mis-executing it.
    language = script.get("language")
    if language != "python":
        raise DeviceComputationError(
            f"script process {process!r} declares unsupported script language {language!r}",
            code="script_language",
        )

    # Run the script and verify its result against the declared outputs (§22.2).
    result = run_python_script(script.get("code") or "", inputs or {})
    if not isinstance(result, dict):
        raise DeviceComputationError(
            f"script process {process!r} returned {type(result).__name__}, not a mapping",
            code="script_output_names",
        )
    if set(result) != set(output_schema):
        raise DeviceComputationError(
            f"script process {process!r} returned output names {sorted(result)}, "
            f"expected exactly {sorted(output_schema)}",
            code="script_output_names",
        )
    for port, descriptor in output_schema.items():
        if not _conforms_to_descriptor(result[port], descriptor):
            raise DeviceComputationError(
                f"script process {process!r} output {port!r} does not conform to its declared type",
                code="script_output_type",
            )
    return result
