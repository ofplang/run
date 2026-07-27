"""Out-of-process child harness for `SubprocessBackend` (dev-notes: labcode 06).

Runs one resolved script in its own process so a long / CPU-bound computation does
not block the runner (the parent discovers completion by polling, never a callback).
It is deliberately thin: it reuses `run_python_script` (the same in-process executor
`script_device_model` uses) and `verify_outputs` (the same §22.2 output check), so a
script behaves identically whether run in-process or here.

Protocol (all JSON, so it is language-neutral at the boundary):

* **in**: a job object on **stdin** --
  ``{"code", "inputs", "output_schema", "process", "language", "result_path"}``.
* **out**: the *outcome* is written to ``result_path`` (NOT stdout, so the user
  script's own ``print`` cannot corrupt it) as one of
  ``{"outputs": {...}}`` (success) or ``{"error": {"code", "message"}}`` (a script /
  verification failure, i.e. a graceful runtime failure, v0 §22.2).
* **stdout / stderr**: left to the user script (the parent forwards / captures them).
* **exit code**: ``0`` whenever a defined outcome was written to ``result_path``
  (success *or* a script error -- both are results). Non-zero only when the harness
  itself could not run (unreadable job, unwritable result); the parent then reports a
  child-level failure and folds stderr into the reason.
"""

from __future__ import annotations

import json
import sys
import traceback

from .script import DeviceComputationError, run_python_script, verify_outputs


def main() -> int:
    try:
        job = json.load(sys.stdin)
        result_path = job["result_path"]
    except Exception as exc:  # cannot even read the job -- a harness-level failure
        print(f"ofp-run child: cannot read job: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    try:
        raw = run_python_script(job.get("code") or "", job.get("inputs") or {})
        outputs = verify_outputs(raw, job.get("output_schema") or {}, job.get("process"))
        payload: dict = {"outputs": outputs}
    except DeviceComputationError as exc:
        # A graceful script / verification failure (v0 §22.2): a defined outcome.
        payload = {"error": {"code": exc.code, "message": str(exc)}}
    except Exception as exc:  # a script error run_python_script did not wrap -- still an outcome
        payload = {"error": {"code": "script_error", "message": f"{type(exc).__name__}: {exc}"}}

    try:
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception as exc:  # cannot deliver the result -- a harness-level failure
        print(f"ofp-run child: cannot write result: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry
    raise SystemExit(main())
