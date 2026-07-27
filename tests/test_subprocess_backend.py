"""Tests for `SubprocessBackend` -- the real-execution backend that runs op scripts
out-of-process, paced to a wall clock, completing on a poll rather than a callback.

Most tests drive the completion / material / value logic deterministically via an
*injected fake spawn* (no real subprocess) on a *fake clock* (no real seconds), the
same fake-injection approach `test_realtime` uses for pacing. A `computing_spawn`
runs the script in-process *eagerly* and hands back a fake handle, so the outputs are
genuinely computed while the test stays deterministic; a `_delayed` variant keeps the
handle `running` for N polls to exercise the still-running / overrun path. A single
real-subprocess smoke exercises the actual `python -m` child, and a child-harness unit
test pins its stdin/result-file protocol directly.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest

from ofplang.run.simulator import SubprocessBackend, subprocess_backend_factory
from ofplang.run.simulator._child import main as child_main
from ofplang.run.simulator.script import DeviceComputationError, run_python_script, verify_outputs

FIXTURES = Path(__file__).parent / "fixtures"
SCRIPT_ENV = str(FIXTURES / "script.env.yaml")
SCRIPT_WF = str(FIXTURES / "script.workflow.yaml")

INT = {"kind": "primitive", "name": "Int"}
STR = {"kind": "primitive", "name": "String"}

ADD_DEF = {"script": {"language": "python", "code": "return {'z': x + y}"}}


class FakeClock:
    """Monotonic clock where only `sleep` (or an explicit jump) makes time pass."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class FakeHandle:
    """A minimal Popen-compatible handle: `poll()` returns None for `running_polls`
    calls, then `returncode`; `stderr` is a readable stream."""

    def __init__(self, running_polls: int = 0, returncode: int = 0, stderr: str = "") -> None:
        self._left = running_polls
        self._rc = returncode
        self.returncode = None
        self.stderr = io.StringIO(stderr)
        self.stdin = None

    def poll(self):
        if self._left > 0:
            self._left -= 1
            return None
        self.returncode = self._rc
        return self._rc


def _computing_spawn(running_polls: int = 0):
    """Build a fake `spawn` that computes the job's script *in-process* (eagerly,
    writing the same result file a real child would) and returns a `FakeHandle` that
    reports done after `running_polls` polls. Deterministic, no real subprocess."""

    def spawn(job: dict) -> FakeHandle:
        try:
            raw = run_python_script(job.get("code") or "", job.get("inputs") or {})
            outputs = verify_outputs(raw, job.get("output_schema") or {}, job.get("process"))
            payload: dict = {"outputs": outputs}
        except DeviceComputationError as exc:
            payload = {"error": {"code": exc.code, "message": str(exc)}}
        with open(job["result_path"], "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return FakeHandle(running_polls=running_polls)

    return spawn


def _crashing_spawn(job: dict) -> FakeHandle:
    """A fake `spawn` whose child 'crashes': non-zero exit, no result file, stderr
    set (the harness-level failure path)."""
    return FakeHandle(returncode=1, stderr="boom traceback")


def _backend(spawn, **kw) -> SubprocessBackend:
    clock = FakeClock()
    return SubprocessBackend(
        SCRIPT_ENV, spawn=spawn, seconds_per_tick=1.0,
        monotonic=clock.monotonic, sleep=clock.sleep, **kw,
    )


# -- unit: completion / outputs / failure via fake spawn -----------------------


def test_coded_op_completes_and_produces_outputs():
    backend = _backend(_computing_spawn())
    uid = backend.dispatch_processing(
        "add", "v0", output_schema={"z": INT}, inputs={"x": 2, "y": 3}, definition=ADD_DEF
    )
    assert backend.state(uid)["status"] == "running"  # dispatch does not block
    backend.advance(1)  # pace + poll: child already done -> settle
    st = backend.state(uid)
    assert st["status"] == "completed"
    assert st["outputs"] == {"z": 5}  # genuinely computed by the script


def test_coded_op_stays_running_until_child_exits():
    # The child reports running for two polls, ignoring the advisory duration (2): the
    # op must stay running past its virtual end until the poll sees it finished.
    backend = _backend(_computing_spawn(running_polls=2))
    uid = backend.dispatch_processing(
        "add", "v0", output_schema={"z": INT}, inputs={"x": 1, "y": 1}, definition=ADD_DEF
    )
    backend.advance(2)  # reaches the virtual end, but child still running
    assert backend.state(uid)["status"] == "running"
    backend.advance(3)
    assert backend.state(uid)["status"] == "running"
    backend.advance(4)  # third poll -> child done
    assert backend.state(uid)["status"] == "completed"
    assert backend.state(uid)["outputs"] == {"z": 2}


def test_script_error_fails_op_with_reason():
    backend = _backend(_computing_spawn())
    bad = {"script": {"language": "python", "code": "return {'z': 1 / 0}"}}
    uid = backend.dispatch_processing(
        "add", "v0", output_schema={"z": INT}, inputs={"x": 0, "y": 0}, definition=bad
    )
    backend.advance(1)
    st = backend.state(uid)
    assert st["status"] == "failed"
    assert st["reason"][0] == "script_error"  # a graceful runtime failure (v0 §22.2)


def test_wrong_output_names_fail_op():
    backend = _backend(_computing_spawn())
    bad = {"script": {"language": "python", "code": "return {'wrong': 1}"}}
    uid = backend.dispatch_processing(
        "add", "v0", output_schema={"z": INT}, inputs={"x": 0, "y": 0}, definition=bad
    )
    backend.advance(1)
    st = backend.state(uid)
    assert st["status"] == "failed"
    assert st["reason"][0] == "script_output_names"


def test_child_crash_fails_op_with_stderr():
    backend = _backend(_crashing_spawn)
    uid = backend.dispatch_processing(
        "add", "v0", output_schema={"z": INT}, inputs={"x": 0, "y": 0}, definition=ADD_DEF
    )
    backend.advance(1)
    st = backend.state(uid)
    assert st["status"] == "failed"
    assert st["reason"][0] == "child_error"
    assert "boom" in st["reason"][1]


def test_scriptless_op_is_timed_and_defaulted():
    # No script in the definition -> resolver returns None -> the op is *timed*: it
    # completes when the clock passes its virtual end (duration 2), with the built-in
    # default model filling a typed default (Int -> 0). No subprocess is spawned.
    backend = _backend(_computing_spawn())
    uid = backend.dispatch_processing(
        "add", "v0", output_schema={"z": INT}, inputs={}, definition={}
    )
    backend.advance(1)  # before end
    assert backend.state(uid)["status"] == "running"
    backend.advance(2)  # reaches end -> timed completion
    st = backend.state(uid)
    assert st["status"] == "completed"
    assert st["outputs"] == {"z": 0}


def test_close_terminates_running_children():
    seen = {}

    class Terminable(FakeHandle):
        def terminate(self):
            seen["terminated"] = True

    def spawn(job):
        with open(job["result_path"], "w", encoding="utf-8") as fh:
            json.dump({"outputs": {"z": 0}}, fh)
        return Terminable(running_polls=99)  # never finishes on its own

    backend = _backend(spawn)
    backend.dispatch_processing(
        "add", "v0", output_schema={"z": INT}, inputs={"x": 0, "y": 0}, definition=ADD_DEF
    )
    backend.close()
    assert seen.get("terminated") is True


# -- child harness protocol (unit, no real subprocess) -------------------------


def test_child_main_writes_outputs(tmp_path, monkeypatch):
    result_path = tmp_path / "r.json"
    job = {
        "code": "return {'z': x + y}", "inputs": {"x": 4, "y": 5},
        "output_schema": {"z": INT}, "process": "add", "result_path": str(result_path),
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(job)))
    assert child_main() == 0
    assert json.loads(result_path.read_text(encoding="utf-8")) == {"outputs": {"z": 9}}


def test_child_main_writes_error_on_script_failure(tmp_path, monkeypatch):
    result_path = tmp_path / "r.json"
    job = {
        "code": "return {'z': 1 / 0}", "inputs": {"x": 0, "y": 0},
        "output_schema": {"z": INT}, "process": "add", "result_path": str(result_path),
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(job)))
    assert child_main() == 0  # a defined outcome, not a harness crash
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["error"]["code"] == "script_error"


# -- integration: drive a full run through the runner --------------------------


def test_e2e_run_script_workflow_computes_via_subprocess_backend():
    pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")
    from ofplang.run.runner import RollingRunner

    clock = FakeClock()
    factory = subprocess_backend_factory(
        spawn=_computing_spawn(), seconds_per_tick=1.0,
        monotonic=clock.monotonic, sleep=clock.sleep,
    )
    runner = RollingRunner(SCRIPT_WF, SCRIPT_ENV, backend_factory=factory, random_seed=0)
    status = runner.run()

    assert isinstance(runner.sim, SubprocessBackend)
    assert all(a["status"] == "completed" for a in status["activities"])
    # The scripts really ran out-of-band: add computes z = a + b (defaulted 0 + 0),
    # label renders "sum=0" -- not typed defaults (a defaulted String would be "").
    assert runner.outputs == {"sum": 0, "summary": "sum=0"}


# -- smoke: the real subprocess path (spends a little real time) ---------------


def test_real_subprocess_smoke():
    """Exercise the actual `python -m ofplang.run.simulator._child` child: dispatch a
    script op with the default (real) spawn and poll on a real clock until it finishes.
    Kept tiny; asserts the computed result, so both the spawn and the result-file
    round-trip are real."""
    backend = SubprocessBackend(SCRIPT_ENV, seconds_per_tick=0.01)
    uid = backend.dispatch_processing(
        "add", "v0", output_schema={"z": INT}, inputs={"x": 7, "y": 8}, definition=ADD_DEF
    )
    deadline = time.monotonic() + 30.0
    tick = 1
    while backend.state(uid)["status"] == "running" and time.monotonic() < deadline:
        backend.advance(tick)
        tick += 1
    backend.close()
    st = backend.state(uid)
    assert st["status"] == "completed", st
    assert st["outputs"] == {"z": 15}
