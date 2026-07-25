"""Mode-id normalization pins a stable id on every process mode (dev-notes D21).

The generated fill (`m<i>`) must not collide with a user-supplied id in the same
process -- modes are keyed by id downstream, so a collision would silently shadow
a mode and mis-resolve the plan's mode reference (review #7).
"""

from __future__ import annotations

from ofplang.run.runner.rolling import _normalize_mode_ids


def test_generated_mode_id_avoids_user_id_collision():
    env = {"processes": {"p": {"modes": [{"id": "m1", "duration": 1}, {"duration": 2}]}}}
    out = _normalize_mode_ids(env)
    ids = [m["id"] for m in out["processes"]["p"]["modes"]]
    assert ids[0] == "m1"           # user id preserved
    assert ids[1] != "m1"           # generated id must not collide with it
    assert len(set(ids)) == len(ids)  # all distinct


def test_all_unnamed_modes_get_sequential_ids():
    env = {"processes": {"p": {"modes": [{"duration": 1}, {"duration": 2}]}}}
    out = _normalize_mode_ids(env)
    ids = [m["id"] for m in out["processes"]["p"]["modes"]]
    assert ids == ["m0", "m1"]


def test_input_is_not_mutated():
    env = {"processes": {"p": {"modes": [{"duration": 1}]}}}
    _normalize_mode_ids(env)
    assert "id" not in env["processes"]["p"]["modes"][0]  # a copy is returned
