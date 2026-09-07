"""Tests for the execution status the rolling runner renders (spec §6/§7).

Most of the document is a straight echo of the commit log, and the tests for the
runners cover it end to end. What is tested here is the one part that is *not* an
echo: the relays (§6.4.1) at the junctions of a multi-leg move. The runner never
dispatches a relay -- it is a scheduling junction, not an operation -- so there is
no record of one, and the status reconstructs each from the pair of committed legs
around it, the same way the scheduler regenerates them on a replan (§7).
"""

from __future__ import annotations

from ofplang.run.runner.provenance import Committed
from ofplang.run.runner.status import build_status

ARC = {
    "from": {"node": ["SampleSource"], "port": "source_out"},
    "to": {"node": ["SampleTarget"], "port": "target_in"},
}
OTHER_ARC = {
    "from": {"node": ["Other"], "port": "out"},
    "to": {"node": ["SampleTarget"], "port": "target_in"},
}


def leg(start, end, src, dst, *, seq=None, status="completed", arc=None, job=None):
    """One committed transport record, as the runner logs it: the plan's activity
    dict plus the observed times. A leg dispatched while its arc still had only one
    carries no `seq`, which is why `seq` defaults to absent here."""
    activity: dict = {"kind": "transport", "from_spot": src, "to_spot": dst, "arc": arc or ARC}
    if seq is not None:
        activity["seq"] = seq
    if job is not None:
        activity["job"] = job
    return Committed(activity, "transport", status, start, end, uuid="op")


def relays(records, now=100):
    return [a for a in build_status(records, now)["activities"] if a["kind"] == "relay"]


def kinds(records, now=100):
    return [a["kind"] for a in build_status(records, now)["activities"]]


# -- the junction of two real legs ------------------------------------------


def test_two_legs_of_one_arc_are_joined_by_a_relay():
    """The re-route shape: a leg delivered the Object to a spot, the destination
    moved, and a second leg carried it on. The junction between them is a relay at
    the shared spot, instantaneous, at the moment the first leg actually arrived."""
    status = build_status([leg(2, 3, "a.core", "b.core"), leg(3, 7, "b.core", "c.core", seq=2)], 9)

    assert [a["kind"] for a in status["activities"]] == ["transport", "relay", "transport"]
    relay = status["activities"][1]
    assert relay["spot"] == "b.core"
    assert relay["start"] == relay["end"] == 3  # instantaneous, at the observed arrival
    assert relay["status"] == "completed"
    assert relay["arc"] == ARC
    # The gap the legs leave is the relay's position -- not a guess (§6.6).
    assert relay["seq"] == 1
    assert [a.get("seq") for a in status["activities"]] == [0, 1, 2]


def test_the_relay_is_timed_by_the_observed_arrival_not_the_plan():
    """The delivering leg's record carries what actually happened; the relay is
    reconstructed from it, so variance moves the junction with the leg."""
    late = leg(2, 5, "a.core", "b.core")  # planned 2-3, observed 2-5
    assert relays([late, leg(5, 9, "b.core", "c.core", seq=2)])[0]["start"] == 5


def test_a_chain_of_real_legs_gets_a_relay_at_every_junction():
    """Repeated re-routes chain (§7): three legs, two junctions, positions 1 and 3."""
    chain = [
        leg(0, 1, "a.core", "b.core"),
        leg(1, 3, "b.core", "c.core", seq=2),
        leg(3, 6, "c.core", "d.core", seq=4),
    ]
    assert [(r["seq"], r["spot"]) for r in relays(chain)] == [(1, "b.core"), (3, "c.core")]


def test_a_revisited_spot_gets_a_relay_at_each_position():
    """A spot may be visited twice on one arc, told apart by `seq` alone (§6.6)."""
    there_and_back = [
        leg(0, 1, "a.core", "b.core"),
        leg(1, 3, "b.core", "a.core", seq=2),
        leg(3, 6, "a.core", "c.core", seq=4),
    ]
    assert [(r["seq"], r["spot"]) for r in relays(there_and_back)] == [(1, "b.core"), (3, "a.core")]


# -- what does *not* get a relay --------------------------------------------


def test_a_single_leg_arc_has_no_junction_and_keeps_its_bare_seq():
    """The ordinary case: one hop from producer to consumer. Nothing is inserted, and
    the leg is not given a `seq` it never needed (§6.6) -- so an ordinary run's status
    is exactly the document it always was."""
    status = build_status([leg(2, 3, "a.core", "b.core")], 3)

    assert [a["kind"] for a in status["activities"]] == ["transport"]
    assert "seq" not in status["activities"][0]


def test_legs_of_different_arcs_are_not_joined():
    """Two arcs that happen to touch the same spot are two separate Objects."""
    unrelated = [leg(0, 1, "a.core", "b.core"), leg(1, 3, "b.core", "c.core", arc=OTHER_ARC)]
    assert relays(unrelated) == []


def test_legs_of_different_jobs_are_not_joined():
    """Two jobs of one workflow render the very same arc (§6.11); pairing one job's
    leg with another's would invent a junction between two separate Objects."""
    pair = [
        leg(0, 1, "a.core", "b.core", job="job1"),
        leg(1, 3, "b.core", "c.core", seq=2, job="job2"),
    ]
    assert relays(pair) == []


def test_a_relay_carries_the_job_of_the_legs_it_joins():
    """A relay is matched by `job` + `arc` + `seq` (§6.6), so the job comes along."""
    pair = [
        leg(0, 1, "a.core", "b.core", job="job1"),
        leg(1, 3, "b.core", "c.core", seq=2, job="job1"),
    ]
    assert relays(pair)[0]["job"] == "job1"


def test_a_stay_put_no_op_departure_stays_folded():
    """The folded case (§6.4.1): the Object is consumed where the previous leg put
    it, so the departing leg goes nowhere. That relay and its no-op leg carry no
    information and are left out -- the scheduler folds them before the runner sees
    them, and reconstruction must not put them back."""
    folded = [leg(0, 1, "a.core", "b.core"), leg(1, 1, "b.core", "b.core", seq=2)]
    assert relays(folded) == []


def test_a_delivery_still_in_flight_makes_no_junction():
    """A running leg's `end` is the *planned* finish, not an arrival; a completed
    relay stated at a time that has not come would contradict `now` (§7). Nothing is
    lost by waiting: the next leg cannot have started either."""
    in_flight = [
        leg(2, 30, "a.core", "b.core", status="running"),
        leg(30, 34, "b.core", "c.core", seq=2),
    ]
    assert relays(in_flight, now=3) == []


def test_legs_that_do_not_meet_are_not_joined():
    """§7 guarantees a committed chain is continuous; this only refuses to invent a
    junction where the legs do not actually connect."""
    disjoint = [leg(0, 1, "a.core", "b.core"), leg(1, 3, "x.core", "c.core", seq=2)]
    assert relays(disjoint) == []


# -- the rest of the document is untouched ----------------------------------


def test_cancelled_work_takes_no_part():
    """A relay records an arrival that happened. Work that never ran has none, and
    the cancelled entries stay where they are -- after the committed history."""
    cancelled = [{"kind": "transport", "from_spot": "b.core", "to_spot": "c.core", "arc": ARC}]
    status = build_status([leg(2, 3, "a.core", "b.core")], 3, cancelled=cancelled)

    assert [a["kind"] for a in status["activities"]] == ["transport", "transport"]
    assert status["activities"][-1]["status"] == "cancelled"


def test_processing_records_are_left_alone():
    """Only transports form chains; a processing activity is echoed as it always
    was, in the order it was committed."""
    processing = Committed({"kind": "processing", "node": ["SampleSource"]}, "processing",
                           "completed", 0, 2, uuid="op")
    records = [processing, leg(2, 3, "a.core", "b.core"), leg(3, 7, "b.core", "c.core", seq=2)]
    assert kinds(records) == ["processing", "transport", "relay", "transport"]
