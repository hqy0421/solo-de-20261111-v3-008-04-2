from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import bytewax.operators as op
from bytewax.dataflow import Dataflow
from bytewax.operators import (
    _SIDE_DIM,
    _SIDE_FACT,
    DimDelete,
    DimUpsert,
    DimWatermark,
    TemporalExpired,
    TemporalJoined,
    _DimVersion,
    _PendingFact,
    _TemporalJoinLogic,
    _TemporalJoinState,
)
from bytewax.testing import TestingSink, TestingSource, run_main
from pytest import raises

ZERO_TD = timedelta(seconds=0)


def _dt(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _fact(tx_id: str, at: datetime) -> Dict[str, Any]:
    return {"tx": tx_id, "at": at}


def _fact_ts(tx: Dict[str, Any]) -> datetime:
    return tx["at"]


def _logic(
    retention: timedelta = timedelta(days=30),
    allowed_lateness: timedelta = ZERO_TD,
    state: Optional[_TemporalJoinState] = None,
) -> _TemporalJoinLogic:
    return _TemporalJoinLogic(
        fact_ts=_fact_ts,
        retention=retention,
        allowed_lateness=allowed_lateness,
        state=_TemporalJoinState() if state is None else state,
    )


def _dim_in(logic: _TemporalJoinLogic, event: Any) -> List[Any]:
    out, discard = logic.on_item((_SIDE_DIM, event))
    assert not discard
    return list(out)


def _fact_in(logic: _TemporalJoinLogic, fact: Dict[str, Any]) -> List[Any]:
    out, discard = logic.on_item((_SIDE_FACT, fact))
    assert not discard
    return list(out)


def _build_flow(
    facts_l: List[Any],
    dim_l: List[Any],
    joined_out: List[Any],
    expired_out: List[Any],
    retention: timedelta = timedelta(days=30),
    allowed_lateness: timedelta = ZERO_TD,
    batch_size: int = 1,
) -> Dataflow:
    flow = Dataflow("test_df")
    facts = op.input("facts", flow, TestingSource(facts_l, batch_size=batch_size))
    dim = op.input("dim", flow, TestingSource(dim_l, batch_size=batch_size))
    out = op.temporal_join(
        "tjoin",
        facts,
        dim,
        fact_ts=_fact_ts,
        retention=retention,
        allowed_lateness=allowed_lateness,
    )
    op.output("joined", out.joined, TestingSink(joined_out))
    op.output("expired", out.expired, TestingSink(expired_out))
    return flow


def test_fact_waits_for_frontier() -> None:
    logic = _logic()
    fact = _fact("f1", _dt(2024, 1, 15))
    # No dimension progress yet: the fact must wait.
    assert _fact_in(logic, fact) == []
    # Frontier advances to 2024-01-10, still not enough.
    assert _dim_in(logic, DimUpsert(_dt(2024, 1, 10), "silver", seq=1)) == []
    # Frontier passes the fact's event time: resolves to the version
    # effective then, not the newest one.
    out = _dim_in(logic, DimUpsert(_dt(2024, 1, 20), "gold", seq=2))
    assert out == [
        TemporalJoined(
            fact=fact, dim="silver", effective_at=_dt(2024, 1, 10), status="matched"
        )
    ]


def test_fact_resolves_immediately_when_frontier_passed() -> None:
    logic = _logic()
    _dim_in(logic, DimUpsert(_dt(2024, 1, 1), "silver", seq=1))
    _dim_in(logic, DimWatermark(_dt(2024, 1, 31)))
    fact = _fact("f1", _dt(2024, 1, 15))
    out = _fact_in(logic, fact)
    assert out == [
        TemporalJoined(
            fact=fact, dim="silver", effective_at=_dt(2024, 1, 1), status="matched"
        )
    ]


def test_same_effective_time_ordered_by_seq() -> None:
    # Versions with the same effective time resolve deterministically
    # to the greatest `seq`, regardless of arrival order.
    for seqs in [(1, 2), (2, 1)]:
        logic = _logic()
        for seq in seqs:
            _dim_in(logic, DimUpsert(_dt(2024, 1, 5), f"v{seq}", seq=seq))
        version = logic._version_at(_dt(2024, 1, 5))
        assert version is not None
        assert version.value == "v2"
        assert version.seq == 2


def test_redelivered_dim_event_is_idempotent() -> None:
    logic = _logic()
    _dim_in(logic, DimUpsert(_dt(2024, 1, 5), "v1", seq=1))
    _dim_in(logic, DimUpsert(_dt(2024, 1, 5), "v1", seq=1))
    assert len(logic.state.versions) == 1


def test_delete_marker() -> None:
    logic = _logic()
    _dim_in(logic, DimUpsert(_dt(2024, 1, 1), "silver", seq=1))
    _dim_in(logic, DimDelete(_dt(2024, 1, 10), seq=2))
    before = _fact("f_before", _dt(2024, 1, 5))
    assert _fact_in(logic, before) == [
        TemporalJoined(
            fact=before, dim="silver", effective_at=_dt(2024, 1, 1), status="matched"
        )
    ]
    _dim_in(logic, DimWatermark(_dt(2024, 1, 31)))
    after = _fact("f_after", _dt(2024, 1, 15))
    assert _fact_in(logic, after) == [
        TemporalJoined(
            fact=after, dim=None, effective_at=_dt(2024, 1, 10), status="deleted"
        )
    ]


def test_allowed_lateness_delays_derived_frontier() -> None:
    logic = _logic(allowed_lateness=timedelta(days=2))
    # Derived frontier is 2024-01-08, not 2024-01-10.
    assert _dim_in(logic, DimUpsert(_dt(2024, 1, 10), "v", seq=1)) == []
    fact = _fact("f1", _dt(2024, 1, 10))
    assert _fact_in(logic, fact) == []
    out = _dim_in(logic, DimWatermark(_dt(2024, 1, 10)))
    assert out == [
        TemporalJoined(
            fact=fact, dim="v", effective_at=_dt(2024, 1, 10), status="matched"
        )
    ]


def test_watermark_does_not_regress_frontier() -> None:
    logic = _logic()
    _dim_in(logic, DimWatermark(_dt(2024, 1, 10)))
    _dim_in(logic, DimWatermark(_dt(2024, 1, 5)))
    assert logic.state.frontier == _dt(2024, 1, 10)


def test_gc_collects_old_versions_but_keeps_floor() -> None:
    logic = _logic(retention=timedelta(days=5))
    for day in range(1, 11):
        _dim_in(logic, DimUpsert(_dt(2024, 1, day), f"t{day}", seq=day))
    # Frontier is 2024-01-10, horizon 2024-01-05: versions before the
    # horizon are collected, except the newest one below it (the floor
    # at 2024-01-04).
    assert [v.effective_at for v in logic.state.versions] == [
        _dt(2024, 1, day) for day in range(4, 11)
    ]
    # Times at or after the horizon still resolve correctly via the
    # floor.
    version = logic._version_at(_dt(2024, 1, 4, 12))
    assert version is not None
    assert version.value == "t4"


def test_gc_never_collects_versions_needed_by_waiting_facts() -> None:
    logic = _logic(retention=timedelta(days=5))
    logic.state.versions = [
        _DimVersion(
            effective_at=_dt(2024, 1, day), seq=day, value=f"t{day}", deleted=False
        )
        for day in range(1, 11)
    ]
    logic.state.frontier = _dt(2024, 1, 10)
    # The frontier is 2024-01-10, so the horizon would be 2024-01-05.
    # A fact still waiting at 2024-01-03 caps the horizon: nothing at
    # or after its event time may be collected. (Waiting facts always
    # have event times beyond the frontier, so this can only happen
    # through a crafted state; the cap is defense-in-depth.)
    logic.state.pending.append(
        _PendingFact(_dt(2024, 1, 3), _fact("f", _dt(2024, 1, 3)))
    )
    logic._collect_versions()
    assert [v.effective_at for v in logic.state.versions] == [
        _dt(2024, 1, day) for day in range(2, 11)
    ]


def test_waiting_facts_resolved_before_versions_collected() -> None:
    logic = _logic(retention=timedelta(days=5))
    _dim_in(logic, DimUpsert(_dt(2024, 1, 1), "silver", seq=1))
    fact = _fact("f1", _dt(2024, 1, 10))
    assert _fact_in(logic, fact) == []
    # The frontier jumps to 2024-01-30, moving the retention horizon
    # to 2024-01-25, past the waiting fact. The fact is resolved
    # first, against the version effective at its event time.
    out = _dim_in(logic, DimWatermark(_dt(2024, 1, 30)))
    assert out == [
        TemporalJoined(
            fact=fact, dim="silver", effective_at=_dt(2024, 1, 1), status="matched"
        )
    ]
    # Afterwards the history is collected down to the floor.
    assert [v.effective_at for v in logic.state.versions] == [_dt(2024, 1, 1)]


def test_snapshot_restore_resolves_identically() -> None:
    logic = _logic()
    _dim_in(logic, DimUpsert(_dt(2024, 1, 1), "silver", seq=1))
    fact = _fact("f1", _dt(2024, 1, 15))
    assert _fact_in(logic, fact) == []

    restored = _logic(state=logic.snapshot())
    out = _dim_in(restored, DimWatermark(_dt(2024, 1, 31)))
    assert out == [
        TemporalJoined(
            fact=fact, dim="silver", effective_at=_dt(2024, 1, 1), status="matched"
        )
    ]
    # The snapshot is independent of the live state.
    _dim_in(restored, DimUpsert(_dt(2024, 2, 1), "gold", seq=2))
    assert [v.effective_at for v in logic.state.versions] == [_dt(2024, 1, 1)]


def test_unknown_dim_event_rejected() -> None:
    logic = _logic()
    with raises(TypeError, match="DimUpsert"):
        logic.on_item((_SIDE_DIM, "not-an-event"))


def test_joins_version_effective_at_fact_time(entry_point) -> None:
    joined: List[Any] = []
    expired: List[Any] = []
    facts_l = [
        ("c1", _fact("f_before", _dt(2023, 12, 31))),
        ("c1", _fact("f_at_t1", _dt(2024, 1, 1))),
        ("c1", _fact("f_mid", _dt(2024, 1, 15))),
        ("c1", _fact("f_at_t2", _dt(2024, 2, 1))),
        ("c1", _fact("f_after", _dt(2024, 3, 1))),
    ]
    dim_l = [
        ("c1", DimUpsert(_dt(2024, 1, 1), "silver", seq=1)),
        ("c1", DimUpsert(_dt(2024, 2, 1), "gold", seq=2)),
        ("c1", DimWatermark(_dt(2024, 3, 31))),
    ]
    flow = _build_flow(facts_l, dim_l, joined, expired, retention=timedelta(days=365))
    entry_point(flow)
    assert joined == [
        (
            "c1",
            TemporalJoined(
                fact=_fact("f_before", _dt(2023, 12, 31)),
                dim=None,
                effective_at=None,
                status="no_version",
            ),
        ),
        (
            "c1",
            TemporalJoined(
                fact=_fact("f_at_t1", _dt(2024, 1, 1)),
                dim="silver",
                effective_at=_dt(2024, 1, 1),
                status="matched",
            ),
        ),
        (
            "c1",
            TemporalJoined(
                fact=_fact("f_mid", _dt(2024, 1, 15)),
                dim="silver",
                effective_at=_dt(2024, 1, 1),
                status="matched",
            ),
        ),
        (
            "c1",
            TemporalJoined(
                fact=_fact("f_at_t2", _dt(2024, 2, 1)),
                dim="gold",
                effective_at=_dt(2024, 2, 1),
                status="matched",
            ),
        ),
        (
            "c1",
            TemporalJoined(
                fact=_fact("f_after", _dt(2024, 3, 1)),
                dim="gold",
                effective_at=_dt(2024, 2, 1),
                status="matched",
            ),
        ),
    ]
    assert expired == []


def test_historical_fact_not_joined_to_newer_version(entry_point) -> None:
    # Regression test: the dimension has already moved on to "gold"
    # when the historical transaction arrives; it must still be
    # attributed to the tier in effect at the transaction time.
    joined: List[Any] = []
    expired: List[Any] = []
    dim_l = [
        ("c1", DimUpsert(_dt(2024, 1, 1), "silver", seq=1)),
        ("c1", DimUpsert(_dt(2024, 2, 1), "gold", seq=2)),
        ("c1", DimWatermark(_dt(2024, 3, 1))),
    ]
    facts_l = [("c1", _fact("old_tx", _dt(2024, 1, 15)))]
    flow = _build_flow(facts_l, dim_l, joined, expired, retention=timedelta(days=365))
    entry_point(flow)
    fact = _fact("old_tx", _dt(2024, 1, 15))
    assert joined == [
        (
            "c1",
            TemporalJoined(
                fact=fact,
                dim="silver",
                effective_at=_dt(2024, 1, 1),
                status="matched",
            ),
        )
    ]
    assert expired == []


def test_fact_older_than_retention_goes_to_expired(recovery_config) -> None:
    joined: List[Any] = []
    expired: List[Any] = []
    # First execution: only dimension events, so the frontier and the
    # retention horizon are established and snapshotted before any
    # fact arrives.
    facts_l = [
        TestingSource.EOF(),
        ("c1", _fact("too_old", _dt(2024, 1, 10))),
        ("c1", _fact("recent", _dt(2024, 1, 25))),
    ]
    dim_l = [
        ("c1", DimUpsert(_dt(2024, 1, 1), "silver", seq=1)),
        ("c1", DimWatermark(_dt(2024, 1, 31))),
        TestingSource.EOF(),
    ]
    flow = _build_flow(facts_l, dim_l, joined, expired, retention=timedelta(days=10))
    run_main(flow, epoch_interval=ZERO_TD, recovery_config=recovery_config)
    assert joined == []
    assert expired == []
    # Second execution: the frontier is 2024-01-31, so the retention
    # horizon is 2024-01-21. The older fact falls below it and is
    # handled explicitly instead of being joined to a collected
    # version.
    run_main(flow, epoch_interval=ZERO_TD, recovery_config=recovery_config)
    too_old = _fact("too_old", _dt(2024, 1, 10))
    assert expired == [
        (
            "c1",
            TemporalExpired(
                fact=too_old, event_at=_dt(2024, 1, 10), horizon=_dt(2024, 1, 21)
            ),
        )
    ]
    recent = _fact("recent", _dt(2024, 1, 25))
    assert joined == [
        (
            "c1",
            TemporalJoined(
                fact=recent,
                dim="silver",
                effective_at=_dt(2024, 1, 1),
                status="matched",
            ),
        )
    ]


def test_keys_are_independent() -> None:
    joined: List[Any] = []
    expired: List[Any] = []
    dim_l = [
        ("c1", DimUpsert(_dt(2024, 1, 1), "silver", seq=1)),
        ("c2", DimUpsert(_dt(2024, 1, 5), "gold", seq=1)),
        ("c2", DimDelete(_dt(2024, 1, 10), seq=2)),
        ("c1", DimWatermark(_dt(2024, 1, 31))),
        ("c2", DimWatermark(_dt(2024, 1, 31))),
    ]
    facts_l = [
        ("c1", _fact("f1", _dt(2024, 1, 15))),
        ("c2", _fact("f2", _dt(2024, 1, 7))),
        ("c2", _fact("f3", _dt(2024, 1, 12))),
    ]
    flow = _build_flow(facts_l, dim_l, joined, expired)
    run_main(flow, epoch_interval=ZERO_TD)
    f1 = _fact("f1", _dt(2024, 1, 15))
    f2 = _fact("f2", _dt(2024, 1, 7))
    f3 = _fact("f3", _dt(2024, 1, 12))
    by_tx = sorted(joined, key=lambda kv: kv[1].fact["tx"])
    assert by_tx == [
        (
            "c1",
            TemporalJoined(
                fact=f1,
                dim="silver",
                effective_at=_dt(2024, 1, 1),
                status="matched",
            ),
        ),
        (
            "c2",
            TemporalJoined(
                fact=f2,
                dim="gold",
                effective_at=_dt(2024, 1, 5),
                status="matched",
            ),
        ),
        (
            "c2",
            TemporalJoined(
                fact=f3,
                dim=None,
                effective_at=_dt(2024, 1, 10),
                status="deleted",
            ),
        ),
    ]
    assert expired == []


def test_pending_facts_flush_at_eof() -> None:
    joined: List[Any] = []
    expired: List[Any] = []
    # No watermarks; the fact's event time is beyond every dimension
    # event, so it can only be resolved at EOF.
    dim_l = [("c1", DimUpsert(_dt(2024, 1, 1), "silver", seq=1))]
    facts_l = [("c1", _fact("f1", _dt(2024, 6, 1)))]
    flow = _build_flow(facts_l, dim_l, joined, expired)
    run_main(flow, epoch_interval=ZERO_TD)
    fact = _fact("f1", _dt(2024, 6, 1))
    assert joined == [
        (
            "c1",
            TemporalJoined(
                fact=fact,
                dim="silver",
                effective_at=_dt(2024, 1, 1),
                status="matched",
            ),
        )
    ]
    assert expired == []


def test_state_survives_continuation(recovery_config) -> None:
    joined: List[Any] = []
    expired: List[Any] = []
    facts_l = [
        ("c1", _fact("f1", _dt(2024, 1, 15))),
        TestingSource.EOF(),
        ("c1", _fact("f2", _dt(2024, 2, 15))),
    ]
    dim_l = [
        ("c1", DimUpsert(_dt(2024, 1, 1), "silver", seq=1)),
        TestingSource.EOF(),
        ("c1", DimUpsert(_dt(2024, 2, 1), "gold", seq=2)),
        ("c1", DimWatermark(_dt(2024, 3, 1))),
    ]
    flow = _build_flow(facts_l, dim_l, joined, expired)
    run_main(flow, epoch_interval=ZERO_TD, recovery_config=recovery_config)
    # First execution: the fact waits past the only known version and
    # is flushed at EOF.
    f1 = _fact("f1", _dt(2024, 1, 15))
    assert joined == [
        (
            "c1",
            TemporalJoined(
                fact=f1,
                dim="silver",
                effective_at=_dt(2024, 1, 1),
                status="matched",
            ),
        )
    ]
    # Second execution: state was retained, so the next fact sees both
    # versions and the new watermark.
    run_main(flow, epoch_interval=ZERO_TD, recovery_config=recovery_config)
    f2 = _fact("f2", _dt(2024, 2, 15))
    assert joined[1:] == [
        (
            "c1",
            TemporalJoined(
                fact=f2,
                dim="gold",
                effective_at=_dt(2024, 2, 1),
                status="matched",
            ),
        )
    ]
    assert expired == []


def test_recovery_after_abort_matches_uninterrupted(recovery_config) -> None:
    dim_l = [
        ("c1", DimUpsert(_dt(2024, 1, 1), "silver", seq=1)),
        ("c1", DimUpsert(_dt(2024, 2, 1), "gold", seq=2)),
        ("c1", DimWatermark(_dt(2024, 12, 31))),
    ]
    facts_l = [
        ("c1", _fact("f1", _dt(2024, 1, 15))),
        ("c1", _fact("f2", _dt(2024, 3, 15))),
        ("c1", _fact("f3", _dt(2024, 1, 20))),
    ]

    clean_joined: List[Any] = []
    clean_expired: List[Any] = []
    flow = _build_flow(
        facts_l, dim_l, clean_joined, clean_expired, retention=timedelta(days=365)
    )
    run_main(flow, epoch_interval=ZERO_TD)

    aborted_facts_l = [
        ("c1", _fact("f1", _dt(2024, 1, 15))),
        ("c1", _fact("f2", _dt(2024, 3, 15))),
        TestingSource.ABORT(),
        ("c1", _fact("f3", _dt(2024, 1, 20))),
    ]
    resumed_joined: List[Any] = []
    resumed_expired: List[Any] = []
    flow = _build_flow(
        aborted_facts_l,
        dim_l,
        resumed_joined,
        resumed_expired,
        retention=timedelta(days=365),
    )
    # f2 is still waiting for the watermark when the abort hits; its
    # buffered state must survive recovery and resolve identically.
    run_main(flow, epoch_interval=ZERO_TD, recovery_config=recovery_config)
    run_main(flow, epoch_interval=ZERO_TD, recovery_config=recovery_config)

    # The join results are identical to the uninterrupted run; only
    # the emission order across the abort boundary may differ.
    by_tx = lambda joined: sorted(joined, key=lambda kv: kv[1].fact["tx"])
    assert by_tx(resumed_joined) == by_tx(clean_joined)
    assert resumed_expired == clean_expired


def test_batch_size_does_not_change_results() -> None:
    facts_l = [
        ("c1", _fact("f1", _dt(2024, 1, 15))),
        ("c1", _fact("f2", _dt(2024, 2, 15))),
        ("c1", _fact("f3", _dt(2023, 12, 15))),
    ]
    dim_l = [
        ("c1", DimUpsert(_dt(2024, 1, 1), "silver", seq=1)),
        ("c1", DimUpsert(_dt(2024, 2, 1), "gold", seq=2)),
        ("c1", DimWatermark(_dt(2024, 3, 1))),
    ]
    outputs = []
    for batch_size in (1, 3):
        joined: List[Any] = []
        expired: List[Any] = []
        flow = _build_flow(
            facts_l,
            dim_l,
            joined,
            expired,
            retention=timedelta(days=365),
            batch_size=batch_size,
        )
        run_main(flow, epoch_interval=ZERO_TD)
        # The join results don't depend on batching; only the
        # emission order across the two merged streams does.
        by_tx = sorted(joined, key=lambda kv: kv[1].fact["tx"])
        outputs.append((by_tx, expired))
    assert outputs[0] == outputs[1]


def test_negative_retention_rejected() -> None:
    with raises(ValueError, match="retention"):
        _build_flow([], [], [], [], retention=timedelta(days=-1))


def test_negative_allowed_lateness_rejected() -> None:
    with raises(ValueError, match="allowed_lateness"):
        _build_flow([], [], [], [], allowed_lateness=timedelta(days=-1))
