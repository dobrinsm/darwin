"""Regression tests for funding vetoes, cross-timeframe gates, and time exits."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darwin.engine import _node_to_bool, compile_signal, simulate
from darwin.optimizer import _spec_key, enumerate_mutations
from darwin.spec_schema import demo_spec, validate_spec


def _bars(index, close, volume=None):
    close = np.asarray(close, dtype=float)
    if volume is None:
        volume = np.ones(len(close))
    return pd.DataFrame({
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": np.asarray(volume, dtype=float),
    }, index=index)


class _StubContext:
    def __init__(self, higher=None, funding=None, funding_present=True):
        self.base_tf = "4h"
        self._higher = higher
        self.funding = funding if funding is not None else pd.Series(dtype=float)
        self.sources_present = {"funding": funding_present}

    def tf_df(self, tf):
        assert tf == "1d"
        return self._higher


# ---------------------------------------------------------------- schema
def test_schema_accepts_new_features_and_preserves_legacy_specs():
    legacy = demo_spec()
    assert validate_spec(legacy) == []

    spec = copy.deepcopy(legacy)
    spec["entry"]["all"].append({"type": "funding_below", "threshold": 0.0003})
    spec["exit"]["any"].append({"type": "funding_above", "threshold": 0.0005})
    spec["exit"]["max_hold_bars"] = 24
    spec["entry"]["all"][0]["tf"] = "4h"
    spec["asset"]["tf"] = "4h"
    assert validate_spec(spec) == []


def test_schema_rejects_bad_funding_timeframe_and_hold_ranges():
    funding_spec = demo_spec()
    funding_spec["entry"]["all"].append(
        {"type": "funding_below", "threshold": 0.02})
    errors = validate_spec(funding_spec)
    assert any("funding_below" in error and "out of range" in error for error in errors)

    tf_spec = demo_spec()
    tf_spec["entry"]["all"][0]["tf"] = "1h"
    errors = validate_spec(tf_spec)
    assert any("param 'tf'='1h'" in error for error in errors)

    lower_tf_spec = demo_spec()
    lower_tf_spec["entry"]["all"][0]["tf"] = "4h"
    errors = validate_spec(lower_tf_spec)
    assert any("node tf '4h' is lower than asset tf '1d'" in error
               for error in errors)

    hold_spec = demo_spec()
    hold_spec["exit"]["max_hold_bars"] = 0
    assert any("max_hold_bars" in error for error in validate_spec(hold_spec))


def test_schema_reports_malformed_node_params_without_raising():
    funding_spec = demo_spec()
    funding_spec["entry"]["all"].append(
        {"type": "funding_below", "threshold": "0.0003"})
    funding_errors = validate_spec(funding_spec)
    assert any("funding_below" in error and "wrong type" in error
               for error in funding_errors)

    tf_spec = demo_spec()
    tf_spec["entry"]["all"][0]["tf"] = ["1d"]
    tf_errors = validate_spec(tf_spec)
    assert any("ema_cross_up" in error and "wrong type" in error
               for error in tf_errors)

    bool_spec = demo_spec()
    bool_spec["entry"]["all"].append(
        {"type": "funding_below", "threshold": False})
    bool_errors = validate_spec(bool_spec)
    assert any("funding_below" in error and "wrong type" in error
               for error in bool_errors)


# ------------------------------------------------------------- funding nodes
def test_funding_nodes_use_only_latest_known_settlement():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-01 04:00", tz="UTC"),
        pd.Timestamp("2024-01-01 08:00", tz="UTC"),
        pd.Timestamp("2024-01-03 04:00", tz="UTC"),
    ])
    df = _bars(idx, [100, 100, 100])
    funding_idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-03 00:00", tz="UTC"),
    ])
    funding = pd.Series([0.0001, 0.0008], index=funding_idx)
    ctx = _StubContext(funding=funding)

    above = _node_to_bool(
        {"type": "funding_above", "threshold": 0.0005}, df, ctx)
    below = _node_to_bool(
        {"type": "funding_below", "threshold": 0.0005}, df, ctx)
    assert above.tolist() == [False, False, True]
    assert below.tolist() == [True, True, False]

    # Appending a future settlement cannot alter earlier decisions.
    past_only = pd.Series([0.0001], index=funding_idx[:1])
    past_ctx = _StubContext(funding=past_only)
    past_above = _node_to_bool(
        {"type": "funding_above", "threshold": 0.0005}, df.iloc[:2], past_ctx)
    assert above.iloc[:2].tolist() == past_above.tolist()


def test_stale_funding_cannot_admit_an_entry():
    idx = pd.date_range("2024-01-01 04:00", periods=3, freq="4h", tz="UTC")
    df = _bars(idx, [100, 100, 100])
    funding = pd.Series(
        [0.0001], index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]))

    below = _node_to_bool(
        {"type": "funding_below", "threshold": 0.0005},
        df, _StubContext(funding=funding))

    assert below.tolist() == [True, True, False]


def test_missing_funding_is_false_and_reported_as_data_gap():
    idx = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    df = _bars(idx, [100, 101, 102])
    spec = demo_spec()
    spec["asset"]["tf"] = "4h"
    spec["entry"] = {"all": [{"type": "funding_below", "threshold": 0.0003}]}
    spec["exit"] = {"any": []}
    ctx = _StubContext(funding_present=False)

    entry, _, gaps = compile_signal(spec, df, ctx)
    assert not entry.any()
    assert gaps == ["entry:funding_below:needs_funding"]


def test_miner_snapshot_omits_stale_funding(monkeypatch):
    from types import SimpleNamespace

    import darwin.miner as miner

    now = pd.Timestamp("2024-06-01", tz="UTC").timestamp()
    idx = pd.date_range("2023-01-01", periods=400, freq="24h", tz="UTC")
    bars = _bars(idx, np.linspace(100, 200, len(idx)))

    class Bus:
        funding = []

        def read(self, event_type=None, **_):
            return self.funding if event_type == "funding" else []

    bus = Bus()
    monkeypatch.setattr(miner, "load_klines", lambda *args: bars)
    monkeypatch.setattr(miner, "time", SimpleNamespace(time=lambda: now))

    bus.funding = [SimpleNamespace(
        ts=now - 9 * 3600, payload={"rate": "0.0001"})]
    assert "FUNDING:" not in miner.bus_snapshot(bus)

    bus.funding = [SimpleNamespace(
        ts=now - 8 * 3600, payload={"rate": "0.0001"})]
    assert "FUNDING:" in miner.bus_snapshot(bus)


# ------------------------------------------------------ multi-timeframe gates
def test_daily_regime_gate_waits_for_daily_close_without_lookahead():
    base_idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-01 12:00", tz="UTC"),
        pd.Timestamp("2024-01-02 12:00", tz="UTC"),
        pd.Timestamp("2024-01-02 20:00", tz="UTC"),
        pd.Timestamp("2024-01-03 00:00", tz="UTC"),
    ])
    base = _bars(base_idx, [100, 100, 100, 100])
    daily_idx = pd.date_range("2024-01-01", periods=3, freq="24h", tz="UTC")
    daily = _bars(daily_idx, [1, 3, 0])
    node = {"type": "price_above_sma", "period": 2, "tf": "1d"}

    full = _node_to_bool(node, base, _StubContext(higher=daily))
    assert full.tolist() == [False, True, True, False]

    # The future Jan-03 daily close changes Jan-03, never Jan-02 decisions.
    without_future = _node_to_bool(
        node, base.iloc[:3], _StubContext(higher=daily.iloc[:2]))
    assert full.iloc[:3].tolist() == without_future.tolist()


def test_rsi_and_volume_nodes_honor_their_declared_timeframe():
    base_idx = pd.date_range("2024-01-04", periods=2, freq="4h", tz="UTC")
    base = _bars(base_idx, [100, 100], volume=[1, 1])

    rsi_idx = pd.date_range("2024-01-01", periods=4, freq="24h", tz="UTC")
    falling_daily = _bars(rsi_idx, [5, 4, 3, 2])
    rsi = _node_to_bool(
        {"type": "rsi_below", "period": 2, "threshold": 30, "tf": "1d"},
        base, _StubContext(higher=falling_daily))
    assert rsi.tolist() == [True, True]

    volume_daily = _bars(rsi_idx, [1, 1, 1, 1], volume=[1, 1, 1, 10])
    spike = _node_to_bool(
        {"type": "vol_spike", "mult": 2, "lookback": 3, "tf": "1d"},
        base, _StubContext(higher=volume_daily))
    assert spike.tolist() == [True, True]


# ----------------------------------------------------------- time-based exit
def test_max_hold_bars_caps_exact_number_of_held_returns():
    spec = demo_spec()
    spec["exit"] = {"any": [], "max_hold_bars": 3, "stops": {}}
    idx = pd.date_range("2024-01-01", periods=8, freq="24h", tz="UTC")
    df = _bars(idx, np.linspace(100, 107, len(idx)))
    entry = pd.Series(False, index=idx)
    entry.iloc[1] = True
    exit_ = pd.Series(False, index=idx)

    _, frame = simulate(spec, df, entry, exit_)
    assert frame["position"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    held_return_bars = list(np.flatnonzero(frame["notional"].to_numpy() > 0))
    assert held_return_bars == [2, 3, 4]


def test_omitting_max_hold_keeps_existing_signal_only_behavior():
    spec = demo_spec()
    spec["exit"] = {"any": [], "stops": {}}
    idx = pd.date_range("2024-01-01", periods=8, freq="24h", tz="UTC")
    df = _bars(idx, np.linspace(100, 107, len(idx)))
    entry = pd.Series(False, index=idx)
    entry.iloc[1] = True
    exit_ = pd.Series(False, index=idx)

    _, frame = simulate(spec, df, entry, exit_)
    assert frame["position"].tolist() == [0, 1, 1, 1, 1, 1, 1, 1]


def test_max_hold_reduces_one_shot_slow_bleed_loss():
    idx = pd.date_range("2024-01-01", periods=12, freq="24h", tz="UTC")
    df = _bars(idx, np.linspace(100, 70, len(idx)))
    entry = pd.Series(False, index=idx)
    entry.iloc[1] = True
    exit_ = pd.Series(False, index=idx)

    legacy = demo_spec()
    legacy["exit"] = {"any": [], "stops": {}}
    timed = copy.deepcopy(legacy)
    timed["exit"]["max_hold_bars"] = 3

    legacy_net, _ = simulate(legacy, df, entry, exit_)
    timed_net, _ = simulate(timed, df, entry, exit_)
    assert timed_net.sum() > legacy_net.sum()


def test_walk_forward_preserves_hold_age_at_oos_boundary(monkeypatch):
    import darwin.gauntlet as gauntlet

    idx = pd.date_range("2023-01-01", periods=548, freq="24h", tz="UTC")
    df = _bars(idx, np.full(len(idx), 100.0))
    spec = demo_spec()
    spec["spec_id"] = "spec_boundary"
    spec["exit"] = {"any": [], "max_hold_bars": 24, "stops": {}}

    def fake_compile(_, bars, __):
        entry = pd.Series(
            (bars.index >= idx[360]) & (bars.index <= idx[365]),
            index=bars.index)
        return entry, pd.Series(False, index=bars.index), []

    monkeypatch.setattr(gauntlet, "compile_signal", fake_compile)
    report = gauntlet.evaluate_spec(object(), spec, df=df, ctx=object())

    assert report["walk_forward"][0]["oos"]["exposure"] == 10.4


def test_arena_times_exit_and_reentry_from_live_state(monkeypatch, tmp_path):
    import darwin.arena as arena

    idx = pd.date_range("2024-01-01", periods=10, freq="24h", tz="UTC")
    all_bars = _bars(idx, np.full(len(idx), 100.0))
    spec = demo_spec()
    spec["spec_id"] = "spec_live_hold"
    spec["entry"]["all"].append(
        {"type": "funding_below", "threshold": 0.0005})
    spec["exit"] = {"any": [], "max_hold_bars": 3, "stops": {}}
    spec["risk"]["cooldown_bars"] = 2
    initial = {
        spec["spec_id"]: {
            "in_pos": 1, "entry_px": 100.0,
            "entry_ts": float(idx[2].timestamp()), "equity": 10_000.0,
            "coins": 90.0, "lev": 3, "symbol": "DOGEUSDT",
            "funding_trade": 0.0, "last_funding_ts": float(idx[2].timestamp()),
        }
    }
    current = {
        "bars": all_bars.iloc[:4], "state": initial,
        "persistent": False, "entry_indices": [], "exit_indices": [],
        "funding_age_h": 0,
    }
    saved = {}

    def fake_context(*_):
        funding_ts = current["bars"].index[-1] - pd.Timedelta(
            hours=current["funding_age_h"])
        funding = pd.Series([0.0001], index=pd.DatetimeIndex([funding_ts]))
        return _StubContext(funding=funding)

    def fake_compile(_, bars, __):
        entry = pd.Series(current["persistent"], index=bars.index)
        if not current["persistent"]:
            entry.iloc[0] = True
        for position in current["entry_indices"]:
            if position < len(entry):
                entry.iloc[position] = True
        exit_ = pd.Series(False, index=bars.index)
        for position in current["exit_indices"]:
            if position < len(exit_):
                exit_.iloc[position] = True
        return entry, exit_, []

    def fake_save(state):
        saved["state"] = copy.deepcopy(state)

    monkeypatch.setattr(arena, "promoted_specs", lambda: [spec])
    monkeypatch.setattr(arena, "load_klines", lambda *args: current["bars"])
    monkeypatch.setattr(arena, "Context", fake_context)
    monkeypatch.setattr(arena, "compile_signal", fake_compile)
    monkeypatch.setattr(arena, "_load_state", lambda: copy.deepcopy(current["state"]))
    monkeypatch.setattr(arena, "_save_state", fake_save)
    monkeypatch.setattr(arena, "_load_frozen", lambda: {})
    monkeypatch.setattr(arena, "_save_frozen", lambda _: None)
    monkeypatch.setattr(arena, "_funding_since", lambda *args: [])
    monkeypatch.setattr(arena, "_log_trade", lambda _: None)
    monkeypatch.setattr(arena, "_log_blocked_once", lambda *args: None)
    monkeypatch.setattr(arena, "EQUITY_P", tmp_path / "equity.jsonl")

    before_limit = arena.step(object())
    assert spec["spec_id"] in before_limit["positions"]

    current["bars"] = all_bars.iloc[:6]
    at_limit = arena.step(object())
    assert any(action.get("action") == "EXIT" for action in at_limit["actions"])
    assert spec["spec_id"] not in at_limit["positions"]

    current["state"] = saved["state"]
    current["bars"] = all_bars
    after_limit = arena.step(object())
    assert not any(action.get("action") == "ENTRY" for action in after_limit["actions"])
    assert spec["spec_id"] not in after_limit["positions"]
    closed_state = copy.deepcopy(saved["state"])

    current.update(state=closed_state, persistent=True,
                   entry_indices=[], exit_indices=[], funding_age_h=0,
                   bars=all_bars.iloc[:8])
    cooling_down = arena.step(object())
    assert not any(action.get("action") == "ENTRY"
                   for action in cooling_down["actions"])

    current["state"] = saved["state"]
    current["bars"] = all_bars.iloc[:9]
    cooldown_complete = arena.step(object())
    assert any(action.get("action") == "ENTRY"
               for action in cooldown_complete["actions"])

    current.update(state=closed_state, persistent=False,
                   entry_indices=[8], exit_indices=[7], funding_age_h=0,
                   bars=all_bars.iloc[:9])
    replay_diverged = arena.step(object())
    assert any(action.get("action") == "ENTRY"
               for action in replay_diverged["actions"])

    current.update(state=closed_state, entry_indices=[8], exit_indices=[],
                   bars=all_bars.iloc[:9])
    monkeypatch.setattr(arena, "MAX_TOTAL_POSITIONS", 0)
    blocked = arena.step(object())
    assert any(action.get("action") == "BLOCKED" for action in blocked["actions"])

    current.update(state=saved["state"], bars=all_bars, funding_age_h=12)
    monkeypatch.setattr(arena, "MAX_TOTAL_POSITIONS", 6)
    funding_blocked = arena.step(object())
    assert not any(action.get("action") == "ENTRY"
                   for action in funding_blocked["actions"])
    assert any(action.get("reason") == "funding_guard"
               for action in funding_blocked["actions"])

    current.update(state=saved["state"], funding_age_h=0)
    retried = arena.step(object())
    assert any(action.get("action") == "ENTRY" for action in retried["actions"])


# -------------------------------------------------------------- optimizer
def test_optimizer_mutates_funding_and_time_exit_without_key_collisions():
    spec = demo_spec()
    spec["entry"]["all"].append({"type": "funding_below", "threshold": 0.0003})
    spec["exit"]["max_hold_bars"] = 12
    seed_key = _spec_key(spec)
    mutations = enumerate_mutations(spec)

    assert any(m["entry"]["all"][-1]["threshold"] != 0.0003 for m in mutations)
    assert any(m["exit"]["max_hold_bars"] != 12 for m in mutations)
    assert all(_spec_key(m) != seed_key for m in mutations)

    changed_hold = copy.deepcopy(spec)
    changed_hold["exit"]["max_hold_bars"] = 24
    assert _spec_key(changed_hold) != seed_key


def test_optimizer_pairs_entry_and_exit_node_mutations():
    spec = demo_spec()
    entry_fast = spec["entry"]["all"][0]["fast"]
    exit_fast = spec["exit"]["any"][0]["fast"]

    mutations = enumerate_mutations(spec, cap=500, pair_cap=500)

    assert any(
        candidate["entry"]["all"][0]["fast"] != entry_fast
        and candidate["exit"]["any"][0]["fast"] != exit_fast
        for candidate in mutations
    )
    assert all("fast" not in candidate["exit"] for candidate in mutations)
