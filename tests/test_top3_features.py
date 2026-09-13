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

    hold_spec = demo_spec()
    hold_spec["exit"]["max_hold_bars"] = 0
    assert any("max_hold_bars" in error for error in validate_spec(hold_spec))


# ------------------------------------------------------------- funding nodes
def test_funding_nodes_use_only_latest_known_settlement():
    idx = pd.date_range("2024-01-01 12:00", periods=3, freq="24h", tz="UTC")
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
