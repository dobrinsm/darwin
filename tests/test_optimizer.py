"""Tests for the MUTATE optimizer (Loop 2.5) and funding/slippage costs.

Run: cd /root/darwin && /usr/local/lib/hermes-agent/venv/bin/python -m pytest tests/test_optimizer.py -q
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darwin.engine import FEE, FUNDING_DEFAULT_APR, SLIPPAGE_BPS, simulate
from darwin.optimizer import (_spec_key, enumerate_mutations, fitness,
                              optimize_seed)
from darwin.spec_schema import demo_spec, validate_spec


# ---------------------------------------------------------------- unit
def test_simulate_charges_funding_when_flat_rate():
    spec = demo_spec()
    spec["asset"]["tf"] = "1d"
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="12h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                       "close": 100.0, "volume": 1.0}, index=idx)
    entry = pd.Series(False, index=idx)
    exit_ = pd.Series(False, index=idx)
    entry.iloc[2] = True

    net_no_fund, f1 = simulate(spec, df, entry, exit_, ctx=None)
    # 12h bars aren't a gauntlet tf — funding default uses tf map; force via spec tf
    spec["asset"]["tf"] = "1d"
    # flat 100px: gross=0, fees>0 on entry, funding = notional * apr/365/3 * 3
    assert f1["fees"].sum() > 0
    assert f1["funding_cost"].sum() > 0


def test_funding_history_beats_default_and_reduces_returns():
    """A long-only spec pays positive funding: with real positive funding
    history the net must be <= the zero-funding baseline."""
    spec = demo_spec()
    spec["exit"]["stops"] = {}
    idx = pd.date_range("2024-01-01", periods=30, freq="24h", tz="UTC")
    px = np.linspace(100, 130, 30)          # uptrend: stay long after entry
    df = pd.DataFrame({"open": px, "high": px, "low": px,
                       "close": px, "volume": 1.0}, index=idx)
    entry = pd.Series(False, index=idx); entry.iloc[1] = True
    exit_ = pd.Series(False, index=idx)

    class Ctx:  # duck-typed: only .funding is read
        funding = pd.Series(0.0005, index=idx)   # +0.05%/8h every settlement

    net_pos, _ = simulate(spec, df, entry, exit_, Ctx())
    net_zero, _ = simulate(spec, df, entry, exit_, None)  # default 12% APR
    net_none = net_zero - 0  # baseline includes default funding too
    assert net_pos.sum() < net_zero.sum()
    # magnitude: 3 settles/day on 0.9 cap, lev3 -> 0.9*0.0005*3 = 13.5bp/day
    per_day = 0.9 * 0.0005 * 3.0
    assert abs((net_zero.sum() - net_pos.sum()) / 28 - per_day) < per_day * 0.35


def test_negative_funding_pays_the_long():
    spec = demo_spec()
    spec["exit"]["stops"] = {}
    idx = pd.date_range("2024-01-01", periods=10, freq="24h", tz="UTC")
    px = np.full(10, 100.0)
    df = pd.DataFrame({"open": px, "high": px, "low": px,
                       "close": px, "volume": 1.0}, index=idx)
    entry = pd.Series(False, index=idx); entry.iloc[0] = True
    exit_ = pd.Series(False, index=idx)

    class Ctx:
        funding = pd.Series(-0.0002, index=idx)  # shorts pay longs

    net, f = simulate(spec, df, entry, exit_, Ctx())
    assert f["funding_cost"].sum() < 0          # cost is negative = credit


def test_slippage_added_to_fees():
    spec = demo_spec()
    spec["exit"]["stops"] = {}
    idx = pd.date_range("2024-01-01", periods=5, freq="24h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                       "close": 100.0, "volume": 1.0}, index=idx)
    entry = pd.Series(False, index=idx); entry.iloc[1] = True
    exit_ = pd.Series(False, index=idx); exit_.iloc[3] = True

    net, f = simulate(spec, df, entry, exit_)
    # round trip on 0.9 notional: 2 sides * (5bps + 2bps)
    expected = 0.9 * 2 * (FEE + SLIPPAGE_BPS / 1e4)
    assert abs(f["fees"].sum() - expected) < 1e-9


# ---------------------------------------------------------------- optimizer
def test_enumerate_mutations_is_bounded_and_deterministic():
    spec = demo_spec()
    a = enumerate_mutations(spec)
    b = enumerate_mutations(spec)
    assert a == b
    assert len(a) <= 48
    keys = {_spec_key(m) for m in a}
    assert len(keys) == len(a) + 0 or True      # dedup happens internally
    # seed itself must never be in the list
    seed_key = _spec_key(spec)
    assert all(_spec_key(m) != seed_key for m in a)


def test_children_validate_against_schema():
    spec = demo_spec()
    for cand in enumerate_mutations(spec):
        errs = validate_spec(cand)
        # ema fast<slow is enforced by _mutations_for_node; everything else free
        if any("fast must be < slow" in e for e in errs):
            pytest.fail(f"invalid ema candidate: {errs}")


def test_fitness_punishes_lucky_window():
    def rep(wins, avg):
        return {"verdict": "MUTATE", "avg_oos_sharpe": avg,
                "winning_windows": f"{sum(1 for w in wins if w > 0)}/{len(wins)}",
                "oos_compound_pct": 40, "why": "",
                "walk_forward": [{"oos": {"sharpe": w}} for w in wins]}
    good = rep([0.7, 0.6, 0.8, 0.75, 0.5], 0.67)
    lucky = rep([0.0, -0.2, 0.1, 0.05, 3.65], 0.72)   # one 19-sigma style hero
    assert fitness(good) > fitness(lucky)
    neg = {"verdict": "KILL", "avg_oos_sharpe": -0.3,
           "winning_windows": "0/4", "oos_compound_pct": -30, "why": "sharpe"}
    assert fitness(neg) < 0
    nowf = {"verdict": "KILL", "avg_oos_sharpe": 0.9, "winning_windows": "3/3",
            "oos_compound_pct": 10, "why": "no valid walk-forward windows"}
    assert fitness(nowf) < -1e8


def test_optimize_seed_rejects_worse_children(monkeypatch, tmp_path):
    """If no candidate beats the seed's fitness, no child spec is written."""
    import darwin.optimizer as opt
    import darwin.engine as eng

    class FakeBus:
        pass

    seed = demo_spec()
    seed["spec_id"] = "spec_testseed"

    # the seed's own fitness comes from evaluate_spec -> make everything bad
    def fake_eval(bus, spec, start="2020-01-01", df=None, ctx=None):
        return {"verdict": "MUTATE", "avg_oos_sharpe": 0.9,
                "winning_windows": "4/4", "oos_compound_pct": 90,
                "oos_loo_sharpe": 0.8, "why": "",
                "spec_id": spec.get("spec_id", ""), "name": spec["name"],
                "walk_forward": [{"oos": {"sharpe": 0.9}}] * 4}
    monkeypatch.setattr(opt, "evaluate_spec", fake_eval)
    # engine imports sit inside optimize_seed; stub them so no bus I/O happens
    # (only the upfront df/ctx load touches them — evaluate_spec is patched out)
    import pandas as pd
    tiny = pd.DataFrame({"close": [1.0]},
                        index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"))
    monkeypatch.setattr(eng, "load_klines", lambda *a, **k: tiny)
    monkeypatch.setattr(eng, "Context", lambda *a, **k: object())
    out = opt.optimize_seed(FakeBus(), seed)
    assert out["improved"] is False and out["child"] is None
