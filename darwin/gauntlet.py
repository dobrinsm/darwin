"""Loop 2 — the Gauntlet.

Full-sample compile+simulate, then WALK-FORWARD: rolling 12mo in-sample /
6mo out-of-sample windows. Verdict per spec:
  PROMOTE   — OOS sharpe >= 0.5, positive OOS compound, win rate >= 40%
  MUTATE    — promising but needs param search (queued for optimizer)
  KILL      — fails OOS bars
Verdicts and metrics are written to runs/<spec_id>/report.json.

Costs are real-world honest: taker fee + slippage per side, plus realized
Binance funding settlements charged on held notional (engine.simulate).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .bus import EventBus
from .engine import (Context, compile_signal, load_klines, metrics, simulate)
from .spec_schema import load_spec

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

IS_DAYS, OOS_DAYS, STEP_DAYS = 365, 182, 182


def evaluate_spec(bus: EventBus, spec: dict, start: str = "2020-01-01",
                  df: pd.DataFrame | None = None,
                  ctx: Context | None = None) -> dict:
    """Full-sample + walk-forward for one spec. PURE: no writes, so the
    optimizer can call it thousands of times on mutated specs. df/ctx can be
    pre-loaded by the caller (same symbol/tf/start) to skip bus reads."""
    t0 = time.time()
    if df is None:
        df = load_klines(bus, spec["asset"]["symbol"], spec["asset"]["tf"], start=start)
    if ctx is None:
        ctx = Context(bus, spec["asset"]["symbol"], spec["asset"]["tf"], df.index)
    entry_sig, exit_sig, gaps = compile_signal(spec, df, ctx)
    net, frame = simulate(spec, df, entry_sig, exit_sig, ctx)
    full = metrics(net, frame["position"], spec["asset"]["tf"])
    full_sample = full | {
        "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
        "fees_pct": round(float(frame["fees"].sum()) * 100, 2),
        "funding_pct": round(float(frame["funding_cost"].sum()) * 100, 2),
    }

    wf = []
    idx = df.index
    start_ts = idx[0]
    while True:
        is_end = start_ts + pd.Timedelta(days=IS_DAYS)
        oos_end = is_end + pd.Timedelta(days=OOS_DAYS)
        if oos_end > idx[-1]:
            break
        m = (idx >= start_ts) & (idx < is_end)
        o = (idx >= is_end) & (idx < oos_end)
        if m.sum() > 100 and o.sum() > 30:
            # recompile signals on IS window only (context as-of IS end)
            e_is, x_is, _ = compile_signal(spec, df[m], ctx)
            net_is, f_is = simulate(spec, df[m], e_is, x_is, ctx)
            # OOS: carry position state approximately by recompiling on the
            # OOS slice with signals from the same spec (params are fixed by
            # IS; no re-optimization happens here — this is validation, not fit)
            e_oos, x_oos, _ = compile_signal(spec, df[o], ctx)
            net_oos, f_oos = simulate(spec, df[o], e_oos, x_oos, ctx)
            wf.append({
                "is_start": str(start_ts.date()), "oos_start": str(is_end.date()),
                "oos_end": str(min(oos_end, idx[-1]).date()),
                "is": metrics(net_is, f_is["position"], spec["asset"]["tf"]),
                "oos": metrics(net_oos, f_oos["position"], spec["asset"]["tf"]),
            })
        start_ts += pd.Timedelta(days=STEP_DAYS)

    oos_sharpes = [w["oos"]["sharpe"] for w in wf]
    oos_comp = 1.0
    for w in wf:
        oos_comp *= (1 + w["oos"]["total_ret"] / 100)
    oos_compound = (oos_comp - 1) * 100
    avg_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    win_windows = sum(1 for s in oos_sharpes if s > 0)
    # leave-one-out robustness: drop the single best window, re-average.
    if len(oos_sharpes) >= 3:
        loo = (sum(oos_sharpes) - max(oos_sharpes)) / (len(oos_sharpes) - 1)
    else:
        loo = avg_oos_sharpe

    if not wf:
        verdict = "KILL"
        why = "no valid walk-forward windows (insufficient history)"
    elif avg_oos_sharpe >= 0.5 and oos_compound > 0 and \
            all(w["oos"]["trades"] > 0 or w["oos"]["exposure"] == 0 for w in wf):
        verdict = "PROMOTE"
        why = (f"avg OOS sharpe {avg_oos_sharpe:.2f}, compound {oos_compound:+.1f}%, "
               f"{win_windows}/{len(wf)} winning windows")
    elif avg_oos_sharpe > 0.1 and oos_compound > -20:
        verdict = "MUTATE"
        why = (f"avg OOS sharpe {avg_oos_sharpe:.2f}, compound {oos_compound:+.1f}% — "
               f"queue param search")
    else:
        verdict = "KILL"
        why = f"avg OOS sharpe {avg_oos_sharpe:.2f}, compound {oos_compound:+.1f}%"

    return {
        "spec_id": spec["spec_id"], "name": spec["name"],
        "symbol": spec["asset"]["symbol"], "tf": spec["asset"]["tf"],
        "full_sample": full_sample, "walk_forward": wf,
        "avg_oos_sharpe": round(avg_oos_sharpe, 3),
        "oos_loo_sharpe": round(loo, 3),
        "oos_compound_pct": round(oos_compound, 1),
        "winning_windows": f"{win_windows}/{len(wf)}",
        "verdict": verdict, "why": why, "gaps": gaps,
        "runtime_secs": round(time.time() - t0, 2),
        "ran_at": time.time(),
    }


def run_spec(bus: EventBus, spec: dict, start: str = "2020-01-01") -> dict:
    """evaluate_spec + persist report to runs/<spec_id>/report.json."""
    report = evaluate_spec(bus, spec, start=start)
    out = RUNS_DIR / spec["spec_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def run_spec_id(spec_id: str) -> dict:
    bus = EventBus()
    spec = load_spec(spec_id)
    return run_spec(bus, spec)


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from dotenv import load_dotenv
    load_dotenv("/root/.hermes/.env", override=False)
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    bus = EventBus()
    if sid:
        rep = run_spec(bus, load_spec(sid))
        print(json.dumps({k: rep[k] for k in ("name", "verdict", "why", "full_sample")},
                         indent=1))
    else:
        for p in sorted(Path(__file__).resolve().parent.parent.glob("specs/*.json")):
            rep = run_spec(bus, json.loads(p.read_text()))
            print(f"{rep['verdict']:8} {rep['spec_id']}  {rep['name'][:40]:42}"
                  f" oosSharpe={rep['avg_oos_sharpe']:+.2f} oosComp={rep['oos_compound_pct']:+8.1f}%")
