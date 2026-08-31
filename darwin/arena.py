"""Loop 3 — the Arena. Champions paper-trade; every position change is logged.

State: /root/darwin/data/arena_state.json  {spec_id: {in_pos, entry_px, entry_ts, equity}}
Trades: /root/darwin/data/arena_trades.jsonl (append-only)
Virtual book: 10,000 USDT margin per spec, spec's own leverage, long-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .bus import EventBus
from .engine import Context, compile_signal, load_klines, simulate
from .spec_schema import SPEC_DIR, load_spec

DATA = Path(__file__).resolve().parent.parent / "data"
STATE_P = DATA / "arena_state.json"
TRADES_P = DATA / "arena_trades.jsonl"
MARGIN = 10_000.0
START_EQ = 10_000.0

# promoted = verdict PROMOTE and not older than STALE_HOURS
STALE_HOURS = 36


def promoted_specs() -> list[dict]:
    out = []
    now = time.time()
    for p in SPEC_DIR.glob("*.json"):
        rep_p = Path(__file__).resolve().parent.parent / "runs" / p.stem / "report.json"
        if not rep_p.exists():
            continue
        rep = json.loads(rep_p.read_text())
        if rep.get("verdict") == "PROMOTE" and now - rep.get("ran_at", 0) < STALE_HOURS * 3600:
            s = load_spec(p.stem)
            out.append(s)
    return out


def _load_state() -> dict:
    if STATE_P.exists():
        return json.loads(STATE_P.read_text())
    return {}


def _save_state(st: dict):
    DATA.mkdir(parents=True, exist_ok=True)
    STATE_P.write_text(json.dumps(st, indent=1))


def _log_trade(row: dict):
    with TRADES_P.open("a") as f:
        f.write(json.dumps(row) + "\n")


def step(bus: EventBus | None = None) -> dict:
    """One arena cycle: reconcile desired positions for every promoted spec."""
    bus = bus or EventBus()
    state = _load_state()
    actions = []
    for spec in promoted_specs():
        sid = spec["spec_id"]
        sym, tf = spec["asset"]["symbol"], spec["asset"]["tf"]
        lev = spec["risk"]["leverage"]
        try:
            df = load_klines(bus, sym, tf)
            ctx = Context(bus, sym, tf, df.index)
            e_sig, x_sig, gaps = compile_signal(spec, df, ctx)
            net, frame = simulate(spec, df, e_sig, x_sig)
        except Exception as e:
            actions.append({"spec_id": sid, "error": str(e)[:120]})
            continue
        desired = int(frame["position"].iloc[-1])       # what spec holds now
        last_close = float(df["close"].iloc[-1])
        last_ts = float(df.index[-1].timestamp())
        st = state.get(sid, {"in_pos": 0, "entry_px": None, "entry_ts": None,
                             "equity": START_EQ})

        if desired == 1 and not st["in_pos"]:
            st.update(in_pos=1, entry_px=last_close, entry_ts=last_ts)
            _log_trade({"ts": last_ts, "spec_id": sid, "name": spec["name"],
                        "action": "ENTRY", "symbol": sym, "px": last_close,
                        "lev": lev, "margin": MARGIN, "gaps": gaps})
            actions.append({"spec_id": sid, "action": "ENTRY", "px": last_close})
        elif desired == 0 and st["in_pos"]:
            pnl = (last_close / st["entry_px"] - 1) * lev * MARGIN if st["entry_px"] else 0.0
            st["equity"] = round(st.get("equity", START_EQ) + pnl, 2)
            _log_trade({"ts": last_ts, "spec_id": sid, "name": spec["name"],
                        "action": "EXIT", "symbol": sym, "px": last_close,
                        "pnl_usd": round(pnl, 2), "equity": st["equity"]})
            actions.append({"spec_id": sid, "action": "EXIT", "px": last_close,
                            "pnl_usd": round(pnl, 2)})
            st.update(in_pos=0, entry_px=None, entry_ts=None)
        state[sid] = st

    # drop state for specs no longer promoted
    live = {s["spec_id"] for s in promoted_specs()}
    for sid in list(state):
        if sid not in live and state[sid].get("in_pos"):
            state[sid]["in_pos"] = 0
    _save_state(state)
    return {"actions": actions,
            "positions": {k: v for k, v in state.items() if v.get("in_pos")}}
