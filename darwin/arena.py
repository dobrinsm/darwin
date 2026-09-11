"""Loop 3 — the Arena. Champions paper-trade; every position change is logged.

The book simulates a REAL live perp account as closely as the data allows:
- sizing identical to the backtest engine: notional = min(leverage, 0.9) x equity
- taker fee + slippage (engine constants) charged per side, on fills
- REAL funding settlements deducted from margin every 8h while held
  (actual Binance funding events from the bus, mark-price valued)
- open positions marked to market every tick (unrealized PnL)
- RETIRED specs are force-closed at mark price, PnL booked (meta flag)

State: /root/darwin/data/arena_state.json  {spec_id: {in_pos, entry_px, entry_ts,
  equity, coins, fees_paid, funding_paid, last_funding_ts, mark, unrealized, ...}}
Trades: /root/darwin/data/arena_trades.jsonl (append-only)
Marks:  /root/darwin/data/arena_equity.jsonl (book-level time series, one row/tick)
Virtual book: 10,000 USDT margin per spec, long-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .bus import EventBus
from .engine import (Context, FEE, SLIPPAGE_BPS, compile_signal, load_klines,
                     simulate)
from .spec_schema import SPEC_DIR, load_spec

DATA = Path(__file__).resolve().parent.parent / "data"
STATE_P = DATA / "arena_state.json"
TRADES_P = DATA / "arena_trades.jsonl"
EQUITY_P = DATA / "arena_equity.jsonl"
MARGIN = 10_000.0
START_EQ = 10_000.0

# promoted = verdict PROMOTE and not older than STALE_HOURS
STALE_HOURS = 36

# Best-of-N search inflates the max-statistic: an optimizer child must clear a
# higher OOS bar than an organic spec before it's trusted in the arena.
OPT_CHILD_MIN_SHARPE = 0.7
OPT_CHILD_MIN_LOO = 0.25

# Every spec must clear this LOO bar to trade, not just optimizer children.
# LOO = leave-one-out OOS sharpe: negative means the edge lives in one lucky
# window. Miner-born specs used to skip this gate — 4 of the 7 open positions
# on 2026-09-11 sat in negative-LOO specs, all underwater. Was asymmetrical;
# now universal.
MIN_LOO = 0.1

# Live demotion: a promoted spec whose paper equity falls below this is frozen
# (no new entries; open position closes on its own exit signal) until a fresh
# gauntlet re-judges it. Spec files live in data/frozen_specs.json.
FREEZE_EQUITY = 9_500.0
FROZEN_P = DATA / "frozen_specs.json"


def _load_frozen() -> dict:
    """{spec_id: {frozen_at, equity, reason}} — frozen until unfrozen explicitly
    or re-judged (gauntlet overwrites runs/<sid>/report.json; unfreeze lives in
    orchestrator after gauntlet phase)."""
    if FROZEN_P.exists():
        try:
            return json.loads(FROZEN_P.read_text())
        except Exception:
            return {}
    return {}


def _save_frozen(fr: dict):
    DATA.mkdir(parents=True, exist_ok=True)
    FROZEN_P.write_text(json.dumps(fr, indent=1))


def unfreeze(spec_id: str) -> bool:
    fr = _load_frozen()
    if spec_id in fr:
        del fr[spec_id]
        _save_frozen(fr)
        return True
    return False

# ---------------------------------------------------------------- book risk
# Portfolio-level entry controls. The backtest gauntlet judges each spec in
# isolation; the arena trades them as ONE book. Left ungated, 7 long specs on
# SOL/BTC/XLM fire into the same dip and the "diversified" book is one bet
# (2026-09-11: 6/7 open positions all red together). These rules are
# deterministic and apply to NEW entries only — exits are never blocked.
MAX_PER_SYMBOL = 2        # concurrent open positions sharing a symbol
MAX_TOTAL_POSITIONS = 6   # concurrent open positions across the whole book

# live-account cost model — single source of truth is engine.py
COST_PER_SIDE = FEE + SLIPPAGE_BPS / 1e4
NOTIONAL_CAP = 0.9   # engine.simulate clips notional fraction at 0.9 of equity


# ---------------------------------------------------------------- promoted set
def promoted_specs() -> list[dict]:
    out = []
    now = time.time()
    frozen = _load_frozen()
    for p in SPEC_DIR.glob("*.json"):
        if p.stem in frozen:
            continue
        rep_p = Path(__file__).resolve().parent.parent / "runs" / p.stem / "report.json"
        if not rep_p.exists():
            continue
        rep = json.loads(rep_p.read_text())
        if rep.get("verdict") != "PROMOTE" or now - rep.get("ran_at", 0) >= STALE_HOURS * 3600:
            continue
        if rep.get("oos_loo_sharpe", 0) < MIN_LOO:
            continue
        prov = (load_spec(p.stem).get("provenance") or {})
        if prov.get("model") == "optimizer-grid-v1" and (
                rep.get("avg_oos_sharpe", 0) < OPT_CHILD_MIN_SHARPE
                or rep.get("oos_loo_sharpe", 0) < OPT_CHILD_MIN_LOO):
            continue
        s = load_spec(p.stem)
        out.append(s)
    return out


# ---------------------------------------------------------------- state io
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


# ---------------------------------------------------------------- pnl math
def _funding_since(bus: EventBus, symbol: str, since_ts: float,
                   until_ts: float) -> list:
    """Funding settlements in (since_ts, until_ts] for this symbol, oldest first."""
    base = symbol.replace("USDT", "")
    evs = bus.read(event_type="funding", source="binance", symbol=base,
                   limit=1_000_000)
    return sorted((e for e in evs if since_ts < e.ts <= until_ts),
                  key=lambda e: e.ts)


def _accrue_funding(st: dict, coins: float, fallback_mark: float,
                    fund_events: list) -> float:
    """Deduct real funding settlements from margin. A long pays positive rate:
    payment = rate x coins x mark_price at settlement. Returns total paid
    (positive = cost). Mutates st (equity, funding_paid, last_funding_ts)."""
    paid = 0.0
    for e in fund_events:
        rate = float(e.payload.get("rate") or 0.0)
        mp = e.payload.get("markPrice") or ""
        mark = float(mp) if mp else fallback_mark
        pay = rate * coins * mark
        st["equity"] = round(st["equity"] - pay, 4)
        st["funding_paid"] = round(st.get("funding_paid", 0.0) + pay, 4)
        st["funding_trade"] = round(st.get("funding_trade", 0.0) + pay, 4)
        st["last_funding_ts"] = e.ts
        paid += pay
    return paid


def _init_position(st: dict, equity: float, px: float) -> dict:
    """Open a long exactly like the engine sizes it: notional fraction
    min(lev, 0.9) of equity; entry fee charged immediately."""
    lev = st.get("lev", 1)
    frac = min(float(lev), NOTIONAL_CAP)
    notional = frac * equity
    coins = notional / px
    fees = notional * COST_PER_SIDE
    st.update(coins=round(coins, 10), fees_entry=round(fees, 4),
              fees_paid=round(st.get("fees_paid", 0.0) + fees, 4),
              funding_trade=0.0, last_funding_ts=st.get("entry_ts"),
              equity=round(equity - fees, 4))
    return st


def _close_math(st: dict, px: float) -> dict:
    """Closing math at price px. Funding was already deducted from equity while
    held; this books gross pnl minus exit fees. Returns the breakdown."""
    coins = st.get("coins") or 0.0
    entry = st.get("entry_px") or px
    gross = coins * (px - entry)
    fees_exit = coins * px * COST_PER_SIDE
    net = gross - fees_exit - st.get("funding_trade", 0.0)
    st["equity"] = round(st["equity"] + gross - fees_exit, 4)
    st["fees_paid"] = round(st.get("fees_paid", 0.0) + fees_exit, 4)
    return {"pnl_gross": round(gross, 2), "fees_exit": round(fees_exit, 2),
            "funding": round(st.get("funding_trade", 0.0), 2),
            "pnl_net": round(net, 2)}


# ---------------------------------------------------------------- main step
def step(bus: EventBus | None = None) -> dict:
    """One arena cycle: evaluate every promoted spec, then reconcile exits,
    then fill entries under book-level risk caps (see MAX_PER_SYMBOL /
    MAX_TOTAL_POSITIONS). Mark open positions to market, accrue funding."""
    bus = bus or EventBus()
    state = _load_state()
    actions = []
    live = {s["spec_id"]: s for s in promoted_specs()}
    now = time.time()

    # ---- Phase 1: evaluate every spec -> desired position + gauntlet rank.
    # Rank = LOO sharpe of the latest report: the book fills with the most
    # robust specs first when caps bite, not in arbitrary filesystem order.
    evals = []   # (loo_rank, sid, spec, sym, lev, desired, last_close, last_ts, gaps)
    for spec in live.values():
        sid = spec["spec_id"]
        sym, tf = spec["asset"]["symbol"], spec["asset"]["tf"]
        lev = spec["risk"]["leverage"]
        try:
            df = load_klines(bus, sym, tf)
            ctx = Context(bus, sym, tf, df.index)
            e_sig, x_sig, gaps = compile_signal(spec, df, ctx)
            net, frame = simulate(spec, df, e_sig, x_sig, ctx)
        except Exception as e:
            actions.append({"spec_id": sid, "error": str(e)[:120]})
            continue
        rep_p = Path(__file__).resolve().parent.parent / "runs" / sid / "report.json"
        loo_rank = 0.0
        if rep_p.exists():
            try:
                loo_rank = float(json.loads(rep_p.read_text()).get("oos_loo_sharpe") or 0)
            except Exception:
                pass
        evals.append((loo_rank, sid, spec, sym, lev,
                      int(frame["position"].iloc[-1]),
                      float(df["close"].iloc[-1]),
                      float(df.index[-1].timestamp()),  # type: ignore[attr-defined]
                      gaps))

    # ---- Phase 2: exits + mark-to-market + funding + freeze checks.
    # Exits are NEVER gated by book risk.
    pending_entries = []   # (loo_rank, sid, spec, sym, lev, last_close, last_ts, gaps)
    for loo_rank, sid, spec, sym, lev, desired, last_close, last_ts, gaps in evals:
        st = state.get(sid, {"in_pos": 0, "entry_px": None, "entry_ts": None,
                             "equity": START_EQ})
        # backfill: positions opened before the cost model stamped no `symbol`
        if st.get("in_pos") and not st.get("symbol"):
            st["symbol"] = sym

        # ---- already in a position: mark to market + accrue funding ----
        if st.get("in_pos"):
            if st.get("coins") is None:
                # migrate pre-cost-model state (positions opened before fees/
                # funding were modeled): reconstruct from entry_px
                st["lev"] = st.get("lev") or lev
                _init_position(st, st.get("equity", START_EQ),
                               st.get("entry_px") or last_close)
            funds = _funding_since(bus, sym, st.get("last_funding_ts")
                                   or st.get("entry_ts", 0), last_ts)
            _accrue_funding(st, st["coins"], last_close, funds)
            st["mark"] = last_close
            st["unrealized"] = round(
                st["coins"] * (last_close - (st.get("entry_px") or last_close))
                - st["coins"] * last_close * COST_PER_SIDE, 2)

        # ---- live demotion check: paper equity below FREEZE_EQUITY -> freeze
        if st.get("equity", START_EQ) < FREEZE_EQUITY:
            fr = _load_frozen()
            if sid not in fr:
                fr[sid] = {"frozen_at": now, "equity": st["equity"],
                           "reason": f"equity below {FREEZE_EQUITY:.0f}"}
                _save_frozen(fr)
                _log_trade({"ts": now, "spec_id": sid, "name": spec["name"],
                            "action": "FROZEN", "symbol": sym,
                            "equity": st["equity"], "meta": True})
                actions.append({"spec_id": sid, "action": "FROZEN",
                                "equity": st["equity"]})

        if desired == 0 and st.get("in_pos"):
            m = _close_math(st, last_close)
            m["equity_after"] = st["equity"]
            _log_trade({"ts": last_ts, "spec_id": sid, "name": spec["name"],
                        "action": "EXIT", "symbol": sym, "px": last_close, **m})
            actions.append({"spec_id": sid, "action": "EXIT", "px": last_close,
                            "pnl_usd": m["pnl_net"]})
            st.update(in_pos=0, entry_px=None, entry_ts=None, coins=None,
                      unrealized=0.0, mark=last_close)
        elif desired == 1 and not st.get("in_pos"):
            pending_entries.append((loo_rank, sid, spec, sym, lev,
                                    last_close, last_ts, gaps))
        state[sid] = st

    # ---- Phase 3: entries under book-level caps, best-LOO first.
    open_now = [s for s in state.values()
                if isinstance(s, dict) and s.get("in_pos")]
    n_open = len(open_now)
    sym_count = {}
    for s in open_now:
        sy = s.get("symbol") or "?"
        sym_count[sy] = sym_count.get(sy, 0) + 1

    for loo_rank, sid, spec, sym, lev, last_close, last_ts, gaps in sorted(
            pending_entries, key=lambda e: -e[0]):
        st = state[sid]
        if sid in _load_frozen():
            continue
        if n_open >= MAX_TOTAL_POSITIONS:
            _log_trade({"ts": last_ts, "spec_id": sid, "name": spec["name"],
                        "action": "BLOCKED", "symbol": sym, "px": last_close,
                        "reason": f"book cap {MAX_TOTAL_POSITIONS} positions",
                        "meta": True})
            actions.append({"spec_id": sid, "action": "BLOCKED",
                            "reason": "book_cap"})
            continue
        if sym_count.get(sym, 0) >= MAX_PER_SYMBOL:
            _log_trade({"ts": last_ts, "spec_id": sid, "name": spec["name"],
                        "action": "BLOCKED", "symbol": sym, "px": last_close,
                        "reason": f"symbol cap {MAX_PER_SYMBOL} on {sym}",
                        "meta": True})
            actions.append({"spec_id": sid, "action": "BLOCKED",
                            "reason": "symbol_cap", "symbol": sym})
            continue
        st.update(in_pos=1, entry_px=last_close, entry_ts=last_ts,
                  lev=lev, symbol=sym, funding_trade=0.0)
        _init_position(st, st.get("equity", START_EQ), last_close)
        st["mark"] = last_close
        st["unrealized"] = round(-st["coins"] * last_close * COST_PER_SIDE, 2)
        _log_trade({"ts": last_ts, "spec_id": sid, "name": spec["name"],
                    "action": "ENTRY", "symbol": sym, "px": last_close,
                    "lev": lev, "margin": MARGIN, "coins": round(st["coins"], 6),
                    "fees": st["fees_entry"], "gaps": gaps})
        actions.append({"spec_id": sid, "action": "ENTRY", "px": last_close})
        n_open += 1
        sym_count[sym] = sym_count.get(sym, 0) + 1
        state[sid] = st

    # ---- specs no longer promoted but holding a paper position: force-close
    # at last mark, PnL booked (administrative close, flagged meta) ----
    for sid in list(state):
        if sid in live:
            continue
        st = state[sid]
        if not isinstance(st, dict) or not st.get("in_pos"):
            continue
        sym = st.get("symbol") or "UNKNOWN"
        px = st.get("mark")
        if not px:
            try:
                df = load_klines(bus, sym, "1d")
                px = float(df["close"].iloc[-1])
            except Exception:
                px = st.get("entry_px")
        m = _close_math(st, px) if px else {"pnl_gross": 0, "fees_exit": 0,
                                            "funding": 0, "pnl_net": 0}
        m["equity_after"] = st["equity"]
        _log_trade({"ts": now, "spec_id": sid, "action": "RETIRED",
                    "symbol": sym, "px": px, "meta": True, **m})
        actions.append({"spec_id": sid, "action": "RETIRED", "px": px,
                        "pnl_usd": m["pnl_net"]})
        st.update(in_pos=0, entry_px=None, entry_ts=None, coins=None,
                  unrealized=0.0, mark=px)
        state[sid] = st

    _save_state(state)

    # ---- book-level mark series (one row per tick) ----
    book_eq = round(sum(s.get("equity", 0) for s in state.values()
                        if isinstance(s, dict)), 2)
    book_unrl = round(sum(s.get("unrealized", 0) for s in state.values()
                          if isinstance(s, dict)), 2)
    try:
        rows = EQUITY_P.read_text().splitlines()[-4999:] if EQUITY_P.exists() else []
        rows.append(json.dumps({"ts": now, "book_equity": book_eq,
                                "book_unrealized": book_unrl}))
        EQUITY_P.write_text("\n".join(rows) + "\n")
    except Exception:
        pass

    return {"actions": actions,
            "book_equity": book_eq, "book_unrealized": book_unrl,
            "positions": {k: v for k, v in state.items()
                          if isinstance(v, dict) and v.get("in_pos")}}
