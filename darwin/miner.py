"""Loop 1 — the Miner.

An LLM reads the bus state (what the collectors saw) and proposes NEW strategy
specs as strict JSON. The schema is the contract: invalid proposals are
rejected before they ever touch the gauntlet. The Miner never sees or writes
code — it composes falsifiable hypotheses from a fixed node vocabulary.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from .bus import EventBus
from .engine import load_klines
from .spec_schema import NODE_TYPES, SPEC_DIR, validate_spec, save_spec

MODEL = os.environ.get("DARWIN_MINER_MODEL", "google/gemini-2.5-flash")
MAX_PROPOSALS = 3

NODE_CHEATSHEET = """
- price_above_sma {period 2-400, tf} / price_below_sma
- ema_cross_up {fast 2-400, slow 3-400, fast<slow, tf} / ema_cross_down
- rsi_below {period 2-400, threshold -5..105, tf} / rsi_above
- vol_spike {mult 1-20, lookback 3-500, tf}
- drawdown_from_high {pct 0.01-0.95, lookback_d 3-500} / runup_from_low
- funding_above {threshold -0.01..0.01} / funding_below
    threshold is the realized 8h perp funding fraction (0.0001 = 0.01%).
    Positive funding means longs pay shorts; funding_below is a long-entry
    crowding/carry veto and funding_above can force an exit from crowded longs.
- fear_greed_below {threshold 0-100} / fear_greed_above
- wsb_rank_above {rank 1-50}
- news_sentiment_below {threshold -1..1, window_h 1-720, min_conf 0-1} / news_sentiment_above
    NOTE: news events are ticker-tagged articles; crypto symbols currently get
    few tags, so news nodes on XLM/DOGE/SOL/BTC/ETH specs mostly never fire.
    For crypto sentiment use fear_greed_below/above instead.
- convergence {min_confidence 0-100, window_d 1-120} / insider_buy {window_d}  [US stocks]
- iv_skew_above {percentile 0-1, lookback_d} (options — NOT yet live, omit)
- cross_asset_score {min_score 0-5, assets ["SPY","QQQ","EUR","XAU"], mom_h 24-336}
    counts how many of those assets have positive momentum; entry requires
    score >= min_score. Equity/FX/gold data is live on the bus.
TA-node tf: "1d" or "4h" and MAY differ from the asset tf. Example: a 4h
entry can require price_above_sma {period 200, tf "1d"}; only completed 1d bars
are visible, making this a point-in-time higher-timeframe regime gate.
Exit safety field (not a node): exit.max_hold_bars is an integer 1-500 and caps
position age in asset-timeframe bars. Symbols: XLMUSDT, DOGEUSDT, SOLUSDT,
BTCUSDT, ETHUSDT.
"""


def bus_snapshot(bus: EventBus) -> str:
    """Compact, honest summary of what's on the bus right now."""
    lines = []
    fg = bus.read(event_type="fear_greed", limit=60)
    if fg:
        vals = [(e.ts, e.payload.get("value", e.payload.get("index"))) for e in reversed(fg)]
        vals = [(t, float(v)) for t, v in vals if v is not None]
        if vals:
            lines.append(f"FEAR-GREED: latest={vals[-1][1]:.0f}, "
                         f"30d-ago={vals[max(0,-30)][1]:.0f} (n={len(vals)})")
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XLMUSDT", "DOGEUSDT"):
        try:
            df = load_klines(bus, sym, "1d")
            c = df["close"]
            r30 = c.iloc[-1] / c.iloc[-31] - 1
            r90 = c.iloc[-1] / c.iloc[-91] - 1
            vol = c.pct_change().std() * (365 ** 0.5) * 100
            ath_dd = c.iloc[-1] / c.rolling(365).max().iloc[-1] - 1
            lines.append(f"{sym}: 30d={r30*100:+.1f}% 90d={r90*100:+.1f}% "
                         f"ann.vol={vol:.0f}% dd-from-365d-high={ath_dd*100:+.1f}%")
            funding = bus.read(event_type="funding", source="binance",
                               symbol=sym.replace("USDT", ""), limit=90)
            if funding:
                rates = [float(e.payload.get("rate") or 0.0) for e in funding]
                latest = rates[0]
                avg = sum(rates) / len(rates)
                lines.append(f"{sym} FUNDING: latest-8h={latest*100:+.4f}% "
                             f"recent-avg={avg*100:+.4f}% "
                             "(positive=longs pay)")
        except Exception:
            continue
    wsb = bus.read(event_type="wsb_sentiment", limit=10)
    if wsb:
        top = ", ".join(f"{e.payload['ticker']}(#{e.payload['rank']},"
                        f"{e.payload.get('sentiment','?')})" for e in wsb[:8])
        lines.append(f"WSB TOP: {top}")
    news = bus.read(event_type="news", limit=20)
    if news:
        sents = [e.payload.get("sentiment") for e in news]
        lines.append(f"NEWS: {len(news)} recent, sentiment mix: "
                     f"pos={sents.count('positive')} neg={sents.count('negative')} "
                     f"neu={sents.count('neutral')}")
    conv = bus.read(event_type="convergence", limit=10)
    if conv:
        lines.append("SMART-MONEY CONVERGENCE: " +
                     ", ".join(f"{e.payload['ticker']}({e.payload.get('confidence')})"
                               for e in conv[:8]))
    if len(lines) < 3:
        lines.append("(bus mostly empty — propose from price/momentum facts only)")
    return "\n".join(lines)


def existing_summary() -> str:
    out = []
    for p in SPEC_DIR.glob("*.json"):
        s = json.loads(p.read_text())
        nodes = [n["type"] for n in (s["entry"].get("all") or []) + (s["entry"].get("any") or [])]
        out.append(f"  - '{s['name']}' on {s['asset']['symbol']}: entry={nodes}")
    return "\n".join(out) or "  (none yet)"


# ---------------------------------------------------------------- death report
def death_report(root: Path | None = None) -> str:
    """What the foundry has learned the hard way, for the miner's prompt.

    Three evidence channels, all from on-disk state:
    - arena outcomes: recent closed trades (net PnL + hold time) and currently
      open positions underwater — what is failing LIVE, not in backtest
    - frozen specs: demoted for paper-equity breach — named and shamed
    - gauntlet KILLs: which theses are structurally dead and which OOS windows
      killed them (the regime fingerprint of the failure)
    Compact by construction: the prompt budget is finite.
    """
    root = root or Path(__file__).resolve().parent.parent
    data, runs, specs = root / "data", root / "runs", root / "specs"
    import time as _t
    now = _t.time()
    lines = []

    def _name(sid: str) -> str:
        p = specs / f"{sid}.json"
        if p.exists():
            try:
                return json.loads(p.read_text()).get("name", sid)
            except Exception:
                return sid
        return sid

    def _spec_sym(sid: str) -> str | None:
        """Symbol from the spec file — arena state predating the cost model
        never stamped `symbol` on the position, hence the '?' fallbacks."""
        p = specs / f"{sid}.json"
        if p.exists():
            try:
                return json.loads(p.read_text()).get("asset", {}).get("symbol")
            except Exception:
                return None
        return None

    # --- live wounds: closed trades last 14d + open positions
    trades_p = data / "arena_trades.jsonl"
    closed = []
    if trades_p.exists():
        for l in trades_p.read_text().splitlines():
            if not l.strip():
                continue
            t = json.loads(l)
            if t.get("action") == "EXIT" and now - t.get("ts", 0) < 14 * 86400:
                closed.append(t)
    if closed:
        lines.append("RECENT LIVE EXITS (real paper money, last 14d):")
        for t in closed[-8:]:
            sym = t.get("symbol") or _spec_sym(t.get("spec_id", "")) or "?"
            lines.append(f"  - {_name(t.get('spec_id','?'))} on {sym}: "
                         f"net {t.get('pnl_net', 0):+.0f}$ "
                         f"(gross {t.get('pnl_gross', 0):+.0f}, "
                         f"fees {t.get('fees_exit', 0):.0f}, "
                         f"funding {t.get('funding', 0):+.1f})")

    state_p = data / "arena_state.json"
    if state_p.exists():
        try:
            st = json.loads(state_p.read_text())
        except Exception:
            st = {}
        losers = [(sid, s) for sid, s in st.items()
                  if isinstance(s, dict) and s.get("in_pos")
                  and (s.get("unrealized") or 0) < -50]
        if losers:
            lines.append("OPEN POSITIONS DEEPLY UNDERWATER (held right now):")
            for sid, s in sorted(losers, key=lambda x: x[1].get("unrealized", 0))[:5]:
                sym = s.get("symbol") or _spec_sym(sid) or "?"
                lines.append(f"  - {_name(sid)} on {sym}: "
                             f"entry {s.get('entry_px')} -> mark {s.get('mark')}, "
                             f"unrealized {s.get('unrealized', 0):+.0f}$")

    # --- frozen: demoted by the arena for losing real paper money
    frozen_p = data / "frozen_specs.json"
    if frozen_p.exists():
        try:
            fr = json.loads(frozen_p.read_text())
        except Exception:
            fr = {}
        if fr:
            lines.append("FROZEN SPECS (demoted for live losses — do NOT propose "
                         "close variants of these):")
            for sid, meta in list(fr.items())[:6]:
                lines.append(f"  - {_name(sid)}: {meta.get('reason','?')} "
                             f"(equity {meta.get('equity', 0):,.0f})")

    # --- structural deaths: KILL verdicts + the windows that killed them
    kills = []
    for p in sorted(runs.glob("*/report.json")):
        rep = json.loads(p.read_text())
        sid = p.parent.name
        if not (specs / f"{sid}.json").exists():
            continue
        if rep.get("verdict") == "KILL":
            wins = [round(w.get("oos", {}).get("sharpe", 0), 2)
                    for w in (rep.get("walk_forward") or [])]
            kills.append((rep.get("ran_at", 0), sid, rep.get("name", sid),
                          rep.get("avg_oos_sharpe", 0), wins))
    if kills:
        kills.sort(key=lambda k: -k[0])
        lines.append("STRUCTURALLY DEAD THESES (KILL verdicts, with OOS window "
                     "sharpes — note WHICH regimes killed them):")
        for _, sid, name, avg, wins in kills[:8]:
            lines.append(f"  - {name} [{sid[:13]}]: avg {avg:+.2f}, "
                         f"windows {wins}")

    if not lines:
        return "(no failure history yet — first generation)"
    return "\n".join(lines)


PROMPT = """You are the Miner in an autonomous trading-strategy foundry.
Below is market state from a point-in-time event bus, the node vocabulary for
strategy specs, and the specs that already exist.

Propose up to @@MAXPROPOSALS@@ NEW long-only crypto strategies as JSON specs.
Rules:
- Each spec: {"name", "asset":{"class":"crypto","symbol","tf"}, "direction":"long",
  "entry":{"all":[...],"any":[...]}, "exit":{"any":[...],"max_hold_bars":1-500,
  "stops":{"trail_pct"|"hard_pct"}},
  "risk":{"leverage" 1-3, "max_pos_frac" 0.05-0.5, "cooldown_bars" 0-10}, "confidence" 0-1,
  "provenance":{"thesis":"one sentence WHY this edge exists"}}
- Long-only. Trends get entered on confirmation, fades get entered on extremes.
- Be NOVEL vs existing specs (different asset, timeframe, or mechanism).
- exit.any should contain the mirror of the entry mechanism; add max_hold_bars
  when the thesis should expire rather than bleed indefinitely.
- Keep entries compact: normally one trigger plus at most one independent
  regime/crowding veto. Extra AND conditions often never fire.
- A TA node may use a different tf from the asset. For 4h entries, prefer a
  completed 1d trend node when the thesis requires broad-regime agreement.
- Think about WHY each edge could exist (behavioral, flow, structure) and put it in provenance.thesis.
- STUDY THE FAILURE REPORT below before proposing. Do not resubmit a thesis
  family that already died the same way (same mechanism + same asset + similar
  params). A death in a falling-market window with long-only entries is a hint
  about regime sensitivity, not just bad luck — say in your thesis why YOUR
  variant survives the regime that killed its predecessors.
- Respond with ONLY a JSON array of specs, no markdown, no commentary.

CRITICAL NODE FORMAT — every node is an object with a "type" KEY plus flat params:
    {"type": "rsi_below", "period": 14, "threshold": 30, "tf": "1d"}
WRONG (will be rejected):  {"rsi_below": {"period": 14}}
FULL EXAMPLE SPEC:
[
  {"name": "example 4h pullback in daily uptrend",
   "asset": {"class": "crypto", "symbol": "BTCUSDT", "tf": "4h"},
   "direction": "long",
   "entry": {"all": [
       {"type": "rsi_below", "period": 14, "threshold": 35, "tf": "4h"},
       {"type": "price_above_sma", "period": 200, "tf": "1d"}]},
   "exit": {"any": [
       {"type": "rsi_above", "period": 14, "threshold": 65, "tf": "4h"}],
       "max_hold_bars": 18, "stops": {"hard_pct": 0.1}},
   "risk": {"leverage": 2, "max_pos_frac": 0.25, "cooldown_bars": 2},
   "confidence": 0.5,
   "provenance": {"thesis": "4h pullbacks expire quickly and only trade with the closed daily regime"}}
]

NODE VOCABULARY (params in {}):
@@CHEATSHEET@@

EXISTING SPECS (do not duplicate):
@@EXISTING@@

FAILURE REPORT — what the foundry has already tried and lost on:
@@DEATHS@@

CURRENT MARKET STATE (point-in-time, honest):
@@SNAPSHOT@@
"""


def _call_llm(prompt: str) -> str:
    key = os.environ["OPENROUTER_API_KEY"]
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 3000,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]


def _parse_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.index("["):]
    start, depth = text.find("["), 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("no JSON array in miner response")


def _duplicate(spec: dict) -> bool:
    """Reject same symbol + same entry mechanism set."""
    key = (spec["asset"]["symbol"],
           tuple(sorted(n["type"] for n in (spec["entry"].get("all") or []) +
                        (spec["entry"].get("any") or []))))
    for p in SPEC_DIR.glob("*.json"):
        s = json.loads(p.read_text())
        k2 = (s["asset"]["symbol"],
              tuple(sorted(n["type"] for n in (s["entry"].get("all") or []) +
                           (s["entry"].get("any") or []))))
        if key == k2:
            return True
    return False


def mine(bus: EventBus | None = None) -> dict:
    """One mining cycle: snapshot -> LLM -> validate -> save. Returns summary."""
    bus = bus or EventBus()
    snapshot = bus_snapshot(bus)
    deaths = death_report()
    prompt = (PROMPT.replace("@@MAXPROPOSALS@@", str(MAX_PROPOSALS))
              .replace("@@CHEATSHEET@@", NODE_CHEATSHEET)
              .replace("@@EXISTING@@", existing_summary())
              .replace("@@DEATHS@@", deaths)
              .replace("@@SNAPSHOT@@", snapshot))
    raw = _call_llm(prompt)
    proposals = _parse_array(raw)
    saved, rejected = [], []
    for pr in proposals:
        errs = validate_spec(pr)
        if errs:
            rejected.append({"name": pr.get("name", "?"), "errors": errs[:3]})
            continue
        if _duplicate(pr):
            rejected.append({"name": pr.get("name", "?"), "errors": ["duplicate of existing spec"]})
            continue
        p = save_spec(pr)
        saved.append({"spec_id": pr["spec_id"], "name": pr["name"],
                      "thesis": (pr.get("provenance") or {}).get("thesis", "")})
    log_path = Path(__file__).resolve().parent.parent / "data" / "miner_log.jsonl"
    with log_path.open("a") as f:
        import time
        f.write(json.dumps({"ts": time.time(), "model": MODEL,
                            "saved": saved, "rejected": rejected}) + "\n")
    return {"saved": saved, "rejected": rejected, "snapshot": snapshot}


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from dotenv import load_dotenv
    load_dotenv("/root/.hermes/.env", override=False)
    out = mine()
    print(f"MINER CYCLE — {len(out['saved'])} saved, {len(out['rejected'])} rejected")
    for s in out["saved"]:
        print(f"  + {s['spec_id']}  {s['name']}")
        if s["thesis"]:
            print(f"      thesis: {s['thesis'][:110]}")
    for r in out["rejected"]:
        print(f"  x {r['name']}: {r['errors']}")
