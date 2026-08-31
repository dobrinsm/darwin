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
tf: "1d" or "4h". Symbols: XLMUSDT, DOGEUSDT, SOLUSDT, BTCUSDT, ETHUSDT.
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


PROMPT = """You are the Miner in an autonomous trading-strategy foundry.
Below is market state from a point-in-time event bus, the node vocabulary for
strategy specs, and the specs that already exist.

Propose up to @@MAXPROPOSALS@@ NEW long-only crypto strategies as JSON specs.
Rules:
- Each spec: {"name", "asset":{"class":"crypto","symbol","tf"}, "direction":"long",
  "entry":{"all":[...],"any":[...]}, "exit":{"any":[...],"stops":{"trail_pct"|"hard_pct"}},
  "risk":{"leverage" 1-3, "max_pos_frac" 0.05-0.5, "cooldown_bars" 0-10}, "confidence" 0-1,
  "provenance":{"thesis":"one sentence WHY this edge exists"}}
- Long-only. Trends get entered on confirmation, fades get entered on extremes.
- Be NOVEL vs existing specs (different asset, timeframe, or mechanism).
- exit.any should contain the mirror of the entry mechanism.
- Think about WHY each edge could exist (behavioral, flow, structure) and put it in provenance.thesis.
- Respond with ONLY a JSON array of specs, no markdown, no commentary.

CRITICAL NODE FORMAT — every node is an object with a "type" KEY plus flat params:
    {"type": "rsi_below", "period": 14, "threshold": 30, "tf": "1d"}
WRONG (will be rejected):  {"rsi_below": {"period": 14}}
FULL EXAMPLE SPEC:
[
  {"name": "example trend with froth veto",
   "asset": {"class": "crypto", "symbol": "DOGEUSDT", "tf": "1d"},
   "direction": "long",
   "entry": {"all": [
       {"type": "ema_cross_up", "fast": 20, "slow": 100, "tf": "1d"},
       {"type": "fear_greed_below", "threshold": 80}]},
   "exit": {"any": [
       {"type": "ema_cross_down", "fast": 20, "slow": 100, "tf": "1d"}],
       "stops": {"trail_pct": 0.25}},
   "risk": {"leverage": 3, "max_pos_frac": 0.3, "cooldown_bars": 2},
   "confidence": 0.5,
   "provenance": {"thesis": "trend entries avoided when retail euphoria peaks"}}
]

NODE VOCABULARY (params in {}):
@@CHEATSHEET@@

EXISTING SPECS (do not duplicate):
@@EXISTING@@

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
    prompt = (PROMPT.replace("@@MAXPROPOSALS@@", str(MAX_PROPOSALS))
              .replace("@@CHEATSHEET@@", NODE_CHEATSHEET)
              .replace("@@EXISTING@@", existing_summary())
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
