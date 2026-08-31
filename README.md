# DARWIN

An **evolutionary strategy foundry** for crypto trading. Not a trading bot — a factory that
invents its own strategies, kills almost all of them in a leakage-proof gauntlet, and
paper-trades only the survivors.

```
LOOP 0  INGEST      heterogeneous feeds → point-in-time SQLite event bus
LOOP 1  MINE        LLM cross-reads the bus → proposes strategy SPECS (pure JSON)
LOOP 2  GAUNTLET    walk-forward backtest of every spec → PROMOTE / MUTATE / KILL
LOOP 3  ARENA       promoted specs paper-trade in parallel on a virtual book
YOU                 read the daily digest; the machine does the rest
```

No hand-coded strategies anywhere. The LLM never writes code — it composes falsifiable
hypotheses from a fixed node vocabulary, and a deterministic engine decides their fate.
The default outcome of a bad idea is death in Loop 2.

## Results from the first 24h (20-spec population)

| Verdict | Count | Example |
|---|---|---|
| PROMOTE | 1 | SOL momentum breakout: OOS Sharpe +0.86, compound +220%, 6/10 winning windows |
| MUTATE | 5 | queued for parameter search |
| KILL | 14 | incl. every spec whose data feed wasn't on the bus — untestable = dead |

The champion sat flat through the 2022 bear market and 2026 chop: 29 trades in 6 years,
19% exposure, profitable in 6/10 walk-forward windows, lost only one.

## The four invariants

Everything is built on these (stolen from the best of TraderHarness, Planar.jl and
DepthSight — the tools that inspired this — but none of them is a dependency):

1. **Point-in-time truth.** Every event carries `ts` (source-claimed event time) and
   `ingested_at`. Backtests filter on `ts <= as_of`. A spec can only see what existed.
2. **No lookahead, mechanically.** Signals computed on bar closes act on the *next* bar.
   Cross-timeframe nodes forward-fill only bars closed before the decision bar.
3. **Determinism.** Same inputs → same backtest. Specs are checksummed pure JSON.
4. **One order path.** The arena is the only writer of positions, fully logged.

## Architecture

```
darwin/
  bus.py          append-only SQLite event bus (dedup by source+dedup_key)
  collectors.py   6 feed collectors: agentservices, binance, LSE, finlight,
                  tradestie/WSB, alphasmo (via jina relay past Cloudflare)
  backfill.py     history: binance 1d/4h to 2020, fear&greed to 2019, LSE cross-asset
  spec_schema.py  the SPEC contract: JSON-schema'd node DAG, strict param ranges
  engine.py       spec compiler → vectorized point-in-time backtest
  gauntlet.py     walk-forward judge: 12mo IS / 6mo OOS rolling windows
  miner.py        LLM mining cycle: bus snapshot → proposals → schema-validated specs
  arena.py        virtual paper book, position reconciliation, trade log
orchestrator.py   the heartbeat (designed for systemd timer, every 15 min)
digest.py         daily Telegram-ready summary
```

## Node vocabulary (what the Miner can compose)

`price_above_sma` · `ema_cross_up/down` · `rsi_below/above` · `vol_spike` ·
`drawdown_from_high` · `runup_from_low` · `fear_greed_below/above` · `wsb_rank_above` ·
`news_sentiment_below/above` · `convergence` (smart money) · `cross_asset_score`
(equities/FX/gold momentum vote) · `iv_skew_above` (reserved)

A spec example lives in `spec_schema.demo_spec()`.

## Setup

```bash
pip install pandas numpy requests jsonschema python-dotenv

# API keys (all have free tiers) in .env or environment:
#   LSE_API_KEY        https://londonstrategicedge.com  (market data vault)
#   FINLIGHT_API_KEY   https://finlight.me              (news)
#   ALPHASMO_API_KEY   https://alphasmo.com/developer   (13F/insider convergence)
#   OPENROUTER_API_KEY https://openrouter.ai            (the Miner)

python darwin/backfill.py     # one-off history load
python run_collectors.py      # test the feeds
python -m darwin.miner        # first mining cycle
python -m darwin.gauntlet     # judge the population
python orchestrator.py        # one full heartbeat
```

### Production (systemd)

```bash
sudo cp deploy/darwin.service deploy/darwin.timer deploy/darwin-digest.timer /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now darwin.timer darwin-digest.timer
```

## Data sources

| Source | What it provides | Auth |
|---|---|---|
| [AgentServices](https://agentservices.to) | fear&greed, trending, prices (free endpoints) | none |
| [Binance](https://binance.com) | klines, the execution venue | public |
| [London Strategic Edge](https://londonstrategicedge.com/data/) | 133B ticks: equities, FX, commodities, options, macro | free key |
| [Finlight](https://finlight.me) | real-time news (sentiment scored locally via lexicon) | free key |
| [Tradestie](https://tradestie.com/apps/reddit/api/) | WallStreetBets top-50 sentiment | none |
| [AlphaSMO](https://alphasmo.com) | 13F + Form 4 smart-money convergence | free key |

## Honest limits

- **One champion is not proof.** +0.86 OOS Sharpe on a 6-year daily backtest is promising,
  not significant. Real capital only after clean paper-trading months.
- The Miner is only as honest as its prompts — the gauntlet is the actual safety net.
- Tradestie's cert is chronically flaky; the collector degrades gracefully.
- Fees modeled flat at 5 bps taker; no slippage or funding-rate modeling yet.
- Long-only, crypto-only (v1). `convergence`/news nodes need richer history before
  walk-forward-meaningful — the gauntlet correctly kills specs that depend on them today.

## Status

v0.1 — first fully autonomous loop closed 2026-08-31. Population grows daily at 04:10 UTC.

*Research project, not investment advice. Crypto trading can lose you everything.*

MIT © 2026
