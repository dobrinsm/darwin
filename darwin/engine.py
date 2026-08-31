"""Gauntlet engine — compiles a SPEC into a deterministic point-in-time backtest.

Point-in-time rules (the invariants):
1. Signals are computed on bar closes; the position acts on the NEXT bar
   (position.shift(1)) — no lookahead, ever.
2. Event nodes (sentiment, wsb, fear-greed, convergence) read the bus with
   as_of = bar close time. The bus only ever returns events with ts <= as_of.
3. Cross-timeframe nodes use bars whose close_ts <= the decision bar's close_ts.
4. Same inputs -> same outputs. No I/O during the run after materialization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bus import EventBus
from .spec_schema import NODE_TYPES

FEE = 0.0005  # Binance taker per side, on notional

TS_COLUMNS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------- materialize
def load_klines(bus: EventBus, symbol: str, interval: str,
                start: str | None = None, end: str | None = None,
                source: str = "binance") -> pd.DataFrame:
    """Bars from the bus indexed by close time (UTC). A bar's facts are only
    'known' at its close_ts."""
    evs = bus.read(event_type="kline", source=source,
                   symbol=symbol.replace("USDT", ""), limit=10_000_000)
    rows = []
    for e in evs:
        p = e.payload
        if p.get("interval") != interval:
            continue
        rows.append((e.ts, p["open"], p["high"], p["low"], p["close"], p["volume"]))
    df = pd.DataFrame(rows, columns=["close_ts", "open", "high", "low", "close", "volume"])
    if df.empty:
        raise RuntimeError(f"no klines on bus for {symbol} {interval} — run backfill")
    df = df.drop_duplicates("close_ts").set_index("close_ts").sort_index()
    df.index = pd.to_datetime(df.index, unit="s", utc=True)
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df[TS_COLUMNS]


def _fear_greed_series(bus: EventBus) -> pd.Series:
    evs = bus.read(event_type="fear_greed", limit=10_000_000)
    if not evs:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([e.ts for e in evs], unit="s", utc=True)
    vals = [float(e.payload.get("value", e.payload.get("index", np.nan))) for e in evs]
    s = pd.Series(vals, index=idx).dropna()
    return s[~s.index.duplicated()].sort_index()


def _wsb_rank_series(bus: EventBus, ticker: str) -> pd.Series:
    evs = bus.read(event_type="wsb_sentiment", symbol=ticker, limit=1_000_000)
    if not evs:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([e.ts for e in evs], unit="s", utc=True)
    s = pd.Series([float(e.payload["rank"]) for e in evs], index=idx)
    return s[~s.index.duplicated()].sort_index()


def _news_sentiment_series(bus: EventBus, ticker: str) -> pd.DataFrame:
    """score in [-1, 1] per article, confidence-weighted downstream."""
    evs = bus.read(event_type="news", symbol=ticker, limit=1_000_000)
    if not evs:
        return pd.DataFrame(columns=["score", "conf"])
    smap = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    idx = pd.to_datetime([e.ts for e in evs], unit="s", utc=True)
    df = pd.DataFrame({
        "score": [smap.get((e.payload.get("sentiment") or "").lower(), 0.0) for e in evs],
        "conf": [float(e.payload.get("confidence") or 0.5) for e in evs],
    }, index=idx)
    return df.sort_index()


def _convergence_series(bus: EventBus) -> pd.DataFrame:
    evs = bus.read(event_type="convergence", limit=1_000_000)
    if not evs:
        return pd.DataFrame(columns=["ticker", "confidence"])
    idx = pd.to_datetime([e.ts for e in evs], unit="s", utc=True)
    df = pd.DataFrame({
        "ticker": [e.payload.get("ticker") for e in evs],
        "confidence": [float(e.payload.get("confidence") or 0) for e in evs],
    }, index=idx)
    return df.sort_index()


# ---------------------------------------------------------------- indicators
def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _crossed_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossed_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


# ---------------------------------------------------------------- node eval
class Context:
    """Everything a node evaluator may need, materialized once per run."""

    def __init__(self, bus: EventBus, symbol: str, base_tf: str, index: pd.DatetimeIndex):
        self.bus = bus
        self.symbol = symbol
        self.base_tf = base_tf
        self.index = index
        self._tf_cache: dict[str, pd.DataFrame] = {}
        self.fg = _fear_greed_series(bus)
        self.wsb = _wsb_rank_series(bus, symbol.replace("USDT", ""))
        self.news = _news_sentiment_series(bus, symbol.replace("USDT", ""))
        self.conv = _convergence_series(bus)
        # track which data sources were actually present in-range (for UNTESTABLE)
        self.sources_present = {
            "fear_greed": len(self.fg) > 0,
            "wsb": len(self.wsb) > 0,
            "news": len(self.news) > 0,
            "convergence": len(self.conv) > 0,
        }

    def tf_df(self, tf: str) -> pd.DataFrame:
        if tf == self.base_tf:
            return None  # caller already has it
        if tf not in self._tf_cache:
            self._tf_cache[tf] = load_klines(self.bus, self.symbol, tf)
        return self._tf_cache[tf]


def _node_to_bool(node: dict, df: pd.DataFrame, ctx: Context) -> pd.Series:
    """Evaluate one node -> boolean Series on df.index. Point-in-time safe."""
    t = node["type"]
    idx = df.index
    c = df["close"]

    def _reindex(s: pd.Series) -> pd.Series:
        return s.reindex(idx, method="ffill")

    if t == "price_above_sma":
        return c > _sma(c, node["period"])
    if t == "price_below_sma":
        return c < _sma(c, node["period"])
    if t == "ema_cross_up":
        if node["tf"] == ctx.base_tf:
            return _crossed_up(_ema(c, node["fast"]), _ema(c, node["slow"]))
        o = ctx.tf_df(node["tf"])
        sig = _crossed_up(_ema(o["close"], node["fast"]), _ema(o["close"], node["slow"]))
        return _reindex(sig.astype(float)).fillna(0.0) > 0
    if t == "ema_cross_down":
        if node["tf"] == ctx.base_tf:
            return _crossed_down(_ema(c, node["fast"]), _ema(c, node["slow"]))
        o = ctx.tf_df(node["tf"])
        sig = _crossed_down(_ema(o["close"], node["fast"]), _ema(o["close"], node["slow"]))
        return _reindex(sig.astype(float)).fillna(0.0) > 0
    if t == "rsi_below":
        return _rsi(c, node["period"]) < node["threshold"]
    if t == "rsi_above":
        return _rsi(c, node["period"]) > node["threshold"]
    if t == "vol_spike":
        vavg = df["volume"].rolling(node["lookback"]).mean().shift(1)
        return df["volume"] > node["mult"] * vavg
    if t == "drawdown_from_high":
        roll_hi = c.rolling(node["lookback_d"]).max()
        return (c / roll_hi - 1) < -node["pct"]
    if t == "runup_from_low":
        roll_lo = c.rolling(node["lookback_d"]).min()
        return (c / roll_lo - 1) > node["pct"]
    if t in ("fear_greed_below", "fear_greed_above"):
        if ctx.fg.empty:
            return pd.Series(False, index=idx)
        known = ctx.fg.reindex(idx, method="ffill")
        return (known < node["threshold"]) if t == "fear_greed_below" else (known > node["threshold"])
    if t == "wsb_rank_above":
        if ctx.wsb.empty:
            return pd.Series(False, index=idx)
        known = ctx.wsb.reindex(idx, method="ffill").fillna(999)
        return known <= node["rank"]
    if t in ("news_sentiment_below", "news_sentiment_above"):
        if ctx.news.empty:
            return pd.Series(False, index=idx)
        n = ctx.news
        # expanding-as-of mean sentiment over trailing window, conf-weighted
        w = n["score"] * n["conf"].clip(lower=0.3)
        min_conf = node.get("min_conf", 0.5)
        n_f = n[n["conf"] >= min_conf]
        w = (n_f["score"] * n_f["conf"]).groupby(pd.Grouper(freq="h")).sum()
        cnt = n_f.groupby(pd.Grouper(freq="h")).size()
        hourly = (w / cnt.replace(0, np.nan)).fillna(0.0)
        known = hourly.rolling(int(node["window_h"]), min_periods=1).mean()
        known = known.reindex(idx, method="ffill").fillna(0.0)
        return (known < node["threshold"]) if t == "news_sentiment_below" else (known > node["threshold"])
    if t in ("convergence", "insider_buy"):
        if ctx.conv.empty:
            return pd.Series(False, index=idx)
        tick = ctx.symbol.replace("USDT", "")
        hit = ctx.conv[ctx.conv["ticker"] == tick]
        if hit.empty:
            return pd.Series(False, index=idx)
        minc = node.get("min_confidence", 60)
        strong = hit[hit["confidence"] >= minc]
        present = strong.reindex(idx, method="ffill").notna()
        if t == "convergence":
            win = pd.Timedelta(days=node.get("window_d", 30))
            fresh = pd.Series(False, index=idx)
            ts_known = strong.index
            pos = ts_known.searchsorted(idx, side="right") - 1
            ok = pos >= 0
            ts_last = pd.Series(ts_known[np.clip(pos, 0, None)], index=idx)[ok]
            fresh[ok] = (idx[ok] - ts_last) <= win
            return present & fresh
        return present
    if t == "iv_skew_above":
        return pd.Series(False, index=idx)  # LSE options tape — future activation
    if t == "cross_asset_score":
        # momentum vote across cross-asset series on the bus (LSE daily bars)
        # assets are bus symbol tags: SPY, QQQ, EUR, XAU ...
        score = pd.Series(0.0, index=idx)
        used = 0
        mom_days = max(int(node.get("mom_h", 96)) // 24, 1)
        for asset in node.get("assets", []):
            try:
                o = load_klines(ctx.bus, asset, "1d", source="lse")
            except Exception:
                continue
            mom = o["close"].pct_change(mom_days)
            up = (mom.reindex(idx, method="ffill").fillna(0.0) > 0).astype(float)
            score = score + up
            used += 1
        if used == 0:
            return pd.Series(False, index=idx)
        ctx.sources_present["cross_asset"] = True
        return score >= node["min_score"]
    raise ValueError(f"unhandled node type {t}")


def compile_signal(spec: dict, df: pd.DataFrame, ctx: Context) -> tuple[pd.Series, list[str]]:
    """Return boolean entry/exit Series + list of data gaps found."""
    gaps = []
    for section in ("entry", "exit"):
        sec = spec.get(section) or {}
        for node in list(sec.get("all") or []) + list(sec.get("any") or []):
            ntype = node["type"]
            if ntype in ("iv_skew_above", "cross_asset_score"):
                gaps.append(f"{section}:{ntype}:needs_lse")
            if ntype in ("convergence", "insider_buy") and not ctx.sources_present["convergence"]:
                gaps.append(f"{section}:{ntype}:needs_alphasmo")
            if ntype.startswith("news_") and not ctx.sources_present["news"]:
                gaps.append(f"{section}:{ntype}:needs_finlight")
            if ntype.startswith("wsb_") and not ctx.sources_present["wsb"]:
                gaps.append(f"{section}:{ntype}:needs_wsb")

    def combine(sec, default):
        all_nodes = sec.get("all") or []
        any_nodes = sec.get("any") or []
        if not all_nodes and not any_nodes:
            return pd.Series(default, index=df.index)
        out = pd.Series(True, index=df.index)
        for n in all_nodes:
            out &= _node_to_bool(n, df, ctx)
        if any_nodes:
            anyv = pd.Series(False, index=df.index)
            for n in any_nodes:
                anyv |= _node_to_bool(n, df, ctx)
            out &= anyv
        return out

    return combine(spec["entry"], False), combine(spec["exit"], False), gaps


# ---------------------------------------------------------------- simulation
def simulate(spec: dict, df: pd.DataFrame, entry_sig: pd.Series,
             exit_sig: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Long-only position state machine. Deterministic, no lookahead:
    signal known at close of bar t -> position held over bar t+1's return."""
    lev = spec["risk"]["leverage"]
    max_frac = spec["risk"]["max_pos_frac"]
    cooldown = spec["risk"].get("cooldown_bars", 0)
    trail = (spec.get("exit", {}).get("stops") or {}).get("trail_pct")
    hard = (spec.get("exit", {}).get("stops") or {}).get("hard_pct")

    pos = np.zeros(len(df))
    state = 0
    bars_since_exit = 10**9
    entry_px = np.nan
    peak = np.nan
    rets = df["close"].pct_change().fillna(0.0).to_numpy()
    closes = df["close"].to_numpy()

    for i in range(len(df)):
        if state == 0:
            bars_since_exit += 1
            if entry_sig.iloc[i] and bars_since_exit > cooldown:
                state = 1
                entry_px = closes[i]
                peak = closes[i]
        else:
            peak = max(peak, closes[i])
            exit_now = bool(exit_sig.iloc[i])
            if trail is not None and closes[i] < peak * (1 - trail):
                exit_now = True
            if hard is not None and closes[i] < entry_px * (1 - hard):
                exit_now = True
            if exit_now:
                state = 0
                bars_since_exit = 0
        pos[i] = state

    pos_s = pd.Series(pos, index=df.index)
    # act on next bar's return; notional = pos*lev capped at 0.9 equity
    notional = (pos_s.shift(1).fillna(0.0) * lev).clip(upper=0.9)
    gross = notional * rets
    fees = notional.diff().abs().fillna(notional.iloc[0] * (notional.iloc[0] > 0)) * FEE
    net = gross - fees
    trades_mask = pos_s.diff().abs() > 0
    return net, pd.DataFrame({
        "position": pos_s, "notional": notional, "ret": rets, "net": net,
        "trade": trades_mask,
    })


def metrics(net: pd.Series, pos: pd.Series, tf: str) -> dict:
    periods = {"1d": 365, "4h": 365 * 6}[tf]
    clean = net.dropna()
    if clean.std() == 0 and clean.abs().sum() == 0:
        return {"total_ret": 0.0, "pa_ret": 0.0, "vol": 0.0, "sharpe": 0.0,
                "max_dd": 0.0, "calmar": 0.0, "trades": 0, "win_rate": 0.0,
                "exposure": 0.0}
    total = (1 + clean).prod() - 1
    n = len(clean)
    pa = (1 + total) ** (periods / max(n, 1)) - 1 if total > -1 else -1.0
    vol = clean.std() * np.sqrt(periods)
    sharpe = pa / vol if vol > 0 else 0.0
    cum = (1 + clean).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    calmar = pa / abs(dd) if dd < 0 else 0.0
    # trades & win rate on position episodes
    p = pos.astype(int)
    entries = ((p == 1) & (p.shift(1, fill_value=0) == 0))
    n_trades = int(entries.sum())
    wins = 0
    if n_trades:
        seg_id = entries.cumsum()
        for _, seg in clean.groupby(seg_id.reindex(clean.index)):
            if seg.sum() > 0:
                wins += 1
    win_rate = wins / n_trades if n_trades else 0.0
    return {
        "total_ret": round(float(total) * 100, 2),
        "pa_ret": round(float(pa) * 100, 2),
        "vol": round(float(vol) * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "max_dd": round(float(dd) * 100, 2),
        "calmar": round(float(calmar), 3),
        "trades": n_trades,
        "win_rate": round(win_rate * 100, 1),
        "exposure": round(float((p > 0).mean()) * 100, 1),
    }
