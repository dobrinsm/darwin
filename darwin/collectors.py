"""Loop 0 collectors — one collect(bus) function per source.

Design rules:
- Never raise: runners wrap each collector; failures are recorded as status.
- No key? Skip cleanly (status="no_key") so the loop self-activates when keys land.
- Every event gets a stable dedup_key -> re-running is always idempotent.
"""
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from .bus import EventBus, Event

HTTP_TIMEOUT = 15
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Darwin/0.1"}


def _get(url: str, headers: dict | None = None, insecure: bool = False,
         timeout: int = HTTP_TIMEOUT):
    """GET returning parsed JSON. insecure=True only for hosts with broken certs."""
    hdrs = dict(UA)
    if headers:
        hdrs.update(headers)
    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _get_text(url: str, headers: dict | None = None,
              timeout: int = HTTP_TIMEOUT) -> str:
    hdrs = dict(UA)
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _env(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------- agentservices
def collect_agentservices(bus: EventBus) -> int:
    """Free endpoints: fear-greed, trending, prices for tracked crypto."""
    events = []
    fg = _get("https://agentservices.to/v1/fear-greed")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    events.append(Event("agentservices", "fear_greed", _now(),
                        fg, f"fg:{day}", symbols=["BTC", "ETH", "SOL", "XLM", "DOGE"]))

    tr = _get("https://agentservices.to/v1/trending")
    toks = [{"symbol": t.get("symbol"), "name": t.get("name"), "rank": t.get("market_cap_rank")}
            for t in tr.get("trending", [])[:10]]
    hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    events.append(Event("agentservices", "trending", _now(),
                        {"tokens": toks}, f"trending:{hour}"))

    syms = "BTC,ETH,SOL,XLM,DOGE"
    px = _get(f"https://agentservices.to/v1/prices?symbols={syms}")
    ts = px.get("timestamp", _now())
    for sym, d in (px.get("prices") or {}).items():
        events.append(Event(
            "agentservices", "price_snapshot", float(ts),
            {"symbol": sym, "price_usd": d.get("price_usd"),
             "change_24h_pct": d.get("change_24h_pct"),
             "volume_24h_usd": d.get("volume_24h_usd"),
             "market_cap_usd": d.get("market_cap_usd")},
            f"px:{sym}:{int(ts)//900}",  # 15-min bucket
            symbols=[sym]))
    return bus.publish(events)


# ---------------------------------------------------------------- binance klines
BINANCE_INTERVALS = {"4h": 4 * 3600, "1d": 86400}
BINANCE_SYMBOLS = ["XLMUSDT", "DOGEUSDT", "SOLUSDT", "BTCUSDT", "ETHUSDT"]


def collect_binance(bus: EventBus, lookback_bars: int = 5) -> int:
    """Recent klines (public, no key). Collectors keep the tail fresh;
    historical backfill lives in gauntlet/backfill.py."""
    events = []
    for sym in BINANCE_SYMBOLS:
        for tf, secs in BINANCE_INTERVALS.items():
            url = (f"https://api.binance.com/api/v3/klines?symbol={sym}"
                   f"&interval={tf}&limit={lookback_bars}")
            rows = _get(url)
            for k in rows:
                open_ts = k[0] / 1000.0
                close_ts = k[6] / 1000.0
                if close_ts > _now():     # candle not closed yet -> skip
                    continue
                events.append(Event(
                    "binance", "kline", close_ts,
                    {"symbol": sym, "interval": tf,
                     "open": float(k[1]), "high": float(k[2]),
                     "low": float(k[3]), "close": float(k[4]),
                     "volume": float(k[5]), "open_ts": open_ts, "close_ts": close_ts},
                    f"kl:{sym}:{tf}:{int(open_ts)}",
                    symbols=[sym.replace("USDT", "")]))
    return bus.publish(events)


# ---------------------------------------------------------------- LSE
def collect_lse(bus: EventBus, symbols: list[str] | None = None,
                timeframes: tuple[str, ...] = ("1d",)) -> int:
    """LSE vault candles + options IV snapshot. Requires LSE_API_KEY."""
    key = _env("LSE_API_KEY")
    if not key:
        raise RuntimeError("no_key")
    from lse import LSE
    client = LSE(api_key=key)
    symbols = symbols or ["BTC/USD", "ETH/USD", "SOL/USD", "XLM/USD", "DOGE/USD",
                          "SPY", "QQQ", "EUR/USD", "XAU/USD"]
    events = []
    for sym in symbols:
        for tf in timeframes:
            try:
                rows = client.candles(sym, tf,
                                      start=(datetime.now(timezone.utc)
                                             - timedelta(days=10)).strftime("%Y-%m-%d"))
            except Exception:  # symbol may not have that tf
                continue
            for r in rows or []:
                ts = r.get("timestamp") or r.get("ts")
                if ts is None:
                    continue
                t = datetime.fromisoformat(str(ts).replace("Z", "+0000"))
                events.append(Event(
                    "lse", "kline", t.timestamp(),
                    {"symbol": sym, "interval": tf,
                     "open": float(r["open"]), "high": float(r["high"]),
                     "low": float(r["low"]), "close": float(r["close"]),
                     "volume": float(r.get("volume") or 0)},
                    f"lse:{sym}:{tf}:{int(t.timestamp())}",
                    symbols=[sym.split("/")[0]]))
    return bus.publish(events)


def _lexicon_sentiment(text: str) -> tuple[str, float]:
    """Tiny deterministic finance lexicon. Returns (sentiment, confidence).
    Used when the API doesn't provide its own sentiment field."""
    t = (text or "").lower()
    pos = ["surge", "soar", "rally", "jump", "beat", "record", "upgrade", "gain",
           "bullish", "strong", "inflow", "adoption", "approval", "breakout",
           "profit", "growth", "boost", "win", "green", "recovery", "halving"]
    neg = ["crash", "plunge", "sink", "slump", "miss", "downgrade", "loss",
           "bearish", "weak", "outflow", "ban", "hack", "fraud", "lawsuit",
           "probe", "selloff", "dump", "liquidation", "fear", "delist", "collapse"]
    p = sum(t.count(w) for w in pos)
    n = sum(t.count(w) for w in neg)
    if p == n:
        return "neutral", 0.5
    if p > n:
        return "positive", min(0.6 + 0.1 * (p - n), 0.95)
    return "negative", min(0.6 + 0.1 * (n - p), 0.95)


# ---------------------------------------------------------------- Finlight
def _post_json(url: str, body: dict, headers: dict | None = None):
    hdrs = dict(UA)
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={**hdrs, "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def collect_finlight(bus: EventBus, query: str = "crypto OR bitcoin OR fed OR earnings") -> int:
    """Latest tagged/scored news. v2 API is POST. Requires FINLIGHT_API_KEY."""
    key = _env("FINLIGHT_API_KEY")
    if not key:
        raise RuntimeError("no_key")
    data = _post_json(
        "https://api.finlight.me/v2/articles",
        {"query": query, "pageSize": 25, "language": "en", "includeEntities": True},
        headers={"X-API-Key": key})
    events = []
    for a in data.get("articles", []):
        link = a.get("link") or ""
        if not link:
            continue
        pub = a.get("publishDate")
        t = datetime.fromisoformat(str(pub).replace("Z", "+0000")).timestamp() if pub else _now()
        tickers = [c.get("ticker") for c in (a.get("companies") or []) if c.get("ticker")]
        sentiment = a.get("sentiment")
        confidence = a.get("confidence")
        if sentiment is None:  # v2 API omits sentiment -> lexicon fallback
            sentiment, confidence = _lexicon_sentiment(
                f"{a.get('title', '')} {a.get('summary', '')}")
        events.append(Event(
            "finlight", "news", t,
            {"title": a.get("title"), "link": link, "source": a.get("source"),
             "summary": a.get("summary"), "sentiment": sentiment,
             "confidence": confidence, "tickers": tickers},
            "news:" + hashlib.sha1(link.encode()).hexdigest()[:16],
            symbols=tickers))
    return bus.publish(events)


# ---------------------------------------------------------------- Tradestie WSB
def collect_wsb(bus: EventBus) -> int:
    """Top-50 WSB tickers w/ sentiment. Their SSL cert is flaky -> insecure fallback."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        rows = _get("https://api.tradestie.com/v1/apps/reddit")
    except (ssl.SSLError, urllib.error.URLError, OSError):
        rows = _get("https://api.tradestie.com/v1/apps/reddit", insecure=True)
    events = [Event(
        "tradestie", "wsb_sentiment", _now(),
        {"rank": i + 1, "ticker": r.get("ticker"),
         "comments": r.get("no_of_comments"), "sentiment": r.get("sentiment"),
         "sentiment_score": r.get("sentiment_score")},
        f"wsb:{day}:{r.get('ticker')}", symbols=[r.get("ticker")])
        for i, r in enumerate(rows if isinstance(rows, list) else []) if r.get("ticker")]
    return bus.publish(events)


# ---------------------------------------------------------------- AlphaSMO
def collect_alphasmo(bus: EventBus) -> int:
    """Smart-money convergence. alphasmo.com Cloudflare-blocks this server's IP,
    so requests go through the r.jina.ai reader relay (server-side fetch).
    Requires ALPHASMO_API_KEY."""
    key = _env("ALPHASMO_API_KEY")
    if not key:
        raise RuntimeError("no_key")
    target = ("https://alphasmo.com/api/v1/insider/smart-money-convergence"
              "?limit=10&min_confidence=60")
    relay = f"https://r.jina.ai/{target}"
    raw = _get_text(relay, headers={"x-api-key": key}, timeout=45)
    # jina wraps content as: "Title:\n\nURL Source: ...\n\nMarkdown Content:\n<json>"
    marker = "Markdown Content:"
    body = raw[raw.index(marker) + len(marker):].strip() if marker in raw else raw.strip()
    data = json.loads(body)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    events = [Event(
        "alphasmo", "convergence", _now(),
        {"ticker": d.get("ticker"),
         "confidence": d.get("confidence_score") or d.get("confidence"),
         "net_flow_usd": d.get("guru_net_flow") or d.get("net_flow_usd"),
         "unique_buyers": d.get("unique_buyers"),
         "detail": d},
        f"conv:{day}:{d.get('ticker')}", symbols=[d.get("ticker")])
        for d in (data if isinstance(data, list) else data.get("data", []))
        if d.get("ticker")]
    return bus.publish(events)


# ---------------------------------------------------------------- runner
ALL = {
    "agentservices": collect_agentservices,
    "binance": collect_binance,
    "wsb": collect_wsb,
    "lse": collect_lse,
    "finlight": collect_finlight,
    "alphasmo": collect_alphasmo,
}

STATUS_PATH = None  # set by run_all


def run_all(bus: EventBus | None = None, only: list[str] | None = None) -> dict:
    """Run every collector, record status. Returns {name: {ok, new, detail}}."""
    global STATUS_PATH
    from pathlib import Path
    bus = bus or EventBus()
    STATUS_PATH = Path(__file__).resolve().parent.parent / "data" / "collector_status.json"
    results = {}
    for name, fn in ALL.items():
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            new = fn(bus)
            results[name] = {"ok": True, "new": new, "detail": "", "secs": round(time.time() - t0, 2)}
        except Exception as e:
            results[name] = {"ok": False, "new": 0,
                             "detail": f"{type(e).__name__}: {e}"[:200],
                             "secs": round(time.time() - t0, 2)}
    try:
        import json as _json
        prev = {}
        if STATUS_PATH.exists():
            prev = _json.loads(STATUS_PATH.read_text())
        prev[datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")] = results
        # keep last 48 runs
        STATUS_PATH.write_text(_json.dumps(dict(list(prev.items())[-48:]), indent=1))
    except Exception:
        pass
    return results
