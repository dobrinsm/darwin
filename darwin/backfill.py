"""Historical ingest -> bus. Everything (klines, fear-greed, WSB) becomes events
with honest `ts` so the engine reads one store and can never see the future."""
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone

from .bus import EventBus, Event
from .collectors import BINANCE_SYMBOLS

HIST_START = "2020-01-01"


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def backfill_binance(bus: EventBus, symbols=None, intervals=("1d", "4h"),
                     start: str = HIST_START) -> int:
    """All closed klines from `start` to now. 1000 bars/call, paged."""
    symbols = symbols or BINANCE_SYMBOLS
    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    total = 0
    for sym in symbols:
        for tf in intervals:
            end_ms = start_ms
            while True:
                url = (f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={tf}"
                       f"&startTime={end_ms}&limit=1000")
                rows = _get_json(url)
                if not rows:
                    break
                events = []
                for k in rows:
                    open_ts = k[0] / 1000.0
                    close_ts = k[6] / 1000.0
                    if close_ts > time.time():
                        continue
                    events.append(Event(
                        "binance", "kline", close_ts,
                        {"symbol": sym, "interval": tf,
                         "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                         "close": float(k[4]), "volume": float(k[5]),
                         "open_ts": open_ts, "close_ts": close_ts},
                        f"kl:{sym}:{tf}:{int(open_ts)}",
                        symbols=[sym.replace("USDT", "")]))
                total += bus.publish(events)
                last_open = rows[-1][0]
                if len(rows) < 1000:
                    break
                end_ms = last_open + 1
    return total


def backfill_fear_greed(bus: EventBus) -> int:
    """alternative.me Fear&Greed index — daily history to 2019-02 (free, no key).
    Same index agentservices serves live; used only for backtest history."""
    data = _get_json("https://api.alternative.me/fng/?limit=0&format=json")
    events = []
    for row in data.get("data", []):
        ts = int(row["timestamp"])
        if ts > time.time():
            continue
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        events.append(Event(
            "feargreed", "fear_greed", ts,
            {"value": int(row["value"]), "label": row["value_classification"]},
            f"fg:{day}", symbols=["BTC", "ETH", "SOL", "XLM", "DOGE"]))
    return bus.publish(events)


def backfill_wsb(bus: EventBus, days: int = 45) -> int:
    """Tradestie allows ?date=MM-DD-YYYY. Availability varies; best effort."""
    total = 0
    for d in range(days, -1, -1):
        day = datetime.now(timezone.utc) - timedelta(days=d)
        ds = day.strftime("%m-%d-%Y")
        iso = day.strftime("%Y-%m-%d")
        try:
            rows = _get_json(f"https://api.tradestie.com/v1/apps/reddit?date={ds}")
        except Exception:
            continue
        events = []
        for i, r in enumerate(rows if isinstance(rows, list) else []):
            if not r.get("ticker"):
                continue
            events.append(Event(
                "tradestie", "wsb_sentiment",
                datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp(),
                {"rank": i + 1, "ticker": r.get("ticker"),
                 "comments": r.get("no_of_comments"), "sentiment": r.get("sentiment"),
                 "sentiment_score": r.get("sentiment_score")},
                f"wsb:{iso}:{r.get('ticker')}", symbols=[r.get("ticker")]))
        total += bus.publish(events)
    return total


def backfill_lse(bus: EventBus, start: str = "2023-01-01") -> int:
    """Cross-asset daily bars from LSE (equities/FX/commodities) — the inputs
    for cross_asset_score nodes. Requires LSE_API_KEY."""
    key = os.environ.get("LSE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("no_key")
    from lse import LSE
    client = LSE(api_key=key)
    symbols = ["SPY", "QQQ", "EUR/USD", "XAU/USD"]
    total = 0
    for sym in symbols:
        try:
            rows = client.candles(sym, "1d", start=start) or []
        except Exception:
            continue
        events = []
        for r in rows:
            ts = r.get("timestamp") or r.get("ts")
            if ts is None:
                continue
            try:
                t = datetime.fromisoformat(str(ts).replace("Z", "+0000"))
            except ValueError:
                continue
            if t.timestamp() > time.time():
                continue
            events.append(Event(
                "lse", "kline", t.timestamp(),
                {"symbol": sym, "interval": "1d",
                 "open": float(r["open"]), "high": float(r["high"]),
                 "low": float(r["low"]), "close": float(r["close"]),
                 "volume": float(r.get("volume") or 0)},
                f"lse:{sym}:1d:{int(t.timestamp())}",
                symbols=[sym.split("/")[0]]))
        total += bus.publish(events)
    return total


def backfill_funding(bus: EventBus, symbols=None) -> int:
    """USDT-M perps funding rate history from listing (free fapi, no key).
    Paged 1000 records/call backwards is not supported by the endpoint, so we
    page forward with startTime. Idempotent via UNIQUE(source, dedup_key)."""
    symbols = symbols or BINANCE_SYMBOLS
    start_ms = int(datetime.fromisoformat(HIST_START)
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    total = 0
    for sym in symbols:
        end_ms = start_ms
        while True:
            url = (f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}"
                   f"&startTime={end_ms}&limit=1000")
            rows = None
            for attempt in range(4):   # 429/5xx: back off and retry
                try:
                    rows = _get_json(url)
                    break
                except Exception:
                    time.sleep(2.0 * (attempt + 1))
            if not isinstance(rows, list) or not rows:
                break  # error dict, exhausted retries, or exhausted history
            events = []
            for r in rows:
                fts = float(r["fundingTime"]) / 1000.0
                if fts > time.time():
                    continue
                try:
                    rate = float(r["fundingRate"])
                except (TypeError, ValueError):
                    continue
                events.append(Event(
                    "binance", "funding", fts,
                    {"symbol": sym, "interval": "8h", "rate": rate,
                     "mark_price": r.get("markPrice") or None},
                    f"funding:{sym}:{int(fts * 1000)}",
                    symbols=[sym.replace("USDT", "")]))
            total += bus.publish(events)
            if len(rows) < 1000:
                break
            end_ms = int(float(rows[-1]["fundingTime"])) + 1
            time.sleep(0.35)   # fapi weight budget: 1 request ≈ 1 weight
    return total


def run_all(bus: EventBus | None = None) -> dict:
    bus = bus or EventBus()
    out = {}
    t0 = time.time()
    out["binance_klines"] = backfill_binance(bus)
    out["funding"] = backfill_funding(bus)
    out["fear_greed"] = backfill_fear_greed(bus)
    out["wsb"] = backfill_wsb(bus)
    try:
        out["lse_cross_asset"] = backfill_lse(bus)
    except Exception as e:
        out["lse_cross_asset"] = f"skipped: {e}"
    out["secs"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from dotenv import load_dotenv
    load_dotenv("/root/.hermes/.env", override=False)
    from darwin.backfill import run_all
    print(json.dumps(run_all(), indent=1))
