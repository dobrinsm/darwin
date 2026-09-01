"""Dashboard — single-file HTML status page for the foundry + tiny text mode.

`python -m darwin.dashboard`            rebuild data/dashboard.html
`python -m darwin.dashboard --text`     one-glance terminal summary
`python -m darwin.dashboard --serve`    serve on http://127.0.0.1:8787 (fresh
                                        render per request; never bind 0.0.0.0
                                        without adding auth first)

The orchestrator rebuilds the page every tick (Phase E). Pure stdlib, no JS
deps, dark theme, readable on a phone.
"""
from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
SPECS = ROOT / "specs"

VERDICT_CLASS = {"PROMOTE": "promote", "MUTATE": "mutate", "KILL": "kill"}


# ---------------------------------------------------------------- data
def _read(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _age_min(ts: float) -> float:
    return max(0.0, (time.time() - ts) / 60.0)


def snapshot(root: Path = ROOT) -> dict:
    """Everything the dashboard shows, as plain data (testable)."""
    root = Path(root)
    data, runs, specs = root / "data", root / "runs", root / "specs"

    reports = []
    for p in sorted((runs).glob("*/report.json")):
        rep = _read(p, None)
        if rep and (specs / f"{p.parent.name}.json").exists():
            rep["live"] = True
            reports.append(rep)
    verdicts = {v: sum(1 for r in reports if r["verdict"] == v)
                for v in ("PROMOTE", "MUTATE", "KILL")}

    opt = []
    for p in sorted(runs.glob("*/optimizer.json")):
        o = _read(p, None)
        if o:
            opt.append(o)

    state = _read(data / "arena_state.json", {})
    trades = []
    tp = data / "arena_trades.jsonl"
    if tp.exists():
        try:
            rows = [json.loads(l) for l in tp.read_text().splitlines() if l.strip()]
        except Exception:
            rows = []
        trades = rows[-40:][::-1]

    coll_latest, coll_age = {}, None
    cs = _read(data / "collector_status.json", {})
    if cs:
        last_ts = list(cs)[-1]
        coll_latest = cs[last_ts]
        try:
            coll_age = _age_min(time.mktime(time.strptime(last_ts, "%Y-%m-%dT%H:%M")))
        except Exception:
            coll_age = None

    theses = []
    ml = data / "miner_log.jsonl"
    if ml.exists():
        try:
            rows = [json.loads(l) for l in ml.read_text().splitlines() if l.strip()]
            for row in rows[-3:]:
                for s in row.get("saved", []):
                    theses.append({"ts": row.get("ts"), "name": s.get("name"),
                                   "thesis": s.get("thesis", "")})
        except Exception:
            pass
    theses = theses[-6:][::-1]

    tick_age = _age_min((data / "collector_status.json").stat().st_mtime) \
        if (data / "collector_status.json").exists() else None
    gauntlet_age = _age_min((data / "last_gauntlet.txt").stat().st_mtime) \
        if (data / "last_gauntlet.txt").exists() else None

    # single source of truth for what the arena actually trades
    from .arena import promoted_specs
    live_ids = {s["spec_id"] for s in promoted_specs()}
    bus_events = None
    db = data / "events.db"
    if db.exists():
        import sqlite3
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            bus_events = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            con.close()
        except Exception:
            pass

    return {"reports": reports, "verdicts": verdicts, "optimizer": opt,
            "state": state, "trades": trades, "collectors": coll_latest,
            "collector_age": coll_age, "theses": theses, "tick_age": tick_age,
            "gauntlet_age": gauntlet_age, "bus_events": bus_events,
            "live_ids": live_ids}


# ---------------------------------------------------------------- render
def _esc(s) -> str:
    return html.escape(str(s))


def _badge(v: str) -> str:
    return f'<span class="badge {VERDICT_CLASS.get(v, "")}">{_esc(v)}</span>'


def _window_bars(rep: dict) -> str:
    bars = []
    for w in (rep.get("walk_forward") or []):
        s = float(w.get("oos", {}).get("sharpe") or 0)
        h = min(abs(s), 3.0) / 3.0 * 16 + 2
        cls = "pos" if s > 0 else ("neg" if s < 0 else "zero")
        title = f"{w.get('oos_start','?')} → {w.get('oos_end','?')}: sharpe {s:+.2f}"
        bars.append(f'<div class="bar {cls}" style="height:{h:.0f}px" title="{_esc(title)}"></div>')
    return f'<div class="bars">{"".join(bars)}</div>'


def _fmt(x, nd=2, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def render(snap: dict) -> str:
    v = snap["verdicts"]
    total = sum(v.values())

    tick = snap["tick_age"]
    if tick is None:
        health, hcls = "no heartbeat yet", "warn"
    elif tick < 20:
        health, hcls = f"heartbeat {tick:.0f}m ago", "ok"
    elif tick < 90:
        health, hcls = f"heartbeat {tick:.0f}m ago", "warn"
    else:
        health, hcls = f"heartbeat {tick:.0f}m ago", "bad"

    state = snap["state"]
    live_ids = snap["live_ids"]
    n_pos = sum(1 for s in state.values() if s.get("in_pos"))
    equity = sum(float(s.get("equity") or 0) for s in state.values()) or None

    rows = []
    ordered = sorted(snap["reports"],
                     key=lambda r: (r["spec_id"] not in live_ids,
                                    r["verdict"] != "PROMOTE",
                                    -r.get("avg_oos_sharpe", 0)))
    for r in ordered:
        fs = r.get("full_sample") or {}
        live = "★" if r["spec_id"] in live_ids else ""
        rows.append(f"""
<tr>
 <td class="mono">{live} {_esc(r['spec_id'])}</td>
 <td>{_esc(r['name'])}</td>
 <td>{_badge(r['verdict'])}</td>
 <td class="num">{_fmt(r.get('avg_oos_sharpe'), sign=True)}</td>
 <td class="num">{_fmt(r.get('oos_loo_sharpe'), sign=True)}</td>
 <td class="num">{_fmt(r.get('oos_compound_pct'), 1, True)}%</td>
 <td class="num">{_esc(r.get('winning_windows', '—'))}</td>
 <td class="num">{_fmt(fs.get('funding_pct'), 1, True)}%</td>
 <td class="num">{_fmt(fs.get('fees_pct'), 1, True)}%</td>
 <td class="num">{_fmt(fs.get('max_dd'), 1)}%</td>
 <td>{_window_bars(r)}</td>
</tr>""")

    arena_rows = []
    for sid, st in state.items():
        if not isinstance(st, dict):
            continue
        if st.get("in_pos"):
            pos = (f"LONG {_esc(st.get('entry_px'))}"
                   f" → <span class='mono'>{_esc(st.get('mark'))}</span>"
                   f" <span class='{'ok' if (st.get('unrealized') or 0) >= 0 else 'bad'}'>"
                   f"{(st.get('unrealized') or 0):+,.0f}$</span>")
        else:
            pos = "flat"
        fund = st.get("funding_paid") or 0
        arena_rows.append(f"""
<tr>
 <td class="mono">{_esc(sid)}</td>
 <td>{pos}</td>
 <td class="num">{_fmt(st.get('equity'), 2)}</td>
 <td class="num">{_fmt(fund, 2, True)}</td>
</tr>""")

    trade_rows = "".join(f"""
<tr>
 <td class="mono">{time.strftime('%m-%d %H:%M', time.gmtime(t.get('ts', 0)))}</td>
 <td class="mono">{_esc(t.get('spec_id', ''))}</td>
 <td>{_esc(t.get('action', ''))}</td>
 <td>{_esc(t.get('symbol') or '')}</td>
 <td class="num">{_esc(t.get('px', ''))}</td>
 <td class="num">{_fmt(t.get('pnl_usd'), 2, True) if 'pnl_usd' in t else ''}</td>
</tr>""" for t in snap["trades"])

    opt_rows = "".join(f"""
<tr>
 <td class="mono">{_esc(o.get('seed_id', ''))}</td>
 <td class="mono">{_esc(o.get('child') or '—')}</td>
 <td class="num">{_fmt(o.get('seed_fitness'), 2, True)}</td>
 <td class="num">{_fmt(o.get('delta_fitness'), 2, True)}</td>
 <td class="num">{_fmt(o.get('child_avg_oos_sharpe'), 2, True)}</td>
 <td class="num">{_fmt(o.get('child_oos_loo_sharpe'), 2, True)}</td>
 <td class="num">{_esc(o.get('candidates_tested', 0))}</td>
 <td class="num">{_esc(o.get('secs', '—'))}s</td>
</tr>""" for o in snap["optimizer"])

    coll_rows = "".join(f"""
<tr>
 <td>{_esc(name)}</td>
 <td>{'<span class="ok">ok</span>' if c.get('ok') else f'<span class="bad">{_esc(c.get("detail", "ERR"))[:40]}</span>'}</td>
 <td class="num">{_esc(c.get('new', 0))}</td>
 <td class="num">{_esc(c.get('secs', '—'))}s</td>
</tr>""" for name, c in snap["collectors"].items())

    thesis_rows = "".join(f"""
<div class="thesis"><b>{_esc(t['name'])}</b><br><span>{_esc((t.get('thesis') or '')[:220])}</span></div>"""
                          for t in snap["theses"])

    gage = snap["gauntlet_age"]
    gage_txt = f"{gage:.0f}m ago" if gage is not None else "never"

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>DARWIN foundry</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ background:#0d1117; color:#c9d1d9; font:14px/1.45 -apple-system,'Segoe UI',Roboto,sans-serif;
        margin:0; padding:16px; }}
 h1 {{ font-size:20px; margin:0 0 2px; }} h2 {{ font-size:15px; margin:22px 0 8px; color:#8b949e; }}
 .sub {{ color:#8b949e; font-size:12px; margin-bottom:14px; }}
 .cards {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:6px; }}
 .card {{ background:#161b22; border:1px solid #21262d; border-radius:8px; padding:10px 14px; min-width:120px; }}
 .card .k {{ color:#8b949e; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
 .card .v {{ font-size:20px; font-weight:600; }}
 table {{ border-collapse:collapse; width:100%; font-size:13px; }}
 th,td {{ text-align:left; padding:5px 8px; border-bottom:1px solid #21262d; vertical-align:middle; }}
 th {{ color:#8b949e; font-weight:500; font-size:11px; text-transform:uppercase; }}
 tr:hover td {{ background:#161b22; }}
 td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .mono {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; }}
 .badge {{ padding:1px 8px; border-radius:10px; font-size:11px; font-weight:600; }}
 .badge.promote {{ background:#0d4429; color:#3fb950; }}
 .badge.mutate {{ background:#4a3a12; color:#d29922; }}
 .badge.kill {{ background:#3d1618; color:#f85149; }}
 .ok {{ color:#3fb950; }} .warn {{ color:#d29922; }} .bad {{ color:#f85149; }}
 .bars {{ display:flex; align-items:flex-end; gap:2px; height:20px; }}
 .bar {{ width:6px; border-radius:1px; }}
 .bar.pos {{ background:#3fb950; }} .bar.neg {{ background:#f85149; }} .bar.zero {{ background:#30363d; }}
 .thesis {{ background:#161b22; border:1px solid #21262d; border-radius:8px; padding:8px 12px; margin:6px 0; font-size:13px; }}
 .thesis span {{ color:#8b949e; }}
 .scroll {{ overflow-x:auto; }}
</style></head><body>
<h1>DARWIN foundry</h1>
<div class="sub">generated {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC ·
<span class="{hcls}">{health}</span> · last gauntlet {gage_txt} · auto-refresh 60s</div>

<div class="cards">
 <div class="card"><div class="k">bus events</div><div class="v">{_esc(snap['bus_events'] if snap['bus_events'] is not None else '—')}</div></div>
 <div class="card"><div class="k">population</div><div class="v">{total}</div></div>
 <div class="card"><div class="k">promoted</div><div class="v ok">{v['PROMOTE']}</div></div>
 <div class="card"><div class="k">mutating</div><div class="v" style="color:#d29922">{v['MUTATE']}</div></div>
 <div class="card"><div class="k">killed</div><div class="v" style="color:#f85149">{v['KILL']}</div></div>
 <div class="card"><div class="k">arena positions</div><div class="v">{n_pos}</div></div>
 <div class="card"><div class="k">arena equity</div><div class="v">{_fmt(equity, 0)}</div></div>
</div>

<h2>Feeds</h2>
<div class="scroll"><table>
<tr><th>collector</th><th>status</th><th>new events</th><th>secs</th></tr>
{coll_rows}
</table></div>

<h2>Population — walk-forward verdicts</h2>
<div class="scroll"><table>
<tr><th>spec</th><th>name</th><th>verdict</th><th>oos sharpe</th><th>loo</th><th>oos comp</th>
<th>windows</th><th>funding</th><th>fees</th><th>maxDD</th><th>oos windows</th></tr>
{''.join(rows)}
</table></div>

<h2>Arena book ({len(state)} specs tracked)</h2>
<div class="scroll"><table>
<tr><th>spec</th><th>position</th><th>equity</th><th>funding paid</th></tr>
{''.join(arena_rows)}
</table></div>

<h2>Recent trades</h2>
<div class="scroll"><table>
<tr><th>time (UTC)</th><th>spec</th><th>action</th><th>symbol</th><th>px</th><th>pnl $</th></tr>
{trade_rows}
</table></div>

<h2>MUTATE optimizer runs</h2>
<div class="scroll"><table>
<tr><th>seed</th><th>child</th><th>seed fit</th><th>Δ fit</th><th>child oos</th><th>child loo</th><th>tested</th><th>secs</th></tr>
{opt_rows}
</table></div>

<h2>Latest miner theses</h2>
{thesis_rows or '<div class="sub">none yet</div>'}

<div class="sub" style="margin-top:20px">research tool, not investment advice</div>
</body></html>"""


def build(root: Path = ROOT, out: str | None = None) -> Path:
    snap = snapshot(root)
    dst = Path(out) if out else Path(root) / "data" / "dashboard.html"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(render(snap))
    return dst


# ---------------------------------------------------------------- text mode
def text_report(root: Path = ROOT) -> str:
    snap = snapshot(root)
    v = snap["verdicts"]
    lines = [f"DARWIN status — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}"]
    tick = snap["tick_age"]
    lines.append(f"heartbeat: {'%.0f min ago' % tick if tick is not None else 'NEVER'}"
                 f" | bus: {snap['bus_events']} events"
                 f" | gauntlet: {'%.0f min ago' % snap['gauntlet_age'] if snap['gauntlet_age'] is not None else 'never'}")
    lines.append(f"population: {v['PROMOTE']} promote / {v['MUTATE']} mutate / {v['KILL']} kill")

    coll = snap["collectors"]
    if coll:
        dead = [k for k, c in coll.items() if not c.get("ok")]
        ok = [k for k, c in coll.items() if c.get("ok")]
        lines.append("feeds: ok=" + ",".join(ok) + (f" DOWN={','.join(dead)}" if dead else ""))

    lines.append("")
    state = snap["state"]
    live_ids = snap["live_ids"]
    pos = {k: s for k, s in state.items() if isinstance(s, dict) and s.get("in_pos")}
    book = sum(s.get("equity", 0) for s in state.values() if isinstance(s, dict))
    unrl = sum(s.get("unrealized", 0) for s in state.values() if isinstance(s, dict))
    lines.append(f"arena book: {book:,.2f} USDT ({book - 10_000 * len(state):+,.2f} vs start)"
                 f" | unrealized {unrl:+,.2f} | {len(pos)} open, {len(live_ids)} live specs")
    for r in sorted(snap["reports"], key=lambda r: r["spec_id"] not in live_ids):
        if r["verdict"] != "PROMOTE" or r["spec_id"] not in live_ids:
            continue
        st = state.get(r["spec_id"], {})
        if st.get("in_pos"):
            ptxt = (f"LONG {st.get('entry_px')}→{st.get('mark')}"
                    f" unreal {st.get('unrealized', 0):+,.0f}$"
                    f" fund {st.get('funding_trade', 0):+,.2f}")
        else:
            ptxt = "flat"
        lines.append(f"  ★ {r['spec_id']} {r['name'][:36]:38}"
                     f" oos={r['avg_oos_sharpe']:+.2f} loo={r.get('oos_loo_sharpe', 0):+.2f}"
                     f" eq={st.get('equity', 0):,.0f}  [{ptxt}]")
    for t in snap["trades"][:6]:
        extra = f" pnl {t['pnl_usd']:+.2f}" if "pnl_usd" in t else ""
        lines.append(f"  {time.strftime('%m-%d %H:%M', time.gmtime(t.get('ts', 0)))}"
                     f" {t.get('action',''):<8} {t.get('spec_id',''):<16}{extra}")
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--text" in args:
        print(text_report())
    elif "--serve" in args:
        host = "127.0.0.1"
        port = int(args[args.index("--port") + 1]) if "--port" in args else 8787
        import os
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        TOKEN = os.environ.get("DASH_TOKEN", "").strip()   # required when set

        class H(BaseHTTPRequestHandler):
            def _authorized(self) -> bool:
                if not TOKEN:
                    return True
                q = self.path.split("?", 1)
                if len(q) == 2 and f"token={TOKEN}" in q[1]:
                    return True
                return self.headers.get("Authorization", "") == f"Bearer {TOKEN}"

            def do_GET(self):
                if not self._authorized():
                    body = b"<h1>401</h1>"
                    self.send_response(401)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                try:
                    body = build().read_bytes()
                except Exception as e:
                    body = f"<pre>dashboard error: {html.escape(str(e))}</pre>".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):  # quiet
                pass

        print(f"darwin dashboard on http://{host}:{port}"
              f" ({'token-protected' if TOKEN else 'NO AUTH — localhost only'})", flush=True)
        ThreadingHTTPServer((host, port), H).serve_forever()
    else:
        p = build()
        print(p)
