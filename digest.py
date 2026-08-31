"""Daily digest for Telegram. Empty output = nothing interesting = silent."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
SPECS = ROOT / "specs"


def main():
    lines = []

    # population
    counts = {"PROMOTE": 0, "MUTATE": 0, "KILL": 0}
    champion, ch_rep = None, None
    for p in RUNS.glob("*/report.json"):
        rep = json.loads(p.read_text())
        v = rep.get("verdict")
        if v in counts:
            counts[v] += 1
        if v == "PROMOTE" and (ch_rep is None or
                               rep["avg_oos_sharpe"] > ch_rep["avg_oos_sharpe"]):
            champion, ch_rep = rep["spec_id"], rep
    total = sum(counts.values())
    if total:
        lines.append(f"*DARWIN daily* — population {total}: "
                     f"{counts['PROMOTE']} promoted / {counts['MUTATE']} mutating / "
                     f"{counts['KILL']} killed")

    # champion + arena position
    state_p = DATA / "arena_state.json"
    if champion and ch_rep:
        fs = ch_rep["full_sample"]
        pos_txt = "flat"
        if state_p.exists():
            st = json.loads(state_p.read_text()).get(champion)
            if st and st.get("in_pos"):
                pos_txt = f"LONG {ch_rep['symbol']} @ {st['entry_px']}"
        lines.append(
            f"Champion: {ch_rep['name']} (`{champion}`)\n"
            f"  OOS sharpe {ch_rep['avg_oos_sharpe']:+.2f} | "
            f"OOS compound {ch_rep['oos_compound_pct']:+.1f}% | "
            f"windows {ch_rep['winning_windows']}\n"
            f"  full-sample {fs['total_ret']:+.0f}% | maxDD {fs['max_dd']:.0f}% | "
            f"{fs['trades']} trades | {fs['exposure']:.0f}% exposed\n"
            f"  paper position: {pos_txt}")

    # trades in last 24h
    trades_p = DATA / "arena_trades.jsonl"
    if trades_p.exists():
        day_ago = time.time() - 86400
        recent = [json.loads(l) for l in trades_p.read_text().splitlines()
                  if l.strip() and json.loads(l).get("ts", 0) > day_ago]
        for t in recent:
            px = t.get("px", "?")
            extra = f" pnl {t['pnl_usd']:+.2f}" if "pnl_usd" in t else ""
            lines.append(f"  {t['action']} {t['symbol']} @ {px}{extra}")

    # freshest miner theses (last 2 days)
    log_p = DATA / "miner_log.jsonl"
    if log_p.exists():
        cutoff = time.time() - 172800
        theses = []
        for l in log_p.read_text().splitlines():
            if not l.strip():
                continue
            row = json.loads(l)
            if row.get("ts", 0) > cutoff:
                theses += [(s["name"], s.get("thesis", "")) for s in row.get("saved", [])]
        if theses:
            lines.append("New hypotheses:")
            for name, th in theses[-4:]:
                lines.append(f"  • {name}" + (f" — {th[:100]}" if th else ""))

    # collector health
    cs_p = DATA / "collector_status.json"
    if cs_p.exists():
        hist = json.loads(cs_p.read_text())
        if hist:
            latest = list(hist.values())[-1]
            dead = [k for k, v in latest.items() if not v.get("ok")]
            alive = [k for k, v in latest.items() if v.get("ok")]
            lines.append(f"Feeds ok: {', '.join(alive) or 'none'}"
                         + (f" | down: {', '.join(dead)}" if dead else ""))

    if lines:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
