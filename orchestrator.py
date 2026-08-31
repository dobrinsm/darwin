#!/usr/bin/env python3
"""DARWIN orchestrator — the heartbeat. Run every 15 min via cron/systemd timer.

Phase A (every tick):   collectors -> bus
Phase B (daily 04:10):  miner proposes new specs
Phase C (after mine):   gauntlet re-runs ALL specs, arena re-syncs
Phase D (every tick):   arena step (paper positions follow promoted specs)
"""
import json
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

DATA = str(pathlib.Path(__file__).resolve().parent / "data")


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)


def main():
    from dotenv import load_dotenv
    load_dotenv("/root/.hermes/.env", override=False)

    from darwin.bus import EventBus
    from darwin.collectors import run_all as run_collectors
    from darwin.miner import mine
    from darwin.gauntlet import run_spec
    from darwin.spec_schema import SPEC_DIR, load_spec
    from darwin.arena import step as arena_step

    bus = EventBus()

    # Phase A — collect
    coll = run_collectors(bus)
    log("collectors: " + json.dumps(
        {k: f"{'ok' if v['ok'] else 'ERR'}(+{v['new']})" for k, v in coll.items()}))

    # Phase B — mine once per day at 04:10-04:59 UTC window
    now = time.gmtime()
    day = time.strftime("%Y-%m-%d", time.gmtime())
    mine_flag = DATA + "/last_mine.txt"
    do_mine = (now.tm_hour == 4 and now.tm_min >= 10) or \
              (not __import__("pathlib").Path(mine_flag).exists())
    if do_mine:
        try:
            out = mine(bus)
            log(f"miner: +{len(out['saved'])} saved, {len(out['rejected'])} rejected")
            open(mine_flag, "w").write(day)
        except Exception as e:
            log(f"miner FAILED: {e}")
    gauntlet_flag = DATA + "/last_gauntlet.txt"
    need_gauntlet = do_mine or not __import__("pathlib").Path(gauntlet_flag).exists()

    # Phase C — gauntlet after new blood
    if need_gauntlet:
        try:
            lines = []
            for p in sorted(SPEC_DIR.glob("*.json")):
                rep = run_spec(bus, load_spec(p.stem))
                lines.append(f"{rep['verdict']:8} {rep['spec_id']} {rep['name'][:38]}"
                             f" oos={rep['avg_oos_sharpe']:+.2f}")
                log(f"gauntlet: {lines[-1]}")
            open(gauntlet_flag, "w").write(day)
        except Exception as e:
            log(f"gauntlet FAILED: {e}\n{traceback.format_exc()[-400:]}")

    # Phase D — arena
    try:
        ar = arena_step(bus)
        log("arena: " + json.dumps(ar["actions"]) +
            (f" positions={list(ar['positions'].keys())}" if ar["positions"] else ""))
    except Exception as e:
        log(f"arena FAILED: {e}\n{traceback.format_exc()[-400:]}")

    # heartbeat
    log(f"bus total: {bus._count()} events")


if __name__ == "__main__":
    main()
