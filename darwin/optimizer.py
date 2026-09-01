"""Loop 2.5 — the MUTATE optimizer.

Turns gauntlet MUTATE verdicts into new fully-tested specs. For each seed:
enumerate a BOUNDED grid of param variations (capped, deterministic, seed
included), re-run every candidate through the exact same walk-forward gauntlet
(IS/OOS windows never change), and rank by an OOS-anchored fitness that
penalizes candidates whose OOS gain comes from one lucky window.

Guardrails (from backtest-history lessons):
- cap search breadth per seed (MAX_CANDIDATES) — no explosion
- rank by OOS sharpe + window consistency, never in-sample metrics
- lucky-window rejection: fitness requires winning windows in >= 1/2 of OOS
  windows and penalizes single-window heroes hard
- the child spec is validated by the same schema and re-judged by the same
  gauntlet before it can ever reach the arena

A promoted child REPLACES its parent in specs/ (parent kept as <id>.json.seeded);
the seed's gauntlet verdict line stays honest via runs/<seed>/optimizer.json.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from .bus import EventBus
from .gauntlet import RUNS_DIR, evaluate_spec
from .spec_schema import SPEC_DIR, load_spec, save_spec, validate_spec

# caps — search breadth is a feature, not a bug
MAX_CANDIDATES = 48          # per seed (seed included in the count)
MAX_PAIR_COMBOS = 24         # round-2 two-param combos per seed

# lucky-window rejection: a candidate must keep avg OOS sharpe >= LOO_MIN even
# when its single best window is excluded. This is the guardrail that stops a
# one-window hero (e.g. an 11-window run with sharpes [0,0,..,19.15,..,-1.12])
# from winning the search on the back of one outlier.
LOO_MIN = 0.1

# ordered grids per node type (only param axes worth searching; tf unchanged)
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "ema_cross_up":         {"fast": [5, 8, 12, 20, 30, 50],
                             "slow": [50, 100, 150, 200, 300]},
    "ema_cross_down":       {"fast": [5, 8, 12, 20, 30, 50],
                             "slow": [50, 100, 150, 200, 300]},
    "price_above_sma":      {"period": [20, 50, 100, 150, 200, 300]},
    "price_below_sma":      {"period": [20, 50, 100, 150, 200, 300]},
    "rsi_below":            {"period": [7, 14, 21], "threshold": [15, 20, 25, 30]},
    "rsi_above":            {"period": [7, 14, 21], "threshold": [55, 65, 70, 75]},
    "vol_spike":            {"mult": [1.5, 2.0, 3.0], "lookback": [10, 20, 30, 50]},
    "drawdown_from_high":   {"pct": [0.1, 0.15, 0.2, 0.3],
                             "lookback_d": [20, 30, 45, 60]},
    "runup_from_low":       {"pct": [0.15, 0.2, 0.25, 0.3, 0.4],
                             "lookback_d": [30, 45, 60, 90]},
    "fear_greed_below":     {"threshold": [25, 35, 45, 55, 65, 80]},
    "fear_greed_above":     {"threshold": [35, 45, 55, 65, 75]},
    "news_sentiment_above": {"threshold": [-0.2, 0.0, 0.2],
                             "window_h": [48, 96, 168]},
    "news_sentiment_below": {"threshold": [-0.2, 0.0, 0.2],
                             "window_h": [48, 96, 168]},
    "convergence":          {"min_confidence": [60, 70], "window_d": [14, 30, 45]},
    "cross_asset_score":    {"min_score": [1, 2, 3], "mom_h": [48, 96, 168]},
    "insider_buy":          {"window_d": [14, 30, 45]},
    "iv_skew_above":        {"percentile": [0.7, 0.85], "lookback_d": [30, 60]},
}
STOPS_GRID: dict[str, list] = {
    "trail_pct": [0.1, 0.15, 0.2, 0.25, 0.3],
    "hard_pct": [0.08, 0.1, 0.15, 0.2],
}


# ---------------------------------------------------------------- helpers
def _node_targets(spec: dict) -> list[tuple[str, str, int, dict]]:
    """[(section, list_name, node_index, node)] in deterministic order."""
    out = []
    for sec in ("entry", "exit"):
        for lst in ("all", "any"):
            for ni, node in enumerate((spec.get(sec) or {}).get(lst) or []):
                out.append((sec, lst, ni, node))
    return out


def _apply_node_param(spec: dict, sec: str, lst: str, ni: int,
                      pname: str, value) -> dict:
    m = copy.deepcopy(spec)
    m[sec][lst][ni][pname] = value
    return m


def _apply_stop(spec: dict, pname: str, value) -> dict:
    m = copy.deepcopy(spec)
    m.setdefault("exit", {}).setdefault("stops", {})[pname] = value
    return m


def _spec_key(spec: dict) -> str:
    """Canonical identity of the mutable surface (params only)."""
    nodes = [[n.get("type"), {k: v for k, v in n.items() if k != "type"}]
             for _, _, _, n in _node_targets(spec)]
    stops = (spec.get("exit") or {}).get("stops") or {}
    risk = spec.get("risk") or {}
    return json.dumps([nodes, stops, risk], sort_keys=True)


def _mutations_for_node(node: dict) -> list[tuple[str, object]]:
    """[(param, value)] variations for one node, schema-sane, seed-val dropped."""
    grid = PARAM_GRIDS.get(node["type"], {})
    out = []
    for pname, values in grid.items():
        for v in values:
            if node.get(pname) == v:
                continue
            if node["type"] == "ema_cross_up" and pname == "fast" \
                    and v >= node.get("slow", 10**9):
                continue
            if node["type"] == "ema_cross_up" and pname == "slow" \
                    and v <= node.get("fast", 0):
                continue
            out.append((pname, v))
    return out


def enumerate_mutations(spec: dict, cap: int = MAX_CANDIDATES,
                        pair_cap: int = MAX_PAIR_COMBOS) -> list[dict]:
    """Bounded, deterministic candidate list. Round 1: single-param changes.
    Round 2: pairs on DIFFERENT targets (never both params of one node)."""
    seen = {_spec_key(spec)}
    singles: list[tuple[tuple, dict]] = []   # ((target-tag, param, value), spec)

    for sec, lst, ni, node in _node_targets(spec):
        for pname, v in _mutations_for_node(node):
            m = _apply_node_param(spec, sec, lst, ni, pname, v)
            k = _spec_key(m)
            if k in seen:
                continue
            seen.add(k)
            singles.append(((f"{sec}.{lst}[{ni}]", pname, v), m))

    stops = (spec.get("exit") or {}).get("stops") or {}
    for pname, values in STOPS_GRID.items():
        if pname not in stops:
            continue
        for v in values:
            if stops[pname] == v:
                continue
            m = _apply_stop(spec, pname, v)
            k = _spec_key(m)
            if k in seen:
                continue
            seen.add(k)
            singles.append(((f"exit.stops.{pname}", pname, v), m))

    out = [m for _, m in singles]
    if len(out) >= cap:
        return out[:cap]

    # round 2 — pair combos across different targets
    for i in range(len(singles)):
        if len(out) >= min(cap, len(singles) + pair_cap):
            break
        (t1, _, _), m1 = singles[i]
        for j in range(i + 1, len(singles)):
            (t2, p2, v2), _ = singles[j]
            if t2 == t1:                 # same node/stops target: skip
                continue
            sec, lst, ni = None, None, None
            if t2.startswith("exit.stops."):
                m = _apply_stop(m1, p2, v2)
            else:
                target, _, _ = t2.rpartition("[")
                sec, lst = t2.split(".")[0], t2.split(".")[1].split("[")[0]
                ni = int(t2.split("[")[1].rstrip("]"))
                m = _apply_node_param(m1, sec, lst, ni, p2, v2)
            k = _spec_key(m)
            if k in seen:
                continue
            seen.add(k)
            out.append(m)
            if len(out) >= min(cap, len(singles) + pair_cap):
                break
    return out[:cap]


def _window_sharpes(rep: dict) -> list[float]:
    return [w["oos"]["sharpe"] for w in rep.get("walk_forward") or []]


def loo_sharpe(rep: dict) -> float:
    """Leave-one-out OOS sharpe: mean over windows EXCLUDING the single best.
    Near zero or negative => the edge lives in one lucky window."""
    s = sorted(_window_sharpes(rep))
    if len(s) < 3:
        return rep.get("avg_oos_sharpe", 0.0)
    return (sum(s) - s[-1]) / (len(s) - 1)


def fitness(rep: dict) -> float:
    """OOS-anchored, lucky-window-rejecting rank score (in-sample NEVER seen).

    Hard gate: excluding its best window, a candidate must still average
    OOS sharpe >= LOO_MIN — otherwise it's disqualified outright, however
    good the headline average looks."""
    if rep["verdict"] == "KILL" and "no valid walk-forward" in rep.get("why", ""):
        return -1e9
    s = rep["avg_oos_sharpe"]
    if s <= 0:
        return s
    loo = loo_sharpe(rep)
    if loo < LOO_MIN:
        return -1e6 + s        # dominated by any candidate that survives LOO
    sharpes = _window_sharpes(rep)
    active = [w for w in sharpes if w != 0.0]   # 0-sharpe windows = no trades
    n_win = sum(1 for x in sharpes if x > 0)
    win_share = (n_win / len(active)) if active else 0.0
    s += 0.5 * win_share                      # reward consistency
    s += 0.25 if rep["oos_compound_pct"] > 0 else -0.5
    s += min(loo, 1.0) * 0.25                 # reward edge that survives LOO
    return s


# ---------------------------------------------------------------- search
def optimize_seed(bus: EventBus, seed: dict, start: str = "2020-01-01") -> dict:
    """Grid-search one MUTATE seed. Returns search summary; writes child spec
    + gauntlet report ONLY if a candidate beats the seed's fitness."""
    t0 = time.time()
    from .engine import Context, load_klines
    df = load_klines(bus, seed["asset"]["symbol"], seed["asset"]["tf"], start=start)
    ctx = Context(bus, seed["asset"]["symbol"], seed["asset"]["tf"], df.index)

    seed_rep = evaluate_spec(bus, seed, df=df, ctx=ctx)
    seed_fit = fitness(seed_rep)

    cands = enumerate_mutations(seed)
    best = {"fit": seed_fit, "spec": None, "rep": None, "key": _spec_key(seed)}
    tested = 0
    for cand in cands:
        if validate_spec(cand):
            continue                       # schema guard: never evaluate junk
        try:
            rep = evaluate_spec(bus, cand, df=df, ctx=ctx)
        except Exception:
            continue
        tested += 1
        fit = fitness(rep)
        if fit > best["fit"]:              # strict >: deterministic tie-break
            best = {"fit": fit, "spec": cand, "rep": rep}

    out = {
        "seed_id": seed["spec_id"], "seed_name": seed["name"],
        "seed_fitness": round(seed_fit, 3),
        "seed_avg_oos_sharpe": seed_rep["avg_oos_sharpe"],
        "seed_oos_loo_sharpe": seed_rep.get("oos_loo_sharpe"),
        "seed_oos_compound_pct": seed_rep["oos_compound_pct"],
        "candidates_enumerated": len(cands),
        "candidates_tested": tested,
        "improved": best["spec"] is not None,
        "secs": round(time.time() - t0, 1),
    }

    if best["spec"] is None:
        out["child"] = None
        out["note"] = "no candidate beat the seed under OOS-anchored fitness"
        return out

    child = copy.deepcopy(best["spec"])
    child.pop("spec_id", None)
    child.pop("checksum", None)
    child.pop("created_ts", None)
    child["name"] = f"{seed['name']} [opt]"[:70]
    child["version"] = int(seed.get("version", 1)) + 1
    child["provenance"] = {
        "mined_from": [seed["spec_id"]],
        "model": "optimizer-grid-v1",
        "parent_spec_id": seed["spec_id"],
        "optimizer": {
            "fitness": round(best["fit"], 3),
            "seed_fitness": round(seed_fit, 3),
            "candidates_tested": tested,
        },
    }
    errs = validate_spec(child)
    if errs:
        out["child"] = None
        out["note"] = "best candidate failed schema: " + "; ".join(errs)
        return out
    save_spec(child)                        # stamps spec_id/checksum in place
    crep = evaluate_spec(bus, load_spec(child["spec_id"]), df=df, ctx=ctx)
    # persist the child's gauntlet report so the arena sees it immediately
    cdir = RUNS_DIR / child["spec_id"]
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "report.json").write_text(json.dumps(crep, indent=1))

    # retire the seed: renamed out of the live population, history kept
    seed_path = SPEC_DIR / f"{seed['spec_id']}.json"
    seed_path.rename(seed_path.with_suffix(".json.seeded"))

    out.update({
        "child": child["spec_id"],
        "child_name": child["name"],
        "child_avg_oos_sharpe": crep["avg_oos_sharpe"],
        "child_oos_loo_sharpe": crep.get("oos_loo_sharpe"),
        "child_oos_compound_pct": crep["oos_compound_pct"],
        "child_verdict": crep["verdict"],
        "delta_fitness": round(best["fit"] - seed_fit, 3),
    })
    return out


def mutate_queue(bus: EventBus) -> list[dict]:
    """Current MUTATE-verdict specs with fresh reports."""
    out = []
    now = time.time()
    for p in sorted(SPEC_DIR.glob("*.json")):
        rep_p = RUNS_DIR / p.stem / "report.json"
        if not rep_p.exists():
            continue
        rep = json.loads(rep_p.read_text())
        if rep.get("verdict") == "MUTATE" and now - rep.get("ran_at", 0) < 36 * 3600:
            out.append(load_spec(p.stem))
    return out


def run_all_mutations(bus: EventBus | None = None) -> dict:
    """Optimize every spec currently queued for MUTATE."""
    bus = bus or EventBus()
    seeds = mutate_queue(bus)
    results = []
    for seed in seeds:
        try:
            res = optimize_seed(bus, seed)
        except Exception as e:
            res = {"seed_id": seed["spec_id"], "error": f"{type(e).__name__}: {e}"[:200]}
        results.append(res)
        opt_p = RUNS_DIR / seed["spec_id"] / "optimizer.json"
        opt_p.parent.mkdir(parents=True, exist_ok=True)
        opt_p.write_text(json.dumps(res, indent=1))
    children = [r for r in results if r.get("child")]
    return {"seeds": len(seeds), "children": children, "results": results}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dotenv import load_dotenv
    load_dotenv("/root/.hermes/.env", override=False)
    bus = EventBus()
    if len(sys.argv) > 1:
        seed = load_spec(sys.argv[1])
        out = {"seeds": 1, "children": [], "results": [optimize_seed(bus, seed)]}
        out["children"] = [r for r in out["results"] if r.get("child")]
    else:
        out = run_all_mutations(bus)
    for r in out["results"]:
        if r.get("child"):
            print(f"{r['seed_id']} -> {r['child']}  {r['child_verdict']}"
                  f"  oos={r['child_avg_oos_sharpe']:+.2f}"
                  f" loo={r['child_oos_loo_sharpe']:+.2f}"
                  f" comp={r['child_oos_compound_pct']:+.1f}%"
                  f"  dFit={r['delta_fitness']:+.2f}"
                  f"  ({r['candidates_tested']} tested, {r['secs']}s)")
        elif r.get("error"):
            print(f"{r['seed_id']}  ERROR {r['error']}")
        else:
            print(f"{r['seed_id']}  no improvement ({r.get('candidates_tested', 0)} tested, "
                  f"seed fit {r['seed_fitness']}, {r['secs']}s)")
    print(f"\n{len(out['children'])}/{out['seeds']} seeds produced children")
