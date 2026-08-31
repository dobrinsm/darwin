"""Strategy SPEC schema — the contract between the Miner (LLM) and the Gauntlet.

A spec is pure-data JSON. No code. The Gauntlet compiles a spec into a
deterministic, point-in-time backtest. Everything is falsifiable before a
run is allowed: unknown nodes, bad params, or stale confidence all reject.

Structure:
{
  "spec_id": "...", "name": "...", "version": 1,
  "created_ts": ..., "provenance": {"mined_from": [dedup_keys...], "model": "..."},
  "asset": {"class": "crypto", "symbol": "DOGEUSDT", "tf": "1d"},
  "entry": {"all": [node...], "any": [node...]},
  "exit":  {"all": [...], "any": [...], "stops": {"trail_pct": 0.2}},
  "risk":  {"leverage": 3, "max_pos_frac": 0.3, "cooldown_bars": 2},
  "direction": "long",           # long-only v1
  "confidence": 0.55             # Miner's prior; gauntlet re-scores it
}

Every node: {"type": "<known node>", ...params} — see NODE_TYPES.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

from jsonschema import Draft202012Validator

SPEC_DIR = Path(__file__).resolve().parent.parent / "specs"

# every node's allowed params and their python types
NODE_TYPES: dict[str, dict] = {
    # --- price/TA (tf-aware) ---
    "price_above_sma":    {"period": int, "tf": str},
    "price_below_sma":    {"period": int, "tf": str},
    "ema_cross_up":       {"fast": int, "slow": int, "tf": str},
    "ema_cross_down":     {"fast": int, "slow": int, "tf": str},
    "rsi_below":          {"period": int, "threshold": (int, float), "tf": str},
    "rsi_above":          {"period": int, "threshold": (int, float), "tf": str},
    "vol_spike":          {"mult": (int, float), "lookback": int, "tf": str},
    "drawdown_from_high": {"pct": (int, float), "lookback_d": int},
    "runup_from_low":     {"pct": (int, float), "lookback_d": int},
    # --- news (finlight / alphai events on the bus) ---
    "news_sentiment_below": {"threshold": (int, float), "window_h": (int, float),
                             "min_conf": (int, float)},
    "news_sentiment_above": {"threshold": (int, float), "window_h": (int, float),
                             "min_conf": (int, float)},
    # --- positioning / sentiment (agentservices, tradestie) ---
    "fear_greed_below":   {"threshold": int},
    "fear_greed_above":   {"threshold": int},
    "wsb_rank_above":     {"rank": int},          # ticker in WSB top-N
    # --- smart money (alphasmo) ---
    "convergence":        {"min_confidence": int, "window_d": int},
    "insider_buy":        {"window_d": int},
    # --- options (lse) ---
    "iv_skew_above":      {"percentile": (int, float), "lookback_d": int},
    # --- cross-asset (lse) ---
    "cross_asset_score":  {"min_score": int, "assets": list, "mom_h": int},
}

_JSON_SCHEMA = {
    "type": "object",
    "required": ["name", "asset", "entry", "exit", "risk", "direction"],
    "properties": {
        "spec_id": {"type": "string"},
        "name": {"type": "string", "minLength": 3},
        "version": {"type": "integer", "minimum": 1},
        "created_ts": {"type": "number"},
        "provenance": {"type": "object"},
        "asset": {
            "type": "object",
            "required": ["class", "symbol", "tf"],
            "properties": {
                "class": {"enum": ["crypto"]},
                "symbol": {"type": "string", "pattern": "^[A-Z]+USDT$"},
                "tf": {"enum": ["4h", "1d"]},
            },
        },
        "direction": {"enum": ["long"]},          # long-only v1
        "entry": {
            "type": "object",
            "properties": {
                "all": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                "any": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            },
            "anyOf": [{"required": ["all"]}, {"required": ["any"]}],
        },
        "exit": {
            "type": "object",
            "properties": {
                "all": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                "any": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                "stops": {
                    "type": "object",
                    "properties": {
                        "trail_pct": {"type": "number", "minimum": 0.05, "maximum": 0.6},
                        "hard_pct": {"type": "number", "minimum": 0.05, "maximum": 0.6},
                    },
                },
            },
        },
        "risk": {
            "type": "object",
            "required": ["leverage", "max_pos_frac"],
            "properties": {
                "leverage": {"type": "integer", "minimum": 1, "maximum": 3},
                "max_pos_frac": {"type": "number", "minimum": 0.05, "maximum": 0.5},
                "cooldown_bars": {"type": "integer", "minimum": 0, "maximum": 10},
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "$defs": {
        "node": {
            "type": "object",
            "properties": {"type": {"type": "string"}},
            "required": ["type"],
            "additionalProperties": True,
        },
    },
}

_PARAM_LIMITS = {
    "period": (2, 400), "fast": (2, 400), "slow": (3, 400),
    "threshold": (-5, 105), "mult": (1.0, 20.0), "lookback": (3, 500),
    "rank": (1, 50), "min_confidence": (0, 100), "window_d": (1, 120),
    "window_h": (1, 720), "min_conf": (0, 1), "pct": (0.01, 0.95),
    "percentile": (0, 1), "min_score": (0, 5), "mom_h": (1, 336),
    "tf": None, "assets": None,
}


def _check_nodes(spec: dict, errors: list[str]) -> None:
    for section in ("entry", "exit"):
        sec = spec.get(section) or {}
        for node in list(sec.get("all") or []) + list(sec.get("any") or []):
            if not isinstance(node, dict) or "type" not in node:
                errors.append(f"{section}: node missing 'type': {node}")
                continue
            ntype = node["type"]
            if ntype not in NODE_TYPES:
                errors.append(f"{section}: unknown node type '{ntype}'")
                continue
            allowed = NODE_TYPES[ntype]
            for k, v in node.items():
                if k == "type":
                    continue
                if k not in allowed:
                    errors.append(f"{section}/{ntype}: unknown param '{k}'")
                    continue
                if not isinstance(v, allowed[k]):
                    errors.append(f"{section}/{ntype}: param '{k}' wrong type"
                                  f" ({type(v).__name__})")
            for k in allowed:
                if k not in node:
                    errors.append(f"{section}/{ntype}: missing param '{k}'")
                    continue
                lim = _PARAM_LIMITS.get(k)
                if lim and not (lim[0] <= node[k] <= lim[1]):
                    errors.append(f"{section}/{ntype}: param '{k}'={node[k]} out of range {lim}")


def validate_spec(spec: dict) -> list[str]:
    """Returns list of errors; empty list == valid."""
    errors: list[str] = []
    try:
        Draft202012Validator(_JSON_SCHEMA).validate(spec)
    except Exception as e:
        return [f"schema: {e}"]
    _check_nodes(spec, errors)
    fast, slow = None, None
    for node in (spec["entry"].get("all") or []) + (spec["entry"].get("any") or []):
        if node.get("type") == "ema_cross_up":
            fast, slow = node["fast"], node["slow"]
    if fast is not None and fast >= slow:
        errors.append("entry/ema_cross_up: fast must be < slow")
    return errors


def new_spec_id() -> str:
    return "spec_" + uuid.uuid4().hex[:10]


def save_spec(spec: dict) -> Path:
    """Validate, stamp, persist. Raises ValueError on invalid."""
    errs = validate_spec(spec)
    if errs:
        raise ValueError("invalid spec: " + "; ".join(errs))
    spec.setdefault("spec_id", new_spec_id())
    spec.setdefault("version", 1)
    spec.setdefault("created_ts", time.time())
    body = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    spec["checksum"] = hashlib.sha256(body.encode()).hexdigest()[:16]
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    p = SPEC_DIR / f"{spec['spec_id']}.json"
    p.write_text(json.dumps(spec, indent=1))
    return p


def load_spec(spec_id: str) -> dict:
    return json.loads((SPEC_DIR / f"{spec_id}.json").read_text())


def demo_spec() -> dict:
    """Sanity-check spec: EMA trend entry + froth veto, long DOGE daily."""
    return {
        "name": "demo: ema trend w/ froth veto",
        "asset": {"class": "crypto", "symbol": "DOGEUSDT", "tf": "1d"},
        "direction": "long",
        "entry": {"all": [
            {"type": "ema_cross_up", "fast": 20, "slow": 100, "tf": "1d"},
            {"type": "fear_greed_below", "threshold": 80},
        ]},
        "exit": {"any": [
            {"type": "ema_cross_down", "fast": 20, "slow": 100, "tf": "1d"},
        ], "stops": {"trail_pct": 0.25}},
        "risk": {"leverage": 3, "max_pos_frac": 0.3, "cooldown_bars": 2},
        "confidence": 0.5,
        "provenance": {"mined_from": ["demo"], "model": "human"},
    }
