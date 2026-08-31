"""DARWIN — evolutionary strategy foundry.

Loop 0: ingest  → darwin.bus + darwin.collectors
Loop 1: mine    → darwin.miner (LLM proposes strategy specs)
Loop 2: gauntlet → darwin.gauntlet (point-in-time backtest, promote/mutate/kill)
Loop 3: arena   → darwin.arena (parallel paper trading)
Loop 4: execute → darwin.execute (CCXT + risk governor)
"""

import pathlib

__version__ = "0.1.0"

# repo root = parent of the package dir (portable; no hardcoded paths)
ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
