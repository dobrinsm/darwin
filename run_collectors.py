#!/usr/bin/env python3
"""Run all collectors once. Used by cron (every 15 min) and manual testing."""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dotenv import load_dotenv  # noqa
load_dotenv("/root/.hermes/.env", override=False)

from darwin.bus import EventBus
from darwin.collectors import run_all

results = run_all(EventBus())
import json
print(json.dumps(results, indent=1))
