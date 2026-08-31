#!/bin/bash
# DARWIN orchestrator heartbeat (used by darwin.timer)
exec /usr/bin/env python3 "$(dirname "$0")/orchestrator.py"
