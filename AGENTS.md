# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Run the full local suite with `python -m pytest -q`.
- Strategy vocabulary changes must keep the registration surfaces in
  `darwin/spec_schema.py`, `darwin/engine.py`, `darwin/optimizer.py`, and
  `darwin/miner.py` aligned; preserve the point-in-time invariants documented
  at the top of `darwin/engine.py`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
