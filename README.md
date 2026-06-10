# helios-core

Cloud-native, headless PV yield simulation engine. Continuous multi-physics
solvers (Perez spline transposition, 3-node transient thermal, stochastic
cell-level mismatch + Arrhenius aging) behind a FastAPI/Celery runtime with a
cryptographically verified component registry.

**Start here:** `CLAUDE.md` (execution plan) and `docs/SPEC.md` (requirements).

## Dev setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" --timeout 120 --retries 10
pre-commit install
pytest
```
