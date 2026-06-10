# CLAUDE.md — Next-Gen PV Yield Simulation Engine ("Helios Core")

## Project owner & review protocol
- Owner: Amos Mante-Kwarteng (O&M Manager, Helios Solar PV Plant; MKA Solutions).
- Claude Code drives implementation and terminal execution autonomously. Amos reviews diffs and validation plots — do not block waiting for approval on routine steps; DO pause and summarize before any destructive action (deleting caches, dropping DB tables, force-pushing).
- Debugging discipline (carried over from prior projects): change ONE variable at a time; never delete a cached/generated artifact without a verified rebuild path; pip installs on Ghanaian connections need `--timeout 120 --retries 10`.

## Mission
Cloud-native, headless PV yield simulation engine replacing legacy desktop tools (PVsyst-class). Microservices, Kubernetes-ready, cryptographically verified component registry, and continuous multi-physics solvers instead of empirical step-bin approximations.

## Tech stack (fixed decisions — do not relitigate without flagging)
- Python 3.11, NumPy/SciPy vectorized solvers (C++ extensions only if profiling proves need).
- FastAPI (async) + Celery with Redis broker. gRPC deferred to Phase 5.
- PostgreSQL 15 (JSONB for module profiles) + Redis cache. SQLite acceptable for local dev fixture DB.
- S3-compatible object storage (MinIO locally) for raster datasets and raw time-series output.
- Pydantic v2 throughout. Ed25519 via `pynacl` or `cryptography`.
- Docker: multi-stage, non-root, `python:3.11-slim`, no GUI/X11 deps.
- Tests: pytest + hypothesis for solver property tests. Target ≥85% coverage on `core/`.

## Repository layout
```
helios-core/
├── core/
│   ├── transposition/      # Continuous Perez engine (Phase 1)
│   ├── thermal/            # 3-node transient solver (Phase 2)
│   ├── mismatch/           # Monte Carlo cell matrix + Arrhenius aging (Phase 3)
│   └── common/             # units, constants, time-series containers
├── registry/               # Pydantic schemas + Ed25519 verification (Phase 4)
├── api/                    # FastAPI app, Celery tasks (Phase 5)
├── validation/             # Notebooks/scripts comparing vs pvlib & PVsyst exports
├── tests/
├── docker/
└── docs/architecture.md    # ASCII topology + ADRs
```

## Phased implementation plan

### Phase 0 — Scaffold & validation harness (do first)
- Repo scaffold, pyproject.toml, pre-commit (ruff + mypy strict on core/).
- Install pvlib as the REFERENCE implementation only — never import it inside `core/`; use it exclusively in `validation/` and tests as ground truth.
- CI: pytest on push (GitHub Actions).

### Phase 1 — Continuous Perez Transposition Engine
- Implement standard Perez-Ineichen first (discrete bins) as `perez_discrete.py` to validate against pvlib within 1e-9. This is the regression anchor.
- Then `perez_continuous.py`: fit mean-preserving cubic splines (scipy `CubicSpline` with constrained integral preservation per bin, or `PchipInterpolator` if monotonicity issues arise) over clearness ε for F11..F23. First derivative must be continuous — verify numerically.
- Expose `transpose(ghi, dni, dhi, solar_pos, surface_tilt, surface_azimuth)` fully vectorized over 8760+ rows.
- Acceptance: continuous model deviates from discrete by <1.5% annual POA on 3 TMY datasets (use Accra/Kumasi/Tamale climates from PVGIS or Open-Meteo); reverse transposition via `scipy.optimize.brentq` converges on 100% of test cases.

### Phase 2 — 3-Node Transient Thermal Solver
- Nodes: glass T_g, cell T_c, backsheet T_b. Explicit finite difference; enforce stability criterion Δt ≤ min(C_p,i·m_i / ΣU_ij) with automatic sub-stepping when violated — make this a hard assertion, not a silent clamp.
- h_front/h_rear as functions of wind speed AND direction relative to row azimuth, plus an exposure factor field per module position.
- Acceptance: steady-state limit converges to Faiman/PVsyst U_c+U_v·v model within 2°C across 0–10 m/s; energy balance closes to <0.1 W/m² residual per step.

### Phase 3 — Stochastic Mismatch & Aging Matrix
- 12×6 cell matrix per module; vectorize across (n_modules, 72) — no Python loops over cells.
- Spatial rack thermal gradient field (corner vs center modules) feeding Phase 2 solver per-position.
- Arrhenius degradation, E_a = 45 kJ/mol, k = k_ref·exp(−E_a/R·(1/T − 1/T_ref)); integrate over hourly T_c history, 25-year horizon with annual time compression (document the compression scheme in an ADR).
- Series-string current bottleneck: I_string = min over cells after degradation + soiling draws. Monte Carlo over manufacturing tolerance (σ from datasheet binning).
- Acceptance: deterministic seed reproducibility; year-1 mismatch loss in 0.3–1.0% plausibility band; degradation trajectory monotonic per cell.

### Phase 4 — Verified Component Registry
- Pydantic v2 `VerifiedModuleProfile`: manufacturer/model, physical geometry, cell topology, STC coefficients, low-light params, polynomial IAM coefficients, `itl_identifier`, `digital_signature`.
- `@model_validator`: assert |P_stc − I_mpp·V_mpp| / P_stc ≤ 0.001, raise `ComponentValidationError` (custom, clean message) on failure.
- Ed25519: canonical JSON serialization (sorted keys, no whitespace) → sign/verify. Maintain an accredited-lab public-key allowlist table. Reject unsigned/invalid payloads at ingestion; log attempt.
- Acceptance: round-trip sign→verify; tampered-field detection; tolerance validator unit tests at boundary values.

### Phase 5 — API, Workers, Docker
- `POST /api/v1/simulations/run` → validate payload → enqueue Celery task → return 202 Accepted (NOTE: the spec document says "222" — that is a typo; use HTTP 202) with task ID. `GET /api/v1/simulations/{task_id}` for status/result pointer (S3 key).
- Concurrency model: FastAPI event loop never runs solver math; all NumPy work in Celery workers (prefork pool, concurrency = physical cores); document why prefork beats threads here (GIL released in NumPy but memory isolation wins for multi-GB rasters).
- Multi-stage Dockerfile: builder stage compiles wheels; runtime stage non-root `appuser`, slim, no X11/GUI libs, HEALTHCHECK, pinned digests.
- Acceptance: full annual sub-hourly simulation for a 45-inverter / 4-MVS plant completes end-to-end through the queue; container passes `docker scout`/trivy scan with no HIGH CVEs from our layers.

## Standing rules for Claude Code in this repo
1. Every solver module ships with: docstring stating governing equations, units of every parameter, and a `validation/` comparison script.
2. No placeholder code or `# TODO: implement math` — full implementations only.
3. When a numerical result disagrees with pvlib, do not "fix" by fudging coefficients; isolate the divergence with a minimal reproduction first.
4. Commit per logical unit with descriptive messages; never squash away validation evidence.
5. Flag (don't silently implement) anything that affects bankability claims — e.g., changing the spline fitting method or degradation model.
