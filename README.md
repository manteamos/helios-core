# Helios Core

**Cloud-native, headless PV yield simulation engine** — replaces legacy desktop tools (PVsyst-class) with continuous multi-physics solvers, a cryptographically verified component registry, and a microservices API built for Kubernetes.

Built by [Amos Mante-Kwarteng](mailto:aakmanteamos@gmail.com) — O&M Manager, Helios Solar PV Plant / MKA Solutions, Ghana.

---

## What it does

Helios Core computes annual photovoltaic energy yield for utility-scale plants using physics-first solvers instead of empirical bin approximations:

| Layer | What runs |
|---|---|
| **Irradiance transposition** | Perez-Ineichen with mean-preserving cubic splines over clearness ε — continuous first derivative, validated against pvlib to <1e-9 |
| **Thermal model** | 3-node explicit FD (glass / cell / backsheet) with auto-substepping stability enforcement; steady-state matches Faiman U_c+U_v·v within 2 °C |
| **Mismatch & aging** | 12×6 cell matrix, Arrhenius degradation (E_a = 45 kJ/mol), 25-year Monte Carlo over manufacturing tolerances |
| **Component registry** | Pydantic v2 `VerifiedModuleProfile` with Ed25519 signatures, canonical JSON, accredited-lab allowlist |
| **API & workers** | FastAPI → 202 Accepted → Celery prefork workers (all NumPy math off the event loop) |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Client                                         │
│  POST /api/v1/simulations/run                   │
│  GET  /api/v1/simulations/{task_id}             │
└───────────────────┬─────────────────────────────┘
                    │ HTTP 202
         ┌──────────▼──────────┐
         │  FastAPI (uvicorn)  │  <- I/O only, no NumPy
         │  api/app.py         │
         └──────────┬──────────┘
                    │ .delay()
         ┌──────────▼──────────┐
         │  Redis broker/back  │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Celery prefork     │  <- all physics runs here
         │  worker             │
         │  ┌───────────────┐  │
         │  │ Phase 1: Perez│  │
         │  │ Phase 2: FD   │  │
         │  │   thermal     │  │
         │  │ Phase 3: MC   │  │
         │  │   mismatch    │  │
         │  └───────────────┘  │
         └─────────────────────┘
```

Why **prefork** over threads: memory isolation for multi-GB raster datasets; avoids GIL contention on short-stride NumPy loops; OOM kills hit only the affected worker. See `api/celery_app.py`.

---

## Repository layout

```
helios-core/
├── core/
│   ├── transposition/      # Perez discrete + continuous spline engines
│   ├── thermal/            # 3-node transient FD solver + Faiman steady-state
│   ├── mismatch/           # Monte Carlo cell matrix + Arrhenius aging
│   └── common/
├── registry/               # Pydantic v2 schema + Ed25519 sign/verify
├── api/                    # FastAPI app, Celery tasks, simulation runner
├── docker/                 # Multi-stage Dockerfile + docker-compose.yml
├── validation/             # Physics validation scripts (pvlib comparison, plots)
├── tests/                  # 116 pytest tests, 98.6% coverage on core/
└── docs/
    ├── SPEC.md
    └── adr/                # Architecture Decision Records
```

---

## Quick start

**Requirements:** Python 3.11, Docker (for the full stack)

```bash
# Install
pip install -e ".[dev]" --timeout 120 --retries 10

# Run all tests
pytest

# Run a quick local annual simulation (no Docker needed)
python validation/phase5_e2e.py
```

### Full docker compose stack

```bash
# Build + start Redis, API, and Celery worker
docker compose -f docker/docker-compose.yml up --build

# Submit a simulation
curl -X POST http://localhost:8000/api/v1/simulations/run \
  -H "Content-Type: application/json" \
  -d '{
    "plant_name": "MKA-Solar-Accra",
    "latitude_deg": 5.55,
    "longitude_deg": -0.20,
    "n_inverters": 45,
    "n_modules_per_inverter": 222,
    "surface_tilt_deg": 10.0,
    "surface_azimuth_deg": 180.0,
    "dt_seconds": 1800.0
  }'
# -> 202 {"task_id": "...", "status": "queued", "status_url": "/api/v1/simulations/..."}

# Poll for result
curl http://localhost:8000/api/v1/simulations/{task_id}
```

---

## API

| Method | Endpoint | Response |
|--------|----------|----------|
| `POST` | `/api/v1/simulations/run` | `202 Accepted` — task ID |
| `GET` | `/api/v1/simulations/{task_id}` | `pending` / `started` / `success` / `failure` |
| `GET` | `/health` | `200 {"status": "ok"}` |

---

## Validation results — 45-inverter / 4 MWp plant (Accra, Ghana)

```
Plant        : MKA-Solar-Accra-4MW  (5.55 N, 0.20 W)
Modules      : 9,990 x 400 W  =  3,996 kWp installed
Time step    : 1,800 s  (17,520 sub-hourly steps/year)
-----------------------------------------------------
Annual yield          :   6,738,235 kWh
Specific yield        :   1,686 kWh/kWp
Performance ratio     :   0.941
Peak POA irradiance   :   724 W/m2
Mean cell temperature :   38.8 degC
-----------------------------------------------------
Elapsed (local, no Docker)  :  0.02 s
```

---

## Phased implementation

| Phase | Description | Key acceptance criterion |
|-------|-------------|--------------------------|
| 0 | Scaffold, CI, pre-commit | pytest green, ruff + mypy strict pass |
| 1 | Continuous Perez transposition | Continuous deviates <1.5% annual POA vs discrete; brentq reverse converges 100% |
| 2 | 3-node transient thermal solver | Steady-state within 2 °C of Faiman 0-10 m/s; energy balance <0.1 W/m2 |
| 3 | Stochastic mismatch + Arrhenius aging | Year-1 mismatch 0.3-1.0%; monotonic 25-year degradation; seeded reproducibility |
| 4 | Ed25519 verified component registry | Round-trip sign->verify; tamper detection; STC power tolerance at boundary values |
| 5 | FastAPI + Celery + Docker | End-to-end 4 MWp annual simulation through queue; multi-stage non-root image |

---

## Development

```bash
# Lint + format
ruff check .
ruff format .

# Type check (strict, core/ and registry/ only)
mypy core registry

# Pre-commit (runs on every commit)
pre-commit run --all-files

# Phase validation scripts
python validation/phase1_perez.py         # Perez vs pvlib comparison plots
python validation/phase2_thermal.py       # Faiman steady-state convergence
python validation/phase3_mismatch.py      # Mismatch + aging trajectory plots
python validation/phase4_registry.py      # Ed25519 sign/verify demo
python validation/phase5_e2e.py           # 4 MWp annual simulation (local)
python validation/phase5_e2e.py --docker  # Full docker compose e2e
```

---

## Tech stack

- **Python 3.11** — NumPy/SciPy vectorized solvers
- **FastAPI** + **Celery** (Redis broker, prefork pool)
- **Pydantic v2** throughout; **pydantic-settings** for config
- **cryptography** — Ed25519 sign/verify
- **Docker** — multi-stage, non-root `appuser`, `python:3.11-slim`, no GUI libs
- **pytest** + **hypothesis** — 116 tests, 98.6% coverage
- **ruff** + **mypy strict** — enforced via pre-commit

---

## License

Proprietary — Amos Mante-Kwarteng / MKA Solutions
