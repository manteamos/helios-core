"""
FastAPI application — Helios Core simulation API.

Endpoints
---------
GET  /
    Serve the interactive site planner UI (Leaflet map + panel layout + simulation).

POST /api/v1/simulations/run
    Validate payload, enqueue Celery task, return 202 Accepted.
    Note: the spec document erroneously lists 222; the correct code is 202.

POST /api/v1/simulations/run-sync
    Run simulation synchronously in the API process — no Celery/Redis required.
    Intended for the UI and local development.  Not for production workloads.

GET  /api/v1/simulations/{task_id}
    Return task status and result once complete.
    States: pending -> started -> success | failure

GET  /api/v1/components/modules
    List available module profiles for the UI selector.

GET  /api/v1/components/inverters
    List available inverter specs for the UI selector.

POST /api/v1/layout/panels
    Compute a solar panel grid layout over a drawn roof polygon.

GET  /health
    Liveness probe (used by Docker HEALTHCHECK and load balancers).

Concurrency model
-----------------
The FastAPI event loop is I/O-only: it validates the request, calls
.delay() to push a message onto the Redis queue, and reads result metadata
from the Redis backend.  No NumPy code ever runs here.  All solver math
runs inside prefork Celery workers; see api/celery_app.py for the rationale.
The /run-sync endpoint is the only exception — it runs math in the event loop
and is documented as unsuitable for concurrent production load.
"""

from __future__ import annotations

from pathlib import Path

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.catalog import get_module_by_id as catalog_module_by_id
from api.catalog import search_inverters, search_modules
from api.celery_app import celery_app
from api.layout import compute_panel_layout
from api.runner import run_annual_simulation
from api.schemas import (
    InverterInfo,
    ModuleInfo,
    PanelLayoutRequest,
    PanelLayoutResponse,
    SimulationEnqueued,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
    SyncSimulationRequest,
)
from api.seeds import INVERTERS, MODULES, module_by_id
from api.tasks import simulation_task

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Helios Core — PV Yield Simulation API",
    version="0.2.0",
    description="Cloud-native PV yield simulation engine with interactive site planner.",
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the interactive site planner UI."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 OK when the process is alive."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Component catalogue
# ---------------------------------------------------------------------------


@app.get(
    "/api/v1/components/modules",
    response_model=list[ModuleInfo],
    tags=["components"],
    summary="List available solar module profiles",
)
async def list_modules() -> list[ModuleInfo]:
    result = []
    for m in MODULES:
        area = m.width_m * m.length_m
        eta = m.p_stc_w / (1000.0 * area) if area > 0 else 0.0
        result.append(
            ModuleInfo(
                id=m.itl_identifier,
                manufacturer=m.manufacturer,
                model_name=m.model_name,
                p_stc_w=m.p_stc_w,
                width_m=m.width_m,
                length_m=m.length_m,
                temp_coeff_p_pct_k=m.temp_coeff_p_pct_k,
                eta_ref=round(eta, 4),
            )
        )
    return result


@app.get(
    "/api/v1/components/inverters",
    response_model=list[InverterInfo],
    tags=["components"],
    summary="List available inverter specs",
)
async def list_inverters() -> list[InverterInfo]:
    return [InverterInfo(**inv) for inv in INVERTERS]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Panel layout
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/layout/panels",
    response_model=PanelLayoutResponse,
    tags=["layout"],
    summary="Compute solar panel grid layout over a roof polygon",
)
async def panel_layout(request: PanelLayoutRequest) -> PanelLayoutResponse:
    """
    Given a roof polygon (WGS84 lat/lon vertices) and a module ID, compute
    a regular panel grid that respects edge setbacks.

    Returns panel count, per-panel corner coordinates for map rendering,
    roof area, usable area, and installed kWp.
    """
    # Check seed modules first (signed VerifiedModuleProfile), then CEC catalog
    seed = module_by_id(request.module_id)
    if seed is not None:
        width_m, length_m, p_stc_w = seed.width_m, seed.length_m, seed.p_stc_w
    else:
        cat = catalog_module_by_id(request.module_id)
        if cat is None:
            raise HTTPException(status_code=404, detail=f"Module {request.module_id!r} not found")
        width_m = float(cat["width_m"])
        length_m = float(cat["length_m"])
        p_stc_w = float(cat["p_stc_w"])

    return compute_panel_layout(
        roof_polygon_latlon=request.roof_polygon,
        panel_width_m=width_m,
        panel_length_m=length_m,
        panel_p_stc_w=p_stc_w,
        setback_m=request.setback_m,
        row_gap_m=request.row_gap_m,
        col_gap_m=request.col_gap_m,
        orientation=request.orientation,
    )


# ---------------------------------------------------------------------------
# CEC / SAM component catalog  (21 500 modules, 3 200 inverters from pvlib)
# ---------------------------------------------------------------------------


@app.get(
    "/api/v1/catalog/modules",
    tags=["catalog"],
    summary="Search CEC module database (~21 500 entries, updates with pvlib)",
)
async def catalog_modules(
    q: str = "",
    limit: int = 20,
    offset: int = 0,
) -> dict:  # type: ignore[type-arg]
    """
    Full-text search over the NREL CEC module database bundled with pvlib.

    - ``q`` — substring search in module name (manufacturer + model, case-insensitive)
    - ``limit`` — page size (max 100)
    - ``offset`` — pagination offset

    The catalog version tracks the installed pvlib package.
    Run ``pip install --upgrade pvlib`` and restart to get the latest database.
    """
    return search_modules(q=q, limit=limit, offset=offset)


@app.get(
    "/api/v1/catalog/inverters",
    tags=["catalog"],
    summary="Search SAM CEC inverter database (~3 200 entries, updates with pvlib)",
)
async def catalog_inverters(
    q: str = "",
    limit: int = 20,
    offset: int = 0,
) -> dict:  # type: ignore[type-arg]
    """
    Full-text search over the NREL SAM inverter database bundled with pvlib.

    - ``q`` — substring search in inverter name
    - ``limit`` — page size (max 100)
    - ``offset`` — pagination offset
    """
    return search_inverters(q=q, limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Simulations — async (Celery) and sync (direct)
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/simulations/run",
    status_code=202,
    response_model=SimulationEnqueued,
    tags=["simulations"],
    summary="Enqueue an annual PV yield simulation",
)
async def enqueue_simulation(request: SimulationRequest) -> SimulationEnqueued:
    """
    Validate the simulation request and enqueue it as a Celery task.

    Returns **202 Accepted** with a task ID.
    Poll ``GET /api/v1/simulations/{task_id}`` for status and result.
    """
    task = simulation_task.delay(request.model_dump())
    return SimulationEnqueued(
        task_id=task.id,
        status="queued",
        status_url=f"/api/v1/simulations/{task.id}",
    )


@app.post(
    "/api/v1/simulations/run-sync",
    response_model=SimulationResult,
    tags=["simulations"],
    summary="Run simulation synchronously (no Celery required — UI / local dev only)",
)
async def run_simulation_sync(request: SyncSimulationRequest) -> SimulationResult:
    """
    Run the annual simulation in the API process and return the result directly.

    **Not for production concurrent load** — ties up the event loop for ~0.02 s
    per request.  Provided so the site planner UI works without a running
    Celery + Redis stack.
    """
    return run_annual_simulation(SimulationRequest(**request.model_dump(exclude={"extra_meta"})))


@app.get(
    "/api/v1/simulations/{task_id}",
    response_model=SimulationStatus,
    tags=["simulations"],
    summary="Poll simulation status and result",
)
async def get_simulation_status(task_id: str) -> SimulationStatus:
    """
    Return the current status of a queued simulation.

    Status values:

    - **pending**  — task enqueued, not yet picked up by a worker.
    - **started**  — worker is executing the simulation.
    - **success**  — complete; ``result`` field is populated.
    - **failure**  — failed; ``error`` field is populated.
    """
    async_result: AsyncResult = AsyncResult(task_id, app=celery_app)

    state = async_result.state

    if state == "SUCCESS":
        raw = async_result.result
        if not isinstance(raw, dict):
            raise HTTPException(status_code=500, detail="Unexpected result format from worker")
        return SimulationStatus(
            task_id=task_id,
            status="success",
            result=SimulationResult.model_validate(raw),
        )

    if state == "FAILURE":
        return SimulationStatus(
            task_id=task_id,
            status="failure",
            error=str(async_result.result),
        )

    if state == "STARTED":
        return SimulationStatus(task_id=task_id, status="started")

    return SimulationStatus(task_id=task_id, status="pending")
