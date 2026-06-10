"""
FastAPI application — Helios Core simulation API.

Endpoints
---------
POST /api/v1/simulations/run
    Validate payload, enqueue Celery task, return 202 Accepted.
    Note: the spec document erroneously lists 222; the correct code is 202.

GET /api/v1/simulations/{task_id}
    Return task status and result once complete.
    States: pending -> started -> success | failure

GET /health
    Liveness probe (used by Docker HEALTHCHECK and load balancers).

Concurrency model
-----------------
The FastAPI event loop is I/O-only: it validates the request, calls
.delay() to push a message onto the Redis queue, and reads result metadata
from the Redis backend.  No NumPy code ever runs here.  All solver math
runs inside prefork Celery workers; see api/celery_app.py for the rationale.
"""

from __future__ import annotations

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException

from api.celery_app import celery_app
from api.schemas import (
    SimulationEnqueued,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
)
from api.tasks import simulation_task

app = FastAPI(
    title="Helios Core — PV Yield Simulation API",
    version="0.1.0",
    description="Cloud-native, headless PV yield simulation engine.",
)


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 OK when the process is alive."""
    return {"status": "ok"}


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

    # PENDING, RETRY, REVOKED, etc.
    return SimulationStatus(task_id=task_id, status="pending")
