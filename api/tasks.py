"""
Celery simulation tasks.

All NumPy/SciPy computation happens here, inside prefork worker processes.
The FastAPI event loop never imports or calls these functions directly.
"""

from __future__ import annotations

import logging

from api.celery_app import celery_app
from api.runner import run_annual_simulation
from api.schemas import SimulationRequest

_LOG = logging.getLogger(__name__)


@celery_app.task(bind=True, name="helios.run_simulation")
def simulation_task(self, params_dict: dict) -> dict:  # type: ignore[type-arg]
    """
    Deserialize a SimulationRequest, run the full annual simulation, return result dict.

    Parameters
    ----------
    params_dict : dict
        JSON-serializable SimulationRequest fields.

    Returns
    -------
    dict
        JSON-serializable SimulationResult fields.
    """
    _LOG.info(
        "task %s: starting simulation for %r",
        self.request.id,
        params_dict.get("plant_name"),
    )
    params = SimulationRequest.model_validate(params_dict)
    result = run_annual_simulation(params)
    _LOG.info(
        "task %s: done -- yield=%.0f kWh  PR=%.3f",
        self.request.id,
        result.annual_yield_kwh,
        result.performance_ratio,
    )
    return result.model_dump()
