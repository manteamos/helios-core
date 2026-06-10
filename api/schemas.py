"""
Pydantic v2 request and response schemas for the simulation API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """Input parameters for an annual PV yield simulation."""

    plant_name: str = Field(min_length=1, max_length=128)
    latitude_deg: float = Field(ge=-90.0, le=90.0, description="WGS84 latitude [deg]")
    longitude_deg: float = Field(ge=-180.0, le=180.0, description="WGS84 longitude [deg]")
    n_inverters: int = Field(ge=1, le=500)
    n_modules_per_inverter: int = Field(ge=1, le=10_000)
    surface_tilt_deg: float = Field(
        ge=0.0, le=90.0, default=10.0, description="Fixed tilt from horizontal [deg]"
    )
    surface_azimuth_deg: float = Field(
        ge=0.0, lt=360.0, default=180.0, description="Clockwise from North [deg]"
    )
    module_p_stc_w: float = Field(
        gt=0.0, default=400.0, description="Module rated power at STC [W]"
    )
    module_temp_coeff_pct_k: float = Field(
        default=-0.38, description="Power temperature coefficient [%/K]"
    )
    module_eta_ref: float = Field(
        gt=0.0, le=1.0, default=0.20, description="Module STC efficiency [0-1]"
    )
    albedo: float = Field(ge=0.0, le=1.0, default=0.25, description="Ground albedo")
    dt_seconds: float = Field(
        ge=60.0, le=3600.0, default=1800.0, description="Simulation time step [s]"
    )
    tmy_year: int = Field(
        ge=2000, le=2030, default=2023, description="Representative TMY year (for labelling)"
    )


class SimulationEnqueued(BaseModel):
    """Returned immediately (HTTP 202) after a simulation is queued."""

    task_id: str
    status: Literal["queued"]
    status_url: str


class SimulationResult(BaseModel):
    """Simulation output returned when the task reaches the SUCCESS state."""

    plant_name: str
    n_modules: int
    p_installed_kw: float = Field(description="Total installed DC capacity [kW]")
    annual_yield_kwh: float = Field(description="Gross annual AC yield [kWh]")
    specific_yield_kwh_kw: float = Field(description="Specific yield [kWh/kWp]")
    performance_ratio: float = Field(description="PR = Y_final / Y_reference")
    peak_poa_wm2: float = Field(description="Peak plane-of-array irradiance [W m-2]")
    mean_cell_temp_c: float = Field(
        description="Mean cell temperature during irradiated hours [degC]"
    )
    simulation_steps: int
    dt_seconds: float


class SimulationStatus(BaseModel):
    """Returned by GET /api/v1/simulations/{task_id}."""

    task_id: str
    status: Literal["pending", "started", "success", "failure"]
    result: SimulationResult | None = None
    error: str | None = None
