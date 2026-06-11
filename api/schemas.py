"""
Pydantic v2 request and response schemas for the simulation API.
"""

from __future__ import annotations

from typing import Any, Literal

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


# ---------------------------------------------------------------------------
# Component catalogue schemas (UI dropdowns)
# ---------------------------------------------------------------------------


class ModuleInfo(BaseModel):
    """Lightweight module descriptor for the UI component selector."""

    id: str
    manufacturer: str
    model_name: str
    p_stc_w: float
    width_m: float
    length_m: float
    temp_coeff_p_pct_k: float
    eta_ref: float = Field(description="STC efficiency derived from P_stc / (1000 × area)")


class InverterInfo(BaseModel):
    """Inverter descriptor for the UI component selector."""

    id: str
    manufacturer: str
    model_name: str
    p_ac_max_kw: float
    p_dc_max_kw: float
    eta_max: float
    mppt_count: int
    v_dc_max_v: float


# ---------------------------------------------------------------------------
# Panel layout schemas
# ---------------------------------------------------------------------------


class PanelPosition(BaseModel):
    """Four WGS84 corners of a single panel rectangle, counter-clockwise from SW."""

    corners: list[tuple[float, float]]


class PanelLayoutRequest(BaseModel):
    """Request body for POST /api/v1/layout/panels."""

    roof_polygon: list[tuple[float, float]] = Field(
        description="Roof outline as [[lat, lon], …] — minimum 3 vertices"
    )
    module_id: str = Field(description="itl_identifier of the module from /components/modules")
    setback_m: float = Field(default=0.5, ge=0.0, le=5.0, description="Edge setback [m]")
    row_gap_m: float = Field(default=0.05, ge=0.0, le=2.0, description="Gap between rows [m]")
    col_gap_m: float = Field(default=0.02, ge=0.0, le=1.0, description="Gap between columns [m]")
    orientation: Literal["portrait", "landscape"] = "portrait"


class PanelLayoutResponse(BaseModel):
    """Response from POST /api/v1/layout/panels."""

    panel_count: int
    panels: list[PanelPosition]
    roof_area_m2: float
    usable_area_m2: float
    installed_kw: float
    centroid: tuple[float, float] = Field(description="[lat, lon] centroid of the roof polygon")


# ---------------------------------------------------------------------------
# Synchronous simulation (used by UI without Celery)
# ---------------------------------------------------------------------------


class SyncSimulationRequest(SimulationRequest):
    """Identical to SimulationRequest; used by the /run-sync endpoint."""

    extra_meta: dict[str, Any] | None = None
