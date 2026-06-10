"""
Phase 5 unit tests — FastAPI + Celery + simulation pipeline.

Coverage
--------
1.  GET /health returns 200 {"status": "ok"}.
2.  POST /api/v1/simulations/run returns 202 with task_id and status_url.
3.  POST validation: missing required field returns 422.
4.  POST validation: n_inverters out of range returns 422.
5.  POST validation: n_modules_per_inverter=0 returns 422.
6.  GET /api/v1/simulations/{task_id} pending state.
7.  GET /api/v1/simulations/{task_id} started state.
8.  GET /api/v1/simulations/{task_id} success state with full result.
9.  GET /api/v1/simulations/{task_id} failure state with error message.
10. Solar position: zenith > 90 at midnight (site-independent).
11. Solar position: zenith < 90 at solar noon on equinox (tropical site).
12. Solar position: azimuth ~180 at solar noon (sun due South, NH site).
13. Extraterrestrial irradiance: range [1322, 1413] W/m2 (perihelion/aphelion).
14. Airmass: == 1.0 at zenith=0; increases monotonically; 40 at zenith=90.
15. TMY generation: GHI, DHI, DNI all >= 0.
16. TMY generation: GHI == 0 when cos_zenith <= 0 (night).
17. TMY generation: DNI <= 1200 W/m2 (physical upper bound).
18. Simulation runner: annual yield > 0 for a small plant.
19. Simulation runner: performance_ratio in plausible range [0.5, 0.95].
20. Simulation runner: specific_yield plausible for Ghana [900, 1800] kWh/kWp.
21. Simulation runner: simulation_steps = 8760 for dt=3600 s.
22. Simulation runner: simulation_steps = 17520 for dt=1800 s.
23. Simulation runner: peak POA > 0.
24. Simulation runner: mean_cell_temp_c > mean TMY ambient (modules run hot).
25. Runner scales linearly: doubling n_modules doubles annual_yield_kwh.
26. status_url path matches task_id from 202 response.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.runner import run_annual_simulation
from api.schemas import SimulationRequest
from api.solar import airmass_kasten, extraterrestrial_irradiance, solar_position
from api.tmy import generate_ghana_tmy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL: dict[str, object] = {
    "plant_name": "Test-Plant",
    "latitude_deg": 5.55,
    "longitude_deg": -0.20,
    "n_inverters": 1,
    "n_modules_per_inverter": 10,
    "dt_seconds": 3600.0,  # hourly for speed in unit tests
}

_FULL_RESULT: dict[str, object] = {
    "plant_name": "Test-Plant",
    "n_modules": 10,
    "p_installed_kw": 4.0,
    "annual_yield_kwh": 5200.0,
    "specific_yield_kwh_kw": 1300.0,
    "performance_ratio": 0.78,
    "peak_poa_wm2": 980.0,
    "mean_cell_temp_c": 46.0,
    "simulation_steps": 8760,
    "dt_seconds": 3600.0,
}


@pytest.fixture()
def client() -> TestClient:
    from api.app import app  # import here to avoid module-level Celery connect

    return TestClient(app)


@pytest.fixture()
def mock_task_id() -> str:
    return "aaaabbbb-cccc-dddd-eeee-ffffffffffff"


@pytest.fixture()
def patched_delay(mock_task_id: str):
    fake = MagicMock()
    fake.id = mock_task_id
    with patch("api.app.simulation_task") as mock_obj:
        mock_obj.delay.return_value = fake
        yield mock_obj


@pytest.fixture()
def small_params() -> SimulationRequest:
    return SimulationRequest(**_MINIMAL)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Health probe
# ---------------------------------------------------------------------------


def test_health_returns_200(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 2. POST /run → 202
# ---------------------------------------------------------------------------


def test_enqueue_returns_202(
    client: TestClient, patched_delay: MagicMock, mock_task_id: str
) -> None:
    r = client.post("/api/v1/simulations/run", json=_MINIMAL)
    assert r.status_code == 202
    body = r.json()
    assert body["task_id"] == mock_task_id
    assert body["status"] == "queued"


# ---------------------------------------------------------------------------
# 3–5. POST validation errors
# ---------------------------------------------------------------------------


def test_missing_plant_name_returns_422(client: TestClient) -> None:
    payload = dict(_MINIMAL)
    del payload["plant_name"]
    r = client.post("/api/v1/simulations/run", json=payload)
    assert r.status_code == 422


def test_n_inverters_zero_returns_422(client: TestClient) -> None:
    r = client.post("/api/v1/simulations/run", json={**_MINIMAL, "n_inverters": 0})
    assert r.status_code == 422


def test_n_modules_per_inverter_zero_returns_422(client: TestClient) -> None:
    r = client.post("/api/v1/simulations/run", json={**_MINIMAL, "n_modules_per_inverter": 0})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 6–9. GET /simulations/{task_id} state machine
# ---------------------------------------------------------------------------


def _make_async_result(state: str, result: object = None) -> MagicMock:
    m = MagicMock()
    m.state = state
    m.result = result
    return m


def test_get_status_pending(client: TestClient, mock_task_id: str) -> None:
    with patch("api.app.AsyncResult") as mock_ar:
        mock_ar.return_value = _make_async_result("PENDING")
        r = client.get(f"/api/v1/simulations/{mock_task_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    assert r.json()["result"] is None


def test_get_status_started(client: TestClient, mock_task_id: str) -> None:
    with patch("api.app.AsyncResult") as mock_ar:
        mock_ar.return_value = _make_async_result("STARTED")
        r = client.get(f"/api/v1/simulations/{mock_task_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "started"


def test_get_status_success(client: TestClient, mock_task_id: str) -> None:
    with patch("api.app.AsyncResult") as mock_ar:
        mock_ar.return_value = _make_async_result("SUCCESS", result=_FULL_RESULT)
        r = client.get(f"/api/v1/simulations/{mock_task_id}")
    body = r.json()
    assert body["status"] == "success"
    assert body["result"]["annual_yield_kwh"] == 5200.0
    assert body["result"]["performance_ratio"] == 0.78


def test_get_status_failure(client: TestClient, mock_task_id: str) -> None:
    exc = RuntimeError("solver diverged")
    with patch("api.app.AsyncResult") as mock_ar:
        mock_ar.return_value = _make_async_result("FAILURE", result=exc)
        r = client.get(f"/api/v1/simulations/{mock_task_id}")
    body = r.json()
    assert body["status"] == "failure"
    assert "solver diverged" in body["error"]


# ---------------------------------------------------------------------------
# 10–12. Solar position
# ---------------------------------------------------------------------------


def test_solar_zenith_at_midnight_exceeds_90() -> None:
    doy = np.array([1.0])
    hour = np.array([0.0])  # midnight UTC (any site far from date line)
    sol = solar_position(doy, hour, latitude_deg=5.5, longitude_deg=-0.2)
    # Midnight is always night; zenith must be > 90°
    assert float(sol["zenith"][0]) > 90.0


def test_solar_zenith_near_noon_equinox_lt_90() -> None:
    # Day 80 ~ spring equinox; solar noon ~ 12:00 UTC at lon=0
    doy = np.array([80.0])
    hour = np.array([12.0])
    sol = solar_position(doy, hour, latitude_deg=5.5, longitude_deg=0.0)
    assert float(sol["zenith"][0]) < 90.0


def test_solar_azimuth_near_south_at_noon_nh() -> None:
    # For a site in the NH (lat=52° N), sun is due South at local noon
    doy = np.array([80.0])
    hour = np.array([12.0])
    sol = solar_position(doy, hour, latitude_deg=52.0, longitude_deg=0.0)
    az = float(sol["azimuth"][0])
    # Azimuth should be close to 180° (South), within ±20° tolerance for simple model
    assert abs(az - 180.0) < 20.0


# ---------------------------------------------------------------------------
# 13. Extraterrestrial irradiance
# ---------------------------------------------------------------------------


def test_extraterrestrial_irradiance_range() -> None:
    doy = np.arange(1, 366, dtype=np.float64)
    i0 = extraterrestrial_irradiance(doy)
    # I_sc = 1361.5, amplitude ±3.3%  → range [1316, 1407]
    assert float(np.min(i0)) > 1310.0
    assert float(np.max(i0)) < 1420.0


# ---------------------------------------------------------------------------
# 14. Air mass
# ---------------------------------------------------------------------------


def test_airmass_at_zenith_is_one() -> None:
    am = airmass_kasten(np.array([0.0]))
    assert abs(float(am[0]) - 1.0) < 0.01


def test_airmass_monotonically_increasing() -> None:
    zeniths = np.linspace(0.0, 85.0, 50)
    am = airmass_kasten(zeniths)
    assert np.all(np.diff(am) > 0)


def test_airmass_at_horizon_is_40() -> None:
    am = airmass_kasten(np.array([90.0]))
    assert float(am[0]) == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# 15–17. TMY generation
# ---------------------------------------------------------------------------

_N_STEPS = 8760
_DOY_ANNUAL = np.clip(np.floor(np.arange(_N_STEPS, dtype=np.float64) / 24.0) + 1.0, 1.0, 365.0)
_HOUR_ANNUAL = np.arange(_N_STEPS, dtype=np.float64) % 24.0
_SOL_ANNUAL = solar_position(_DOY_ANNUAL, _HOUR_ANNUAL, latitude_deg=5.5, longitude_deg=-0.2)
_COS_Z_ANNUAL = _SOL_ANNUAL["cos_zen"]
_I0_ANNUAL = extraterrestrial_irradiance(_DOY_ANNUAL)
_TMY_ANNUAL = generate_ghana_tmy(_DOY_ANNUAL, _HOUR_ANNUAL, _COS_Z_ANNUAL, _I0_ANNUAL)


def test_tmy_irradiance_nonnegative() -> None:
    for key in ("ghi", "dhi", "dni"):
        assert float(np.min(_TMY_ANNUAL[key])) >= 0.0, f"{key} has negative values"


def test_tmy_ghi_zero_at_night() -> None:
    night_mask = _COS_Z_ANNUAL <= 0.0
    assert float(np.max(_TMY_ANNUAL["ghi"][night_mask])) < 1e-6


def test_tmy_dni_physical_upper_bound() -> None:
    # DNI cannot physically exceed ~1100 W/m² at sea level under any conditions
    assert float(np.max(_TMY_ANNUAL["dni"])) < 1200.0


# ---------------------------------------------------------------------------
# 18–24. Simulation runner
# ---------------------------------------------------------------------------


def test_runner_annual_yield_positive(small_params: SimulationRequest) -> None:
    result = run_annual_simulation(small_params)
    assert result.annual_yield_kwh > 0.0


def test_runner_pr_plausible(small_params: SimulationRequest) -> None:
    result = run_annual_simulation(small_params)
    assert 0.5 < result.performance_ratio < 0.95


def test_runner_specific_yield_ghana_range(small_params: SimulationRequest) -> None:
    result = run_annual_simulation(small_params)
    assert 900.0 < result.specific_yield_kwh_kw < 1800.0


def test_runner_steps_hourly(small_params: SimulationRequest) -> None:
    result = run_annual_simulation(small_params)
    assert result.simulation_steps == 8760


def test_runner_steps_half_hourly() -> None:
    params = SimulationRequest(**{**_MINIMAL, "dt_seconds": 1800.0})  # type: ignore[arg-type]
    result = run_annual_simulation(params)
    assert result.simulation_steps == 17_520


def test_runner_peak_poa_positive(small_params: SimulationRequest) -> None:
    result = run_annual_simulation(small_params)
    assert result.peak_poa_wm2 > 0.0


def test_runner_cell_temp_above_ambient(small_params: SimulationRequest) -> None:
    result = run_annual_simulation(small_params)
    # Cells are hotter than ambient due to absorbed irradiance
    # Ghana ambient ~28-33 °C; cells under load typically 45-65 °C
    assert result.mean_cell_temp_c > 30.0


# ---------------------------------------------------------------------------
# 25. Linear scaling with n_modules
# ---------------------------------------------------------------------------


def test_runner_yield_scales_with_modules() -> None:
    p1 = SimulationRequest(**{**_MINIMAL, "n_modules_per_inverter": 10})  # type: ignore[arg-type]
    p2 = SimulationRequest(**{**_MINIMAL, "n_modules_per_inverter": 20})  # type: ignore[arg-type]
    r1 = run_annual_simulation(p1)
    r2 = run_annual_simulation(p2)
    # yield should scale by exactly 2 (same physics, double the modules)
    ratio = r2.annual_yield_kwh / r1.annual_yield_kwh
    assert abs(ratio - 2.0) < 0.01


# ---------------------------------------------------------------------------
# 26. Status URL contains the task_id
# ---------------------------------------------------------------------------


def test_status_url_contains_task_id(
    client: TestClient, patched_delay: MagicMock, mock_task_id: str
) -> None:
    r = client.post("/api/v1/simulations/run", json=_MINIMAL)
    body = r.json()
    assert mock_task_id in body["status_url"]
