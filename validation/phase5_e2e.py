"""
Phase 5 end-to-end validation — annual simulation for a 45-inverter / 4 MWp plant.

Two modes
---------
Local mode (default):
    Runs the full simulation pipeline in-process.
    No Docker or Redis required.  Exercises the complete Phase 1 + Phase 2
    code path for a 45-inverter / 4 MWp plant with 30-min timesteps.

Docker mode (pass --docker flag):
    Builds and starts the full docker compose stack (api + worker + redis),
    submits the job via HTTP, polls until complete, verifies the result,
    then tears down.  Requires Docker and docker compose to be installed.

Usage
-----
    python validation/phase5_e2e.py           # local pipeline run
    python validation/phase5_e2e.py --docker  # full docker compose e2e
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from api.runner import run_annual_simulation
from api.schemas import SimulationRequest

# ---------------------------------------------------------------------------
# 45-inverter / 4 MWp plant definition
# ---------------------------------------------------------------------------
# 45 inverters × 222 modules × 400 W = 3,996 kWp ≈ 4 MWp
# Location: Accra, Ghana (5.55 N, -0.20 E)
# Tilt: 10° (near-flat; optimal for tropical latitudes)
# Azimuth: 180° (true South)

PLANT = SimulationRequest(
    plant_name="MKA-Solar-Accra-4MW",
    latitude_deg=5.55,
    longitude_deg=-0.20,
    n_inverters=45,
    n_modules_per_inverter=222,
    surface_tilt_deg=10.0,
    surface_azimuth_deg=180.0,
    module_p_stc_w=400.0,
    module_temp_coeff_pct_k=-0.38,
    module_eta_ref=0.20,
    albedo=0.25,
    dt_seconds=1800.0,  # 30-min sub-hourly
    tmy_year=2023,
)

_COMPOSE_FILE = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
_API_URL = "http://localhost:8000"


def _hr(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def _check_plausibility(result: object) -> list[str]:
    """Return list of failed checks (empty = all pass)."""
    failures = []
    from api.schemas import SimulationResult

    r: SimulationResult = result  # type: ignore[assignment]

    if not (0.5 < r.performance_ratio < 0.95):
        failures.append(f"PR={r.performance_ratio:.3f} outside [0.5, 0.95]")
    if not (900.0 < r.specific_yield_kwh_kw < 1800.0):
        failures.append(f"specific_yield={r.specific_yield_kwh_kw:.0f} outside [900, 1800] kWh/kWp")
    if r.peak_poa_wm2 < 500.0:
        failures.append(f"peak_poa={r.peak_poa_wm2:.0f} W/m2 seems too low")
    if r.mean_cell_temp_c < 30.0:
        failures.append(f"mean_cell_temp={r.mean_cell_temp_c:.1f} C seems too low")
    if r.annual_yield_kwh < 1e6:
        failures.append(f"annual_yield={r.annual_yield_kwh:.0f} kWh < 1 GWh for 4 MWp plant")
    return failures


# ---------------------------------------------------------------------------
# Local mode
# ---------------------------------------------------------------------------


def run_local() -> None:
    _hr("Phase 5 -- Local pipeline (no Docker required)")

    n_modules = PLANT.n_inverters * PLANT.n_modules_per_inverter
    p_cap_kwp = n_modules * PLANT.module_p_stc_w / 1000.0

    print(f"  Plant       : {PLANT.plant_name}")
    print(f"  Location    : {PLANT.latitude_deg} N, {PLANT.longitude_deg} E")
    print(f"  Inverters   : {PLANT.n_inverters}")
    print(f"  Modules     : {n_modules:,} x {PLANT.module_p_stc_w:.0f} W")
    print(f"  Installed   : {p_cap_kwp:.0f} kWp")
    n_annual = int(365 * 24 * 3600 / PLANT.dt_seconds)
    print(f"  Time step   : {PLANT.dt_seconds:.0f} s  ({n_annual:,} steps/year)")
    print("\n  Running simulation ...", end=" ", flush=True)

    t0 = time.perf_counter()
    result = run_annual_simulation(PLANT)
    elapsed = time.perf_counter() - t0

    print(f"done in {elapsed:.2f} s")

    _hr("Results")
    print(f"  Annual yield          : {result.annual_yield_kwh:>12,.0f} kWh")
    print(f"  Installed capacity    : {result.p_installed_kw:>12,.1f} kWp")
    print(f"  Specific yield        : {result.specific_yield_kwh_kw:>12,.1f} kWh/kWp")
    print(f"  Performance ratio     : {result.performance_ratio:>12.3f}")
    print(f"  Peak POA irradiance   : {result.peak_poa_wm2:>12.1f} W/m2")
    print(f"  Mean cell temperature : {result.mean_cell_temp_c:>12.1f} degC")
    print(f"  Simulation steps      : {result.simulation_steps:>12,}")

    _hr("Plausibility checks")
    failures = _check_plausibility(result)
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        sys.exit(1)
    else:
        print("  All plausibility checks PASS")


# ---------------------------------------------------------------------------
# Docker mode
# ---------------------------------------------------------------------------


def _docker_compose(*args: str) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    return subprocess.run(
        ["docker", "compose", "-f", str(_COMPOSE_FILE), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _wait_for_api(timeout_s: int = 90) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{_API_URL}/health", timeout=3) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError(f"API did not become healthy within {timeout_s} s")


def _post_simulation(payload: dict) -> str:  # type: ignore[type-arg]
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{_API_URL}/api/v1/simulations/run",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 202, f"Expected 202, got {resp.status}"
        return json.loads(resp.read())["task_id"]


def _poll_result(task_id: str, timeout_s: int = 300) -> dict:  # type: ignore[type-arg]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        url = f"{_API_URL}/api/v1/simulations/{task_id}"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())
        if data["status"] == "success":
            return data["result"]
        if data["status"] == "failure":
            raise RuntimeError(f"Task failed: {data.get('error')}")
        time.sleep(5)
    raise TimeoutError(f"Task {task_id} did not complete within {timeout_s} s")


def run_docker() -> None:
    _hr("Phase 5 -- Docker compose end-to-end test")
    print("  Building images ...", end=" ", flush=True)
    _docker_compose("build", "--quiet")
    print("done")

    print("  Starting stack  ...", end=" ", flush=True)
    _docker_compose("up", "-d")
    print("done")

    try:
        print("  Waiting for API health ...", end=" ", flush=True)
        _wait_for_api()
        print("healthy")

        payload = PLANT.model_dump()
        # Use hourly steps for the docker test to keep wall-clock time reasonable
        payload["dt_seconds"] = 3600.0

        print("  Submitting simulation ...", end=" ", flush=True)
        task_id = _post_simulation(payload)
        print(f"task_id={task_id}")

        print("  Polling for result  ...", end=" ", flush=True)
        result_dict = _poll_result(task_id)
        print("complete")

        _hr("Docker E2E Results")
        for k, v in result_dict.items():
            print(f"  {k:<30} : {v}")

        # Re-use the plausibility checks
        from api.schemas import SimulationResult

        result = SimulationResult.model_validate(result_dict)
        failures = _check_plausibility(result)

        _hr("Plausibility checks")
        if failures:
            for f in failures:
                print(f"  FAIL  {f}")
            sys.exit(1)
        else:
            print("  All plausibility checks PASS")

    finally:
        print("\n  Tearing down stack ...", end=" ", flush=True)
        _docker_compose("down", "--remove-orphans")
        print("done")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 e2e validation")
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run full docker compose stack end-to-end test",
    )
    args = parser.parse_args()

    if args.docker:
        run_docker()
    else:
        run_local()

    print("\nPhase 5 validation complete.\n")


if __name__ == "__main__":
    main()
