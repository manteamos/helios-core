"""
Phase 2 unit tests -- 3-node transient thermal solver.

Coverage
--------
1. Steady-state cell temperature within 2 degC of Faiman model across 0-10 m/s.
2. Energy balance residual < 0.1 W m^-2 at steady state.
3. Stability assertion fires for degenerate (near-zero capacitance) inputs.
4. Sub-stepping activates when dt > dt_stable, and produces correct results.
5. Time-series solve runs on 8760 rows without NaN / assertion failure.
6. h_front / h_rear physical sanity (increases with wind, h_rear < h_front).
7. Convergence: smaller sub-steps give the same steady state.
8. Zero-irradiance limit: module cools to ambient.
9. _stable_dt returns a positive finite value for standard params.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.thermal.solver import (
    ModuleParams,
    _stable_dt,
    faiman_cell_temp,
    h_front,
    h_rear,
    solve_thermal,
    steady_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(**overrides: float) -> ModuleParams:
    p = ModuleParams()
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


# ---------------------------------------------------------------------------
# 1. Steady-state vs Faiman within 2 degC (0-10 m/s sweep)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("v_wind", [0.0, 1.0, 2.0, 5.0, 8.0, 10.0])
def test_steady_state_vs_faiman_within_2c(v_wind: float) -> None:
    g_poa = 900.0
    t_amb = 25.0
    _, tc_3node, _ = steady_state(g_poa, t_amb, v_wind)
    tc_faiman = float(faiman_cell_temp(g_poa, t_amb, v_wind)[0])
    diff = abs(tc_3node - tc_faiman)
    assert diff < 2.0, (
        f"v={v_wind} m/s: 3-node T_c={tc_3node:.2f} degC, "
        f"Faiman={tc_faiman:.2f} degC, diff={diff:.3f} degC > 2 degC limit"
    )


# ---------------------------------------------------------------------------
# 2. Energy balance residual < 0.1 W m^-2 at steady state
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("v_wind", [0.0, 3.0, 10.0])
def test_energy_balance_at_steady_state(v_wind: float) -> None:
    g_poa = 800.0
    t_amb = 30.0
    n = 3000
    result = solve_thermal(
        g_poa=np.full(n, g_poa),
        t_amb=np.full(n, t_amb),
        wind_speed=np.full(n, v_wind),
        dt=60.0,
    )
    residual_last = abs(float(result["residual_Wm2"][-1]))
    assert (
        residual_last < 0.1
    ), f"v={v_wind}: energy balance residual={residual_last:.4f} W/m^2 > 0.1 limit"


# ---------------------------------------------------------------------------
# 3. Stability assertion fires for degenerate params
# ---------------------------------------------------------------------------
def test_stability_assertion_fires_for_tiny_capacitance() -> None:
    p = _make_params(C_glass=1e-10, C_cell=1e-10, C_back=1e-10)
    with pytest.raises(AssertionError, match="MAX_SUBSTEPS"):
        solve_thermal(
            g_poa=np.array([800.0]),
            t_amb=np.array([25.0]),
            wind_speed=np.array([3.0]),
            dt=3600.0,
            params=p,
        )


# ---------------------------------------------------------------------------
# 4. Sub-stepping activates and produces consistent results
# ---------------------------------------------------------------------------
def test_substepping_activates_and_matches_fine_dt() -> None:
    """Coarse dt (3600 s) with sub-stepping should match fine dt (60 s) closely."""
    g_arr = np.full(24, 700.0)
    ta_arr = np.full(24, 28.0)
    ws_arr = np.full(24, 4.0)

    r_coarse = solve_thermal(g_arr, ta_arr, ws_arr, dt=3600.0)
    r_fine = solve_thermal(g_arr, ta_arr, ws_arr, dt=60.0)

    assert np.any(r_coarse["substeps"] > 1), "Expected sub-stepping at dt=3600 s"

    diff = abs(float(r_coarse["T_cell"][-1]) - float(r_fine["T_cell"][-1]))
    assert diff < 0.5, f"Coarse vs fine dt T_cell diff = {diff:.3f} degC > 0.5 degC"


# ---------------------------------------------------------------------------
# 5. Full 8760-row time-series runs clean
# ---------------------------------------------------------------------------
def test_8760_row_no_nan() -> None:
    rng = np.random.default_rng(42)
    n = 8760
    g = np.abs(rng.normal(400.0, 300.0, n)).clip(0.0, 1200.0)
    ta = rng.uniform(15.0, 40.0, n)
    ws = rng.uniform(0.0, 10.0, n)
    # dt=60 s: 78 sub-steps/step vs 4660 at dt=3600 s; verifies NaN-free shape.
    result = solve_thermal(g, ta, ws, dt=60.0)
    for key in ("T_glass", "T_cell", "T_back"):
        assert not np.any(np.isnan(result[key])), f"NaN in {key}"
    assert result["T_cell"].shape == (n,)


# ---------------------------------------------------------------------------
# 6. h_front / h_rear sanity
# ---------------------------------------------------------------------------
def test_h_front_increases_with_wind() -> None:
    v = np.array([0.0, 1.0, 5.0, 10.0])
    h = h_front(v)
    assert np.all(np.diff(h) > 0), "h_front must be strictly increasing with wind speed"


def test_h_rear_less_than_h_front() -> None:
    v = np.array([0.0, 2.0, 5.0, 10.0])
    assert np.all(h_rear(v) < h_front(v)), "h_rear must be < h_front at all wind speeds"


def test_h_front_free_convection_floor() -> None:
    h0 = float(h_front(0.0)[0])
    assert h0 >= 5.0, f"h_front at v=0 should be >= 5 W/(m^2 K) free-conv floor, got {h0}"


@pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2])
def test_h_front_angular_dependence(angle: float) -> None:
    h_headon = float(h_front(5.0, 0.0)[0])
    h_oblique = float(h_front(5.0, angle)[0])
    if angle > 0:
        angle_deg = float(np.degrees(angle))
        assert h_headon >= h_oblique, (
            f"h_front head-on ({h_headon:.2f}) should be >= oblique "
            f"at {angle_deg:.0f} deg ({h_oblique:.2f})"
        )


# ---------------------------------------------------------------------------
# 7. Convergence: smaller sub-step gives same steady state
# ---------------------------------------------------------------------------
def test_steady_state_independent_of_substep_size() -> None:
    g_poa, t_amb_val, v_wind = 800.0, 25.0, 3.0
    _, tc_ss, _ = steady_state(g_poa, t_amb_val, v_wind)

    n = 5000
    r_fine = solve_thermal(np.full(n, g_poa), np.full(n, t_amb_val), np.full(n, v_wind), dt=30.0)
    tc_fine = float(r_fine["T_cell"][-1])
    assert (
        abs(tc_ss - tc_fine) < 0.05
    ), f"steady_state={tc_ss:.3f} vs fine-dt={tc_fine:.3f} degC, diff > 0.05"


# ---------------------------------------------------------------------------
# 8. Zero irradiance: module cools to ambient
# ---------------------------------------------------------------------------
def test_zero_irradiance_cools_to_ambient() -> None:
    n = 2000
    t_target = 20.0
    result = solve_thermal(
        g_poa=np.zeros(n),
        t_amb=np.full(n, t_target),
        wind_speed=np.full(n, 3.0),
        dt=60.0,
        t_init=(60.0, 65.0, 58.0),
    )
    final_tc = float(result["T_cell"][-1])
    assert abs(final_tc - t_target) < 0.5, (
        f"Module should cool to ambient ({t_target} degC) under zero irradiance, "
        f"got T_cell={final_tc:.2f} degC"
    )


# ---------------------------------------------------------------------------
# 9. _stable_dt: positive and finite for typical params
# ---------------------------------------------------------------------------
def test_stable_dt_positive_and_finite() -> None:
    p = ModuleParams()
    area = p.area
    hfa = float(h_front(3.0)[0]) * area
    hra = float(h_rear(3.0)[0]) * area
    rad_g = p.eps_front * p.sigma_lin * area
    rad_b = p.eps_back * p.sigma_lin * area
    dt_s = _stable_dt(p.C_glass, p.C_cell, p.C_back, hfa, hra, p.U_gc, p.U_cb, rad_g, rad_b)
    assert 0.0 < dt_s < np.inf, f"dt_stable = {dt_s} is not a positive finite number"
