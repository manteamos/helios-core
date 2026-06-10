"""
3-Node Transient Thermal Solver for PV modules.

Physical model
--------------
Three lumped thermal nodes per module:

    Glass   (node g): front cover, faces sky + sun
    Cell    (node c): semiconductor layer, heat source
    Backsheet (node b): rear cover, faces ambient or rack

Energy balance for each node i (explicit forward-Euler):

    C_i * dT_i/dt = Q_in,i - SUM_j U_ij*(T_i-T_j) - h_i*A_i*(T_i-T_amb)

where:
    C_i = c_p,i * m_i        [J K^-1] -- thermal capacitance
    Q_in,i                   [W]      -- absorbed irradiance (cell only by default)
    U_ij                     [W K^-1] -- internal conductance between nodes i and j
    h_i                      [W m^-2 K^-1] -- external convective coefficient
    A_i                      [m^2]    -- exposed area of node i

Stability criterion (Fourier number, explicit FD):
    dt_max,i = C_i / (SUM_j U_ij + h_i*A_i)
    dt_stable = min_i(dt_max,i)

If the requested dt exceeds dt_stable, the solver sub-steps automatically.
A hard AssertionError is raised if the sub-step count would exceed MAX_SUBSTEPS
(defence against degenerate inputs rather than silent performance degradation).

Steady-state limit
------------------
Setting C_i -> 0 and dT/dt = 0 recovers the Faiman (2001) / PVsyst U-value model:

    T_c = T_amb + G_poa*(1-eta) / (U_c + U_v*v_wind)

where:
    U_c ~  25 W m^-2 K^-1   (constant heat-loss coefficient)
    U_v ~   6 W m^-2 K^-1 / (m/s)

The three-node steady-state matches this within 2 degC across 0-10 m/s
(validated in validation/phase2_thermal.py and tests).

Units
-----
Temperatures              : degC (or K -- differences are identical)
Irradiance (g_poa)        : W m^-2
Wind speed                : m s^-1
Wind angle from row normal: radians
Time step (dt)            : s
Conductances (u_ij, h*A)  : W K^-1
Capacitances              : J K^-1
Energy balance residual   : W m^-2 (per module area)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Hard safety limit on sub-stepping
# ---------------------------------------------------------------------------
MAX_SUBSTEPS: int = 10_000


# ---------------------------------------------------------------------------
# Default module physical parameters
# ---------------------------------------------------------------------------


class ModuleParams:
    """
    Physical parameters for a standard 60-cell glass-cell-backsheet module
    (approx. 1.64 m^2 aperture area, 18 kg total mass).

    All values are per-module (not per-m^2) except where noted.
    """

    # Geometry
    area: float = 1.64  # m^2 -- module aperture area

    # Thermal capacitances  C = c_p * mass  [J K^-1]
    C_glass: float = 7_000.0  # J K^-1  (soda-lime glass, ~3 mm, ~4 kg)
    C_cell: float = 1_200.0  # J K^-1  (silicon + encapsulant, ~0.8 kg)
    C_back: float = 400.0  # J K^-1  (PVF/TPT backsheet, ~0.2 kg)

    # Internal conductances U [W K^-1]  (node-pair, per module)
    # 500 W/K per bond ~ k_EVA/d_EVA * A = 0.35/0.00115 * 1.64 -- high enough
    # that external convection dominates heat loss; steady-state aligns with
    # Faiman within 1.25 degC at all wind speeds (see validation/phase2).
    U_gc: float = 500.0  # glass <-> cell  (EVA encapsulant bond)
    U_cb: float = 500.0  # cell  <-> backsheet

    # Absorptance / emittance
    # alpha_glass=0 lumps all optical absorption into the cell node; combined
    # with eta_ref this gives Q_c = (1-eta)*G*A, matching Faiman's heat source.
    alpha_glass: float = 0.0  # fraction of G_poa absorbed in glass (lumped to cell)
    alpha_cell: float = 1.0  # cell optical absorption (eta already removed via eta_ref)
    eta_ref: float = 0.20  # reference STC efficiency

    # Longwave radiative parameters (linearised around ~320 K)
    eps_front: float = 0.85  # emissivity, glass front surface
    eps_back: float = 0.85  # emissivity, backsheet rear surface
    sigma_lin: float = 4.0 * 5.67e-8 * 310.0**3  # ~6.76 W m^-2 K^-1 linearised


def _default_params() -> ModuleParams:
    return ModuleParams()


# ---------------------------------------------------------------------------
# Convection coefficients
# ---------------------------------------------------------------------------

# McAdams (1954) calibrated coefficients -- derived to match Faiman U_c=25 and
# U_v=6 [W m^-2 K^-1 / (m/s)] at head-on incidence, combined with
# h_rear=0.6*h_front and linearised longwave radiation (~11.5 W/(m^2 K) total):
#   h_front(v,0) + h_rear(v,0) = 1.6*(8.446 + 3.75*v) = 13.51 + 6*v
#   13.51 + 6*v + 11.49_rad = 25 + 6*v  => U_eff = Faiman (infinite U_gc/U_cb)
#   With finite U_gc=U_cb=500 W/K the residual is < 1.25 degC at all wind speeds.
_H_MC_FLOOR: float = 8.446  # free-convection + head-on stagnation [W m^-2 K^-1]
_H_MC_SLOPE: float = 3.750  # forced-convection wind-speed slope [W m^-2 K^-1 / (m/s)]


def h_front(
    wind_speed: float | FloatArray,
    wind_angle_rad: float | FloatArray = 0.0,
    exposure: float | FloatArray = 1.0,
) -> FloatArray:
    """
    Convective heat-transfer coefficient for the front (glass) surface.

    Head-on (wind_angle_rad=0) calibration: h = _H_MC_FLOOR + _H_MC_SLOPE*v,
    which together with h_rear=0.6*h_front and linearised radiation matches
    the Faiman model (U_c=25, U_v=6 W m^-2 K^-1) within 1.25 degC at all
    wind speeds 0-10 m/s.

    Angular reduction:  f = cos^2(angle) + 0.5*sin^2(angle)
        1.0 at head-on (angle=0), 0.5 at parallel flow (angle=pi/2).
    A floor of _H_MC_FLOOR is applied (free-convection lower bound).

    Parameters
    ----------
    wind_speed     : v [m s^-1], scalar or array
    wind_angle_rad : angle from row normal [rad]; 0 = perpendicular to panel
    exposure       : dimensionless rack-position shielding factor (0-1)

    Returns
    -------
    h [W m^-2 K^-1]
    """
    v = np.atleast_1d(np.asarray(wind_speed, dtype=np.float64))
    theta = np.atleast_1d(np.asarray(wind_angle_rad, dtype=np.float64))
    exp = np.atleast_1d(np.asarray(exposure, dtype=np.float64))

    cos2 = np.cos(theta) ** 2
    sin2 = np.sin(theta) ** 2
    # Angular factor: 1.0 head-on, 0.5 parallel; reduces forced convection only
    ang = cos2 + 0.5 * sin2
    h_forced: FloatArray = np.asarray((_H_MC_FLOOR + _H_MC_SLOPE * v) * ang, dtype=np.float64)
    h_floor: FloatArray = np.full_like(h_forced, _H_MC_FLOOR)
    return exp * np.maximum(h_forced, h_floor)


def h_rear(
    wind_speed: float | FloatArray,
    wind_angle_rad: float | FloatArray = 0.0,
    exposure: float | FloatArray = 1.0,
) -> FloatArray:
    """
    Convective heat-transfer coefficient for the rear (backsheet) surface.

    h_rear = 0.6 * h_front(v, angle, exposure)

    The 0.6 factor reflects rack shielding; consistent with Skoplaki & Palyvos
    (2009) review of rear-surface correlations for rack-mounted modules.

    Parameters
    ----------
    wind_speed, wind_angle_rad, exposure : same as h_front

    Returns
    -------
    h [W m^-2 K^-1]
    """
    return 0.6 * h_front(wind_speed, wind_angle_rad, exposure)


# ---------------------------------------------------------------------------
# Core explicit finite-difference step
# ---------------------------------------------------------------------------


def _stable_dt(
    cap_g: float,
    cap_c: float,
    cap_b: float,
    hfa: float,
    hra: float,
    u_gc: float,
    u_cb: float,
    rad_g: float,
    rad_b: float,
) -> float:
    """
    Maximum stable explicit-FD time step (minimum node time constant).

    dt_max,i = C_i / sum(conductances out of node i)

    Parameters: all conductance terms in W K^-1 (area already multiplied in).
    """
    sum_g = u_gc + hfa + rad_g  # glass: internal bond + front conv + front rad
    sum_c = u_gc + u_cb  # cell:  two internal bonds (no direct exterior)
    sum_b = u_cb + hra + rad_b  # back:  internal bond + rear conv + rear rad

    dt_g = cap_g / sum_g if sum_g > 0.0 else np.inf
    dt_c = cap_c / sum_c if sum_c > 0.0 else np.inf
    dt_b = cap_b / sum_b if sum_b > 0.0 else np.inf

    return float(min(dt_g, dt_c, dt_b))


def _fd_step(
    tg: float,
    tc: float,
    tb: float,
    t_amb: float,
    qg: float,
    qc: float,
    qb: float,
    hfa: float,
    hra: float,
    u_gc: float,
    u_cb: float,
    rad_g: float,
    rad_b: float,
    cap_g: float,
    cap_c: float,
    cap_b: float,
    dt: float,
) -> tuple[float, float, float]:
    """
    Single explicit forward-Euler step for the 3-node system.

    All conductances in W K^-1 (area already folded in).
    q terms are absorbed power [W] (positive = heat input to node).

    Returns (tg_new, tc_new, tb_new).
    """
    delta_g = dt / cap_g * (qg - u_gc * (tg - tc) - (hfa + rad_g) * (tg - t_amb))
    delta_c = dt / cap_c * (qc - u_gc * (tc - tg) - u_cb * (tc - tb))
    delta_b = dt / cap_b * (qb - u_cb * (tb - tc) - (hra + rad_b) * (tb - t_amb))
    return tg + delta_g, tc + delta_c, tb + delta_b


def _energy_balance_residual(
    tg: float,
    tc: float,
    tb: float,
    t_amb: float,
    qg: float,
    qc: float,
    qb: float,
    hfa: float,
    hra: float,
    u_gc: float,
    u_cb: float,
    rad_g: float,
    rad_b: float,
    area: float,
) -> float:
    """
    Energy balance residual [W m^-2] at a given thermal state.

    residual = (total power in - total power out) / module_area.
    Should be < 0.1 W m^-2 at true steady state.

    u_gc and u_cb are not used directly in the boundary-flux residual but are
    included in the signature for internal-consistency checks in tests.
    """
    _ = u_gc  # internal conductances cancel at steady state (no storage)
    _ = u_cb
    power_in = qg + qc + qb
    loss_front = (hfa + rad_g) * (tg - t_amb)
    loss_rear = (hra + rad_b) * (tb - t_amb)
    power_out = loss_front + loss_rear
    return float((power_in - power_out) / area)


# ---------------------------------------------------------------------------
# Public API: time-series solver
# ---------------------------------------------------------------------------


def solve_thermal(
    g_poa: FloatArray,
    t_amb: FloatArray,
    wind_speed: FloatArray,
    wind_angle_rad: FloatArray | None = None,
    exposure: float | FloatArray = 1.0,
    dt: float = 3600.0,
    params: ModuleParams | None = None,
    t_init: tuple[float, float, float] | None = None,
) -> dict[str, FloatArray]:
    """
    Simulate 3-node module temperatures over a time series.

    Uses explicit forward-Euler with automatic sub-stepping when the requested
    dt exceeds the stability limit. Raises AssertionError if sub-step count
    would exceed MAX_SUBSTEPS (degenerate input guard).

    Parameters
    ----------
    g_poa         : POA irradiance [W m^-2], shape (N,)
    t_amb         : Ambient temperature [degC], shape (N,)
    wind_speed    : Wind speed [m s^-1], shape (N,)
    wind_angle_rad: Wind angle from row normal [rad], shape (N,) or None (-> 0)
    exposure      : Rack-position exposure factor, scalar or shape (N,)
    dt            : Nominal time step [s] (default 3600 = 1 hour)
    params        : ModuleParams instance; uses defaults if None
    t_init        : Initial (tg, tc, tb) [degC]; defaults to (t_amb[0], ..., ...)

    Returns
    -------
    dict with keys:
        T_glass     : Glass temperature [degC], shape (N,)
        T_cell      : Cell temperature [degC], shape (N,)
        T_back      : Backsheet temperature [degC], shape (N,)
        substeps    : Sub-steps used per time step [float64], shape (N,)
        residual_Wm2: Energy balance residual [W m^-2] at end of each step, shape (N,)
    """
    p = params if params is not None else _default_params()

    n = len(g_poa)
    g = np.atleast_1d(np.asarray(g_poa, dtype=np.float64))
    ta = np.atleast_1d(np.asarray(t_amb, dtype=np.float64))
    ws = np.atleast_1d(np.asarray(wind_speed, dtype=np.float64))
    wa = (
        np.zeros(n, dtype=np.float64)
        if wind_angle_rad is None
        else np.atleast_1d(np.asarray(wind_angle_rad, dtype=np.float64))
    )
    exp_arr = np.broadcast_to(np.asarray(exposure, dtype=np.float64), (n,)).copy()

    # Output arrays
    tg_out: FloatArray = np.empty(n, dtype=np.float64)
    tc_out: FloatArray = np.empty(n, dtype=np.float64)
    tb_out: FloatArray = np.empty(n, dtype=np.float64)
    subs_out: FloatArray = np.empty(n, dtype=np.float64)
    res_out: FloatArray = np.empty(n, dtype=np.float64)

    # Initial conditions
    if t_init is not None:
        tg, tc, tb = float(t_init[0]), float(t_init[1]), float(t_init[2])
    else:
        tg = tc = tb = float(ta[0])

    for i in range(n):
        g_i = float(g[i])
        ta_i = float(ta[i])
        ws_i = float(ws[i])
        wa_i = float(wa[i])
        exp_i = float(exp_arr[i])

        # Absorbed power per node [W]  (area-weighted)
        mod_a = p.area
        qg_i = p.alpha_glass * g_i * mod_a
        qc_i = p.alpha_cell * g_i * mod_a * (1.0 - p.eta_ref)
        qb_i = 0.0

        # Convection terms [W K^-1]
        hf = float(h_front(ws_i, wa_i, exp_i)[0])
        hr = float(h_rear(ws_i, wa_i, exp_i)[0])
        hfa_i = hf * mod_a
        hra_i = hr * mod_a

        # Linearised radiation [W K^-1]
        rad_g = p.eps_front * p.sigma_lin * mod_a
        rad_b = p.eps_back * p.sigma_lin * mod_a

        # Stability check
        dt_stab = _stable_dt(
            p.C_glass,
            p.C_cell,
            p.C_back,
            hfa_i,
            hra_i,
            p.U_gc,
            p.U_cb,
            rad_g,
            rad_b,
        )
        n_sub = int(np.ceil(dt / dt_stab)) if dt > dt_stab else 1
        assert n_sub <= MAX_SUBSTEPS, (
            f"Step {i}: required sub-steps {n_sub} exceeds MAX_SUBSTEPS={MAX_SUBSTEPS}. "
            f"Inputs: dt={dt}, dt_stable={dt_stab:.4f}, ws={ws_i}, g={g_i}. "
            f"Check for degenerate (near-zero capacitance) parameters."
        )

        dt_sub = dt / n_sub
        for _ in range(n_sub):
            tg, tc, tb = _fd_step(
                tg,
                tc,
                tb,
                ta_i,
                qg_i,
                qc_i,
                qb_i,
                hfa_i,
                hra_i,
                p.U_gc,
                p.U_cb,
                rad_g,
                rad_b,
                p.C_glass,
                p.C_cell,
                p.C_back,
                dt_sub,
            )

        tg_out[i] = tg
        tc_out[i] = tc
        tb_out[i] = tb
        subs_out[i] = float(n_sub)
        res_out[i] = _energy_balance_residual(
            tg,
            tc,
            tb,
            ta_i,
            qg_i,
            qc_i,
            qb_i,
            hfa_i,
            hra_i,
            p.U_gc,
            p.U_cb,
            rad_g,
            rad_b,
            mod_a,
        )

    return {
        "T_glass": tg_out,
        "T_cell": tc_out,
        "T_back": tb_out,
        "substeps": subs_out,
        "residual_Wm2": res_out,
    }


# ---------------------------------------------------------------------------
# Steady-state helper (for validation and testing)
# ---------------------------------------------------------------------------


def steady_state(
    g_poa: float,
    t_amb: float,
    wind_speed: float,
    wind_angle_rad: float = 0.0,
    exposure: float = 1.0,
    params: ModuleParams | None = None,
    tol: float = 1e-6,
    max_iter: int = 100_000,
) -> tuple[float, float, float]:
    """
    Find steady-state (T_glass, T_cell, T_back) by running until convergence.

    Uses a small synthetic dt (60 s) with sub-stepping; iterates until
    max node temperature change per step < tol [degC].

    Returns (T_glass, T_cell, T_back) at steady state [degC].
    """
    p = params if params is not None else _default_params()
    tg = tc = tb = t_amb

    mod_a = p.area
    qg = p.alpha_glass * g_poa * mod_a
    qc = p.alpha_cell * g_poa * mod_a * (1.0 - p.eta_ref)
    qb = 0.0

    hfa = float(h_front(wind_speed, wind_angle_rad, exposure)[0]) * mod_a
    hra = float(h_rear(wind_speed, wind_angle_rad, exposure)[0]) * mod_a
    rad_g = p.eps_front * p.sigma_lin * mod_a
    rad_b = p.eps_back * p.sigma_lin * mod_a

    dt_stab = _stable_dt(p.C_glass, p.C_cell, p.C_back, hfa, hra, p.U_gc, p.U_cb, rad_g, rad_b)
    dt_ss = min(60.0, dt_stab * 0.9)

    for _ in range(max_iter):
        tg_n, tc_n, tb_n = _fd_step(
            tg,
            tc,
            tb,
            t_amb,
            qg,
            qc,
            qb,
            hfa,
            hra,
            p.U_gc,
            p.U_cb,
            rad_g,
            rad_b,
            p.C_glass,
            p.C_cell,
            p.C_back,
            dt_ss,
        )
        delta = max(abs(tg_n - tg), abs(tc_n - tc), abs(tb_n - tb))
        tg, tc, tb = tg_n, tc_n, tb_n
        if delta < tol:
            break

    return float(tg), float(tc), float(tb)


def faiman_cell_temp(
    g_poa: float | FloatArray,
    t_amb: float | FloatArray,
    wind_speed: float | FloatArray,
    u_c: float = 25.0,
    u_v: float = 6.0,
    eta: float = 0.20,
) -> FloatArray:
    """
    Faiman (2001) / PVsyst cell temperature model.

    T_c = T_amb + G_poa*(1-eta) / (u_c + u_v*v)

    Used as the validation reference for steady-state comparison.

    Parameters
    ----------
    g_poa      : POA irradiance [W m^-2]
    t_amb      : Ambient temperature [degC]
    wind_speed : [m s^-1]
    u_c        : Constant loss coefficient [W m^-2 K^-1] (default 25)
    u_v        : Wind-speed coefficient [W m^-2 K^-1 / (m/s)] (default 6)
    eta        : Module efficiency (default 0.20)

    Returns
    -------
    T_cell [degC]
    """
    g = np.atleast_1d(np.asarray(g_poa, dtype=np.float64))
    ta = np.atleast_1d(np.asarray(t_amb, dtype=np.float64))
    v = np.atleast_1d(np.asarray(wind_speed, dtype=np.float64))
    result: FloatArray = ta + g * (1.0 - eta) / (u_c + u_v * v)
    return result
