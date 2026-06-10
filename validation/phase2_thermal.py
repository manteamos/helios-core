"""
Phase 2 validation: 3-node thermal solver vs Faiman/PVsyst model.

Outputs
-------
validation/output/phase2_steady_state.png  -- T_cell vs wind speed: 3-node vs Faiman
validation/output/phase2_convergence.png   -- dt convergence study (5 dt values)
validation/output/phase2_timeseries.png    -- 24-h transient response, Accra clearsky

Terminal output
---------------
Steady-state deviation table (3-node vs Faiman) at G=900 W/m^2, T_amb=25 degC.
Convergence table: T_cell at 12 h for 5 dt values (60 s ref down to 1800 s).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

from core.thermal.solver import faiman_cell_temp, solve_thermal, steady_state

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Steady-state sweep: 3-node vs Faiman at G=900, T_amb=25
# ---------------------------------------------------------------------------


def run_steady_state_comparison() -> dict[str, object]:
    v_range = np.linspace(0.0, 10.0, 41)
    tc_3node = np.empty_like(v_range)
    tg_3node = np.empty_like(v_range)
    tb_3node = np.empty_like(v_range)

    for i, v in enumerate(v_range):
        tg, tc, tb = steady_state(900.0, 25.0, float(v))
        tc_3node[i] = tc
        tg_3node[i] = tg
        tb_3node[i] = tb

    tc_faiman = faiman_cell_temp(900.0, 25.0, v_range)

    return {
        "v": v_range,
        "tc_3node": tc_3node,
        "tg_3node": tg_3node,
        "tb_3node": tb_3node,
        "tc_faiman": tc_faiman,
    }


def plot_steady_state(data: dict[str, object]) -> None:
    v = data["v"]
    tc_3n = data["tc_3node"]
    tg_3n = data["tg_3node"]
    tb_3n = data["tb_3node"]
    tc_f = data["tc_faiman"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(v, tc_3n, "b-", lw=2, label="T_cell (3-node)")
    ax.plot(v, tg_3n, "g--", lw=1.5, label="T_glass (3-node)")
    ax.plot(v, tb_3n, "r--", lw=1.5, label="T_back (3-node)")
    ax.plot(v, tc_f, "k:", lw=2, label="T_cell (Faiman ref)")
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Temperature [degC]")
    ax.set_title("Steady-state node temperatures (G=900 W/m^2, T_amb=25 degC)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    diff = np.asarray(tc_3n) - np.asarray(tc_f)
    ax2.plot(v, diff, "b-", lw=2)
    ax2.axhline(2.0, color="r", ls="--", lw=1, label="+2 degC limit")
    ax2.axhline(-2.0, color="r", ls="--", lw=1, label="-2 degC limit")
    ax2.fill_between(v, -2, 2, alpha=0.1, color="green", label="tolerance band")
    ax2.set_xlabel("Wind speed [m/s]")
    ax2.set_ylabel("T_cell(3-node) - T_cell(Faiman) [degC]")
    ax2.set_title("3-node vs Faiman deviation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase2_steady_state.png", dpi=150)
    plt.close(fig)


def print_steady_state_table(data: dict[str, object]) -> None:
    v_check = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]
    v_arr = np.asarray(data["v"])
    tc_3n = np.asarray(data["tc_3node"])
    tc_f = np.asarray(data["tc_faiman"])

    print("\n" + "=" * 60)
    print("PHASE 2 STEADY-STATE TABLE (G=900 W/m^2, T_amb=25 degC)")
    print("=" * 60)
    print(f"{'v [m/s]':>8}  {'Faiman [C]':>11}  {'3-node [C]':>11}  {'diff [C]':>10}")
    print("-" * 46)
    max_diff = 0.0
    for v in v_check:
        idx = int(np.argmin(np.abs(v_arr - v)))
        diff = float(tc_3n[idx] - tc_f[idx])
        max_diff = max(max_diff, abs(diff))
        print(f"{v:>8.1f}  {float(tc_f[idx]):>11.2f}  {float(tc_3n[idx]):>11.2f}  {diff:>+10.3f}")
    print("=" * 60)
    print(
        f"Max deviation: {max_diff:.3f} degC  (limit: 2.0 degC)  "
        f"{'PASS' if max_diff < 2.0 else 'FAIL'}"
    )


# ---------------------------------------------------------------------------
# 2. Convergence study: T_cell at steady state for 5 dt values
# ---------------------------------------------------------------------------


def run_convergence_study() -> dict[str, object]:
    dt_values = [30.0, 60.0, 120.0, 600.0, 1800.0]
    n_steps = 2000  # enough to reach steady state at each dt
    g_ss = 800.0
    ta_ss = 30.0
    ws_ss = 3.0

    tc_final = []
    subs_max = []

    for dt_val in dt_values:
        n = max(n_steps, int(3 * 3600 / dt_val))  # at least 3 h
        result = solve_thermal(
            np.full(n, g_ss),
            np.full(n, ta_ss),
            np.full(n, ws_ss),
            dt=dt_val,
        )
        tc_final.append(float(result["T_cell"][-1]))
        subs_max.append(int(np.max(result["substeps"])))

    return {
        "dt_values": dt_values,
        "tc_final": tc_final,
        "subs_max": subs_max,
        "g": g_ss,
        "t_amb": ta_ss,
        "ws": ws_ss,
    }


def plot_convergence(data: dict[str, object]) -> None:
    dt_vals = data["dt_values"]
    tc_vals = data["tc_final"]
    ref = float(tc_vals[0])  # finest dt is reference

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(dt_vals, tc_vals, "bo-", lw=2, ms=8)
    ax.axhline(
        ref, color="gray", ls="--", lw=1, label=f"Reference (dt={dt_vals[0]}s): {ref:.3f} degC"
    )
    ax.set_xlabel("Time step dt [s]")
    ax.set_ylabel("Steady-state T_cell [degC]")
    ax.set_title(
        f"dt convergence (G={data['g']} W/m^2, T_amb={data['t_amb']} degC, v={data['ws']} m/s)"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase2_convergence.png", dpi=150)
    plt.close(fig)


def print_convergence_table(data: dict[str, object]) -> None:
    ref = float(data["tc_final"][0])
    print("\n" + "=" * 60)
    print("PHASE 2 CONVERGENCE TABLE")
    print(f"  G={data['g']} W/m^2, T_amb={data['t_amb']} degC, v={data['ws']} m/s")
    print("=" * 60)
    print(f"{'dt [s]':>8}  {'T_cell [C]':>11}  {'delta vs 30s [mC]':>18}  {'max substeps':>13}")
    print("-" * 55)
    for dt_v, tc, nsub in zip(data["dt_values"], data["tc_final"], data["subs_max"], strict=False):
        delta_mc = (float(tc) - ref) * 1000.0
        print(f"{dt_v:>8.0f}  {float(tc):>11.4f}  {delta_mc:>+18.1f}  {nsub:>13d}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 3. 24-h transient response using Accra clearsky
# ---------------------------------------------------------------------------


def run_timeseries_validation() -> dict[str, object]:
    loc = Location(latitude=5.55, longitude=-0.22, altitude=61.0, tz="UTC")
    times = pd.date_range("2019-06-21", periods=24, freq="1h", tz="UTC")
    cs = loc.get_clearsky(times)
    solar_pos = loc.get_solarposition(times)

    zenith = solar_pos["apparent_zenith"].values
    ghi = cs["ghi"].values.astype(float)
    dhi = cs["dhi"].values.astype(float)
    dni = cs["dni"].values.astype(float)
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values.astype(float)

    # Simple POA at tilt=10, azimuth=180
    import core.transposition.perez_discrete as disc

    az = solar_pos["azimuth"].values
    am_raw = pvlib.atmosphere.get_relative_airmass(zenith, model="kastenyoung1989")
    am = np.where(np.isnan(am_raw), 0.0, np.asarray(am_raw, dtype=float))
    r_poa = disc.transpose(ghi, dni, dhi, zenith, az, 10.0, 180.0, dni_extra, am)
    g_poa = r_poa["poa_global"]

    t_amb = np.full(24, 28.0)  # constant for clear illustration
    ws = np.full(24, 3.0)  # constant 3 m/s

    result = solve_thermal(g_poa, t_amb, ws, dt=60.0)

    tc_faiman = faiman_cell_temp(g_poa, t_amb, ws)

    return {
        "hours": np.arange(24),
        "g_poa": g_poa,
        "t_cell": result["T_cell"],
        "t_glass": result["T_glass"],
        "t_back": result["T_back"],
        "tc_faiman": tc_faiman,
        "t_amb": t_amb,
    }


def plot_timeseries(data: dict[str, object]) -> None:
    hrs = data["hours"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(hrs, data["g_poa"], "orange", lw=2, label="G_POA [W/m^2]")
    ax1.set_ylabel("G_POA [W m^-2]")
    ax1.set_title("Phase 2: 24-h transient response — Accra clearsky, 21 June")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(hrs, data["t_cell"], "b-", lw=2, label="T_cell (3-node)")
    ax2.plot(hrs, data["t_glass"], "g--", lw=1.5, label="T_glass (3-node)")
    ax2.plot(hrs, data["t_back"], "r--", lw=1.5, label="T_back (3-node)")
    ax2.plot(hrs, np.asarray(data["tc_faiman"]), "k:", lw=2, label="T_cell (Faiman)")
    ax2.plot(hrs, data["t_amb"], "gray", lw=1, ls="-.", label="T_amb")
    ax2.set_xlabel("Hour of day [UTC]")
    ax2.set_ylabel("Temperature [degC]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase2_timeseries.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("Phase 2 validation — 3-node thermal solver ...")

    print("  Running steady-state sweep ...")
    ss_data = run_steady_state_comparison()
    plot_steady_state(ss_data)
    print_steady_state_table(ss_data)

    print("  Running convergence study ...")
    conv_data = run_convergence_study()
    plot_convergence(conv_data)
    print_convergence_table(conv_data)

    print("  Running 24-h transient simulation ...")
    ts_data = run_timeseries_validation()
    plot_timeseries(ts_data)

    print(f"\nPlots saved to {OUTPUT_DIR}/")
    print("  phase2_steady_state.png")
    print("  phase2_convergence.png")
    print("  phase2_timeseries.png")


if __name__ == "__main__":
    main()
