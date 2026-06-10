"""
Phase 3 validation: stochastic mismatch + 25-year Arrhenius degradation.

Outputs
-------
validation/output/phase3_mismatch_year1.png   -- year-1 mismatch distribution
validation/output/phase3_degradation_25y.png  -- degradation trajectories (corner vs center)
validation/output/phase3_dose_map.png         -- per-module annual Arrhenius dose heatmap

Terminal output
---------------
Year-0 vs year-1 mismatch loss table (mean ± std over modules).
25-year degradation summary: year-25 mean factor, corner vs center delta.
Rack exposure map (4×6 rack).
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

from core.mismatch.aging import annual_dose_factor
from core.mismatch.cell_matrix import (
    rack_exposure_map,
    simulate_mismatch_25years,
)
from core.thermal.solver import solve_thermal

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_RACK_ROWS = 4
N_RACK_COLS = 6
N_MODULES = N_RACK_ROWS * N_RACK_COLS  # 24


# ---------------------------------------------------------------------------
# Build per-module T_c history using Phase 2 thermal solver + exposure map
# ---------------------------------------------------------------------------


def build_tc_history() -> np.ndarray:
    """
    Run the Phase 2 thermal solver for each module position in the rack.

    Returns
    -------
    (8760, N_MODULES), float64  — hourly cell temperature in Kelvin
    """
    loc = Location(latitude=5.55, longitude=-0.22, altitude=61.0, tz="UTC")
    times = pd.date_range("2019-01-01", periods=8760, freq="1h", tz="UTC")
    cs = loc.get_clearsky(times, model="ineichen")
    solar_pos = loc.get_solarposition(times)

    zenith = solar_pos["apparent_zenith"].values
    ghi = cs["ghi"].values.astype(float)
    dhi = cs["dhi"].values.astype(float)
    dni = cs["dni"].values.astype(float)
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values.astype(float)

    # Simple POA using discrete Perez
    import core.transposition.perez_discrete as disc

    az = solar_pos["azimuth"].values
    am_raw = pvlib.atmosphere.get_relative_airmass(zenith, model="kastenyoung1989")
    am = np.where(np.isnan(am_raw), 0.0, np.asarray(am_raw, dtype=float))
    g_poa = disc.transpose(ghi, dni, dhi, zenith, az, 10.0, 180.0, dni_extra, am)["poa_global"]

    t_amb = np.full(8760, 28.0)
    ws = np.full(8760, 3.0)

    exposure_map = rack_exposure_map(N_RACK_ROWS, N_RACK_COLS)  # (24,)

    tc_kelvin = np.empty((8760, N_MODULES), dtype=np.float64)
    for m in range(N_MODULES):
        result = solve_thermal(g_poa, t_amb, ws, dt=3600.0, exposure=float(exposure_map[m]))
        tc_kelvin[:, m] = np.asarray(result["T_cell"]) + 273.15

    return tc_kelvin


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_year1_mismatch(result: dict[str, np.ndarray]) -> None:
    loss_y0 = result["mismatch_loss"][0] * 100.0  # year 0 (manufacturing only)
    loss_y1 = result["mismatch_loss"][1] * 100.0  # year 1 (+ soiling + degradation)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(loss_y0, bins=20, alpha=0.6, label=f"Year 0 (mean {loss_y0.mean():.2f}%)")
    ax.hist(loss_y1, bins=20, alpha=0.6, label=f"Year 1 (mean {loss_y1.mean():.2f}%)")
    ax.axvspan(0.3, 1.0, alpha=0.08, color="green", label="Plausibility band [0.3–1.0%]")
    ax.set_xlabel("Mismatch loss [%]")
    ax.set_ylabel("Module count")
    ax.set_title("Phase 3: Year-0 vs Year-1 mismatch loss distribution (4x6 rack, Accra)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase3_mismatch_year1.png", dpi=150)
    plt.close(fig)


def plot_degradation_trajectories(
    result: dict[str, np.ndarray],
    exposure_map: np.ndarray,
) -> None:
    deg = result["degradation_factor"]  # (26, 24, 72)
    # Mean degradation factor per module per year: mean over cells
    deg_mod = deg.mean(axis=2)  # (26, 24)

    years = np.arange(26)
    corner_idx = [0]  # module 0 = top-left corner
    center_idx = [int(N_RACK_ROWS // 2 * N_RACK_COLS + N_RACK_COLS // 2)]

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, label, color in [
        (corner_idx[0], f"Corner (exp={exposure_map[corner_idx[0]]:.2f})", "steelblue"),
        (center_idx[0], f"Center (exp={exposure_map[center_idx[0]]:.2f})", "tomato"),
    ]:
        ax.plot(years, deg_mod[:, idx] * 100.0, "-o", ms=4, color=color, label=label)

    ax.set_xlabel("Project year")
    ax.set_ylabel("Mean cell degradation factor [%]")
    ax.set_title("Phase 3: 25-year degradation trajectory (corner vs. center module)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase3_degradation_25y.png", dpi=150)
    plt.close(fig)


def plot_dose_heatmap(tc_kelvin: np.ndarray) -> None:
    d_annual = annual_dose_factor(tc_kelvin)  # (24,)
    dose_map = d_annual.reshape(N_RACK_ROWS, N_RACK_COLS)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(dose_map, cmap="hot_r", vmin=dose_map.min() * 0.98)
    plt.colorbar(im, ax=ax, label="Annual Arrhenius dose factor [equiv. ref. years/year]")
    ax.set_title("Phase 3: Per-module annual Arrhenius dose (4x6 rack, Accra TMY)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    for r in range(N_RACK_ROWS):
        for c in range(N_RACK_COLS):
            ax.text(c, r, f"{dose_map[r, c]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase3_dose_map.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------


def print_mismatch_table(result: dict[str, np.ndarray]) -> None:
    print("\n" + "=" * 62)
    print("PHASE 3 MISMATCH SUMMARY  (4x6 rack, Accra TMY, seed=0)")
    print("=" * 62)
    print(f"{'Year':>5}  {'Mean loss [%]':>14}  {'Std [%]':>9}  {'Min [%]':>9}  {'Max [%]':>9}")
    print("-" * 52)
    for y in [0, 1, 5, 10, 15, 20, 25]:
        loss = result["mismatch_loss"][y] * 100.0
        print(
            f"{y:>5}  {loss.mean():>14.3f}  {loss.std():>9.3f}"
            f"  {loss.min():>9.3f}  {loss.max():>9.3f}"
        )
    print("=" * 62)
    band_pass = 0.3 <= float(result["mismatch_loss"][1].mean() * 100.0) <= 1.0
    print(f"Year-1 plausibility band [0.3–1.0%]: {'PASS' if band_pass else 'FAIL'}")


def print_degradation_summary(result: dict[str, np.ndarray], exposure_map: np.ndarray) -> None:
    deg = result["degradation_factor"]  # (26, 24, 72)
    deg25_per_module = deg[25].mean(axis=1)  # (24,) mean over cells

    corner_deg = float(deg25_per_module[0])
    center_idx = int(N_RACK_ROWS // 2 * N_RACK_COLS + N_RACK_COLS // 2)
    center_deg = float(deg25_per_module[center_idx])

    mono_ok = bool(np.all(np.diff(deg, axis=0) <= 0.0))

    print("\n" + "=" * 62)
    print("PHASE 3 DEGRADATION SUMMARY  (year 25)")
    print("=" * 62)
    print(f"  Mean degradation factor (all modules): {deg25_per_module.mean():.4f}")
    print(
        f"  Corner module (exp={exposure_map[0]:.2f}):  {corner_deg:.4f}"
        f"  ({(1 - corner_deg)*100:.2f}% loss)"
    )
    print(
        f"  Center module (exp={exposure_map[center_idx]:.2f}):  {center_deg:.4f}"
        f"  ({(1 - center_deg)*100:.2f}% loss)"
    )
    print(f"  Monotonic trajectory: {'PASS' if mono_ok else 'FAIL'}")
    print("=" * 62)


def print_exposure_map(exposure_map: np.ndarray) -> None:
    print("\nRack exposure map (4x6, row-major):")
    arr = exposure_map.reshape(N_RACK_ROWS, N_RACK_COLS)
    for row in arr:
        print("  " + "  ".join(f"{v:.2f}" for v in row))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("Phase 3 validation — building per-module T_c history (may take ~30 s) ...")

    exposure_map = rack_exposure_map(N_RACK_ROWS, N_RACK_COLS)
    print_exposure_map(exposure_map)

    tc_kelvin = build_tc_history()
    d_annual = annual_dose_factor(tc_kelvin)
    print(f"\nAnnual Arrhenius dose factor range: {d_annual.min():.3f} – {d_annual.max():.3f}")
    print("(1.0 = equivalent to T_ref=25 degC all year)")

    print("\nRunning 25-year Monte Carlo simulation ...")
    result = simulate_mismatch_25years(tc_kelvin, n_years=25, seed=0)

    print_mismatch_table(result)
    print_degradation_summary(result, exposure_map)

    plot_year1_mismatch(result)
    plot_degradation_trajectories(result, exposure_map)
    plot_dose_heatmap(tc_kelvin)

    print(f"\nPlots saved to {OUTPUT_DIR}/")
    print("  phase3_mismatch_year1.png")
    print("  phase3_degradation_25y.png")
    print("  phase3_dose_map.png")


if __name__ == "__main__":
    main()
