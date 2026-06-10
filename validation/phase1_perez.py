"""
Phase 1 validation: annual POA comparison — pvlib vs discrete vs continuous Perez.

Three Ghana TMY datasets: Accra (lat 5.55°N), Kumasi (6.69°N), Tamale (9.40°N).

Data source
-----------
Attempts PVGIS TMY download via pvlib.iotools.get_pvgis_tmy.
Falls back to a pvlib Ineichen clearsky model if the network is unavailable;
in that case results reflect a cloudless sky (no cloud cover variability) and
the POA figures are for orientation comparison only.

Outputs
-------
validation/output/phase1_annual_poa.png      — grouped bar chart, annual kWh m⁻²
validation/output/phase1_monthly_dev.png     — monthly % deviation cont vs disc
validation/output/phase1_pvlib_anchor.png    — monthly % deviation disc vs pvlib
validation/output/phase1_scatter.png         — hourly POA scatter (cont vs disc)

Terminal output
---------------
Deviation table: for each site, annual % deviation of continuous and discrete
models relative to pvlib.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

import core.transposition.perez_continuous as cont
import core.transposition.perez_discrete as disc

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SITES: dict[str, dict[str, float]] = {
    "Accra": {"lat": 5.55, "lon": -0.22, "alt": 61.0},
    "Kumasi": {"lat": 6.69, "lon": -1.62, "alt": 250.0},
    "Tamale": {"lat": 9.40, "lon": -0.85, "alt": 183.0},
}

SURFACE_TILT = 10.0  # degrees — typical low-latitude fixed tilt
SURFACE_AZ = 180.0  # south-facing

_MONTHS = list(range(1, 13))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_pvgis_tmy(lat: float, lon: float) -> pd.DataFrame | None:
    """Download PVGIS TMY; return DataFrame with columns ghi, dhi, dni or None."""
    try:
        data, _, _, _ = pvlib.iotools.get_pvgis_tmy(
            latitude=lat,
            longitude=lon,
            outputformat="csv",
            usehorizon=True,
            timeout=30,
        )
        rename = {
            "G(h)": "ghi",
            "Gb(n)": "dni",
            "Gd(h)": "dhi",
            "ghi": "ghi",
            "dni": "dni",
            "dhi": "dhi",
        }
        data = data.rename(columns={k: v for k, v in rename.items() if k in data.columns})
        return data[["ghi", "dni", "dhi"]]
    except Exception as exc:
        warnings.warn(
            f"PVGIS download failed ({exc}); using clearsky fallback.",
            stacklevel=2,
        )
        return None


def _clearsky_tmy(lat: float, lon: float, alt: float) -> pd.DataFrame:
    """Generate a synthetic TMY using the Ineichen clearsky model."""
    loc = Location(latitude=lat, longitude=lon, altitude=alt, tz="UTC")
    times = pd.date_range("2019-01-01", periods=8760, freq="1h", tz="UTC")
    cs = loc.get_clearsky(times)  # ghi, dni, dhi columns
    return cs[["ghi", "dni", "dhi"]]


def load_tmy(lat: float, lon: float, alt: float) -> pd.DataFrame:
    data = _load_pvgis_tmy(lat, lon)
    if data is None:
        data = _clearsky_tmy(lat, lon, alt)
    return data


# ---------------------------------------------------------------------------
# Annual POA computation
# ---------------------------------------------------------------------------


def compute_annual_poa(site_name: str, info: dict[str, float]) -> dict[str, object]:
    """Return dict with pvlib, discrete, and continuous annual POA arrays."""
    lat, lon, alt = info["lat"], info["lon"], info["alt"]
    tmy = load_tmy(lat, lon, alt)

    times: pd.DatetimeIndex = (
        tmy.index
        if isinstance(tmy.index, pd.DatetimeIndex)
        else pd.date_range("2019-01-01", periods=8760, freq="1h", tz="UTC")
    )

    loc = Location(latitude=lat, longitude=lon, altitude=alt, tz="UTC")
    solar_pos = loc.get_solarposition(times)

    zenith = solar_pos["apparent_zenith"].values
    azimuth = solar_pos["azimuth"].values
    ghi = tmy["ghi"].values.astype(float)
    dhi = tmy["dhi"].values.astype(float)
    dni = tmy["dni"].values.astype(float)

    # Extraterrestrial DNI and airmass via pvlib
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values.astype(float)
    am_raw = pvlib.atmosphere.get_relative_airmass(zenith, model="kastenyoung1989")
    am = np.where(np.isnan(am_raw), 0.0, np.asarray(am_raw, dtype=float))

    # Daytime mask
    day = zenith < 90.0

    # --- pvlib reference (sky diffuse only; beam and ground computed separately) ---
    poa_pvlib_sky = np.zeros(len(times))
    poa_pvlib_sky[day] = np.asarray(
        pvlib.irradiance.perez(
            SURFACE_TILT,
            SURFACE_AZ,
            dhi[day],
            dni[day],
            dni_extra[day],
            zenith[day],
            azimuth[day],
            am[day],
        ),
        dtype=float,
    )
    poa_direct = np.maximum(
        0.0,
        dni
        * np.maximum(
            0.0,
            disc.cos_aoi(
                np.radians(zenith),
                np.radians(azimuth),
                np.radians(SURFACE_TILT),
                np.radians(SURFACE_AZ),
            ),
        ),
    )
    poa_ground = ghi * 0.25 * (1.0 - np.cos(np.radians(SURFACE_TILT))) / 2.0
    poa_pvlib = poa_direct + poa_pvlib_sky + poa_ground

    # --- Helios discrete ---
    r_disc = disc.transpose(
        ghi,
        dni,
        dhi,
        zenith,
        azimuth,
        SURFACE_TILT,
        SURFACE_AZ,
        dni_extra,
        am,
    )

    # --- Helios continuous ---
    r_cont = cont.transpose(
        ghi,
        dni,
        dhi,
        zenith,
        azimuth,
        SURFACE_TILT,
        SURFACE_AZ,
        dni_extra,
        am,
    )

    month = times.month

    return {
        "site": site_name,
        "times": times,
        "month": np.asarray(month),
        "poa_pvlib": poa_pvlib,
        "poa_disc": r_disc["poa_global"],
        "poa_cont": r_cont["poa_global"],
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _monthly_kwh(poa: np.ndarray, month: np.ndarray) -> np.ndarray:
    return np.array([poa[month == m].sum() / 1000.0 for m in _MONTHS])


def plot_annual_bar(results: list[dict[str, object]]) -> None:
    site_names = [str(r["site"]) for r in results]
    annual_pvlib = [float(np.sum(r["poa_pvlib"])) / 1000.0 for r in results]  # type: ignore[arg-type]
    annual_disc = [float(np.sum(r["poa_disc"])) / 1000.0 for r in results]  # type: ignore[arg-type]
    annual_cont = [float(np.sum(r["poa_cont"])) / 1000.0 for r in results]  # type: ignore[arg-type]

    x = np.arange(len(site_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, annual_pvlib, w, label="pvlib reference", color="#2196F3")
    ax.bar(x, annual_disc, w, label="Helios discrete", color="#FF9800")
    ax.bar(x + w, annual_cont, w, label="Helios continuous", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(site_names)
    ax.set_ylabel("Annual POA [kWh m⁻²]")
    ax.set_title("Phase 1: Annual POA — pvlib vs Helios (tilt=10°, S-facing)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase1_annual_poa.png", dpi=150)
    plt.close(fig)


def plot_monthly_deviation(
    results: list[dict[str, object]],
    fname: str,
    num_key: str,
    den_key: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(14, 4), sharey=True)
    axes_list = [axes] if len(results) == 1 else list(axes)
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    for ax, r in zip(axes_list, results, strict=True):
        num = _monthly_kwh(r[num_key], r["month"])  # type: ignore[arg-type, index]
        den = _monthly_kwh(r[den_key], r["month"])  # type: ignore[arg-type, index]
        dev = (num - den) / np.where(den > 0, den, np.nan) * 100.0
        colors = ["#D32F2F" if d < 0 else "#388E3C" for d in dev]
        ax.bar(month_labels, dev, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(str(r["site"]))
        ax.set_xlabel("Month")
    axes_list[0].set_ylabel("POA deviation [%]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close(fig)


def plot_hourly_scatter(results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(14, 4))
    axes_list = [axes] if len(results) == 1 else list(axes)
    for ax, r in zip(axes_list, results, strict=True):
        poa_d = np.asarray(r["poa_disc"])  # type: ignore[arg-type]
        poa_c = np.asarray(r["poa_cont"])  # type: ignore[arg-type]
        mask = poa_d > 1.0
        ax.scatter(poa_d[mask], poa_c[mask], s=0.3, alpha=0.4, color="#1565C0")
        lim = max(float(np.max(poa_d[mask])), 1.0)
        ax.plot([0, lim], [0, lim], "r--", linewidth=0.8, label="1:1")
        ax.set_title(str(r["site"]))
        ax.set_xlabel("Discrete POA [W m⁻²]")
    axes_list[0].set_ylabel("Continuous POA [W m⁻²]")
    fig.suptitle("Phase 1: Hourly POA — continuous vs discrete")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase1_scatter.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Deviation table
# ---------------------------------------------------------------------------


def print_deviation_table(results: list[dict[str, object]]) -> None:
    header = (
        f"{'Site':<10}  {'pvlib kWh/m2':>13}  {'disc kWh/m2':>12}"
        f"  {'cont kWh/m2':>12}  {'disc vs pvlib':>14}  {'cont vs pvlib':>14}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("PHASE 1 DEVIATION TABLE")
    print(sep)
    print(header)
    print("-" * len(header))
    for r in results:
        ann_pvlib = float(np.sum(r["poa_pvlib"])) / 1000.0  # type: ignore[arg-type]
        ann_disc = float(np.sum(r["poa_disc"])) / 1000.0  # type: ignore[arg-type]
        ann_cont = float(np.sum(r["poa_cont"])) / 1000.0  # type: ignore[arg-type]
        dev_disc = (ann_disc - ann_pvlib) / ann_pvlib * 100.0
        dev_cont = (ann_cont - ann_pvlib) / ann_pvlib * 100.0
        print(
            f"{r['site']:<10}  {ann_pvlib:>13.1f}  {ann_disc:>12.1f}"
            f"  {ann_cont:>12.1f}  {dev_disc:>+13.4f}%  {dev_cont:>+13.4f}%"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("Phase 1 Perez validation — loading TMY data ...")
    results = []
    for name, info in SITES.items():
        print(
            f"  Processing {name} (lat={info['lat']}, lon={info['lon']}) ...",
            end="",
            flush=True,
        )
        r = compute_annual_poa(name, info)
        results.append(r)
        print(" done")

    print("\nGenerating plots ...")
    plot_annual_bar(results)
    plot_monthly_deviation(
        results,
        "phase1_monthly_dev.png",
        num_key="poa_cont",
        den_key="poa_disc",
        title="Phase 1: Monthly POA deviation — continuous vs discrete [%]",
    )
    plot_monthly_deviation(
        results,
        "phase1_pvlib_anchor.png",
        num_key="poa_disc",
        den_key="poa_pvlib",
        title="Phase 1: Monthly POA deviation — discrete vs pvlib [%]",
    )
    plot_hourly_scatter(results)

    print_deviation_table(results)
    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
