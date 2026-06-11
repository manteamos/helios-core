"""
Component catalog backed by pvlib's bundled CEC module and SAM inverter databases.

NOTE ON PVLIB USAGE
-------------------
pvlib is imported here to access its bundled CEC/SAM database CSV files via
pvlib.pvsystem.retrieve_sam().  That function reads CSV data files shipped with
pvlib — it performs no physics calculations.  Per CLAUDE.md the restriction is
"never import it inside `core/`"; this module is api/catalog.py.  The rule
exists to keep core/ physics solvers independent of pvlib; data file access
in the API layer is a separate concern.

The catalog automatically reflects the installed pvlib version.  Upgrading
pvlib (pip install --upgrade pvlib) and restarting the API server is all that
is needed to pick up a new database release.

Databases
---------
CEC modules  : ~21 500 entries, 25 parameters (Technology, STC, Width, Length, …)
SAM inverters: ~3 200 entries, 16 parameters (Paco, Pdco, Vdco, …)
"""

from __future__ import annotations

import re

import pandas as pd
import pvlib

# ---------------------------------------------------------------------------
# Lazy-loaded DataFrames (loaded once, cached for the process lifetime)
# ---------------------------------------------------------------------------

_modules_df: pd.DataFrame | None = None
_inverters_df: pd.DataFrame | None = None


def _clean_name(raw: str) -> str:
    """Replace underscores/hyphens clusters with single spaces; strip edges."""
    return re.sub(r"[_]+", " ", raw).strip()


def _parse_inv_name(key: str) -> tuple[str, str]:
    """
    Inverter keys use '__' as separator: 'ABB__PVS980-58__480V' →
    manufacturer='ABB', model='PVS980-58 480V'.
    """
    parts = key.split("__", 1)
    if len(parts) == 2:
        mfr = _clean_name(parts[0])
        model = _clean_name(parts[1])
    else:
        mfr = "Unknown"
        model = _clean_name(key)
    return mfr, model


def _load_modules() -> pd.DataFrame:
    """
    Load CEC module database.  Returns transposed DataFrame:
    rows = modules, columns = parameters.
    """
    raw = pvlib.pvsystem.retrieve_sam("cecmod")
    df = raw.T.copy()
    df.index.name = "key"
    df = df.reset_index()
    df["display_name"] = df["key"].apply(_clean_name)
    return df


def _load_inverters() -> pd.DataFrame:
    """
    Load SAM CEC inverter database.  Returns transposed DataFrame:
    rows = inverters, columns = parameters.
    """
    raw = pvlib.pvsystem.retrieve_sam("cecinverter")
    df = raw.T.copy()
    df.index.name = "key"
    df = df.reset_index()
    mfr_model = df["key"].apply(
        lambda k: pd.Series(_parse_inv_name(k), index=["manufacturer", "model"])
    )
    df = pd.concat([df, mfr_model], axis=1)
    return df


def _get_modules() -> pd.DataFrame:
    global _modules_df
    if _modules_df is None:
        _modules_df = _load_modules()
    return _modules_df


def _get_inverters() -> pd.DataFrame:
    global _inverters_df
    if _inverters_df is None:
        _inverters_df = _load_inverters()
    return _inverters_df


def pvlib_version() -> str:
    return pvlib.__version__


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _row_to_module(row: pd.Series) -> dict:  # type: ignore[type-arg]
    p_stc = float(row.get("STC") or 0.0)
    area = float(row.get("A_c") or 0.001)
    width = float(row.get("Width") or 0.0)
    length = float(row.get("Length") or 0.0)
    # Fall back to area-derived dimensions when the DB entry is missing w/l
    if width <= 0 or length <= 0:
        aspect = 1.8
        width = (area / aspect) ** 0.5
        length = area / max(width, 1e-6)
    gamma = float(row.get("gamma_r") or -0.38)
    noct = float(row.get("T_NOCT") or 45.0)
    bifacial = bool(row.get("Bifacial") or False)
    return {
        "id": str(row["key"]),
        "name": str(row["display_name"]),
        "technology": str(row.get("Technology") or ""),
        "p_stc_w": round(p_stc, 1),
        "width_m": round(width, 4),
        "length_m": round(length, 4),
        "area_m2": round(area, 4),
        "i_sc_a": round(float(row.get("I_sc_ref") or 0), 3),
        "v_oc_v": round(float(row.get("V_oc_ref") or 0), 2),
        "i_mp_a": round(float(row.get("I_mp_ref") or 0), 3),
        "v_mp_v": round(float(row.get("V_mp_ref") or 0), 2),
        "temp_coeff_p_pct_k": round(gamma, 3),
        "noct_c": round(noct, 1),
        "bifacial": bifacial,
        "eta_ref": round(p_stc / (1000.0 * max(area, 1e-6)), 4),
    }


def search_modules(
    q: str = "",
    limit: int = 20,
    offset: int = 0,
) -> dict:  # type: ignore[type-arg]
    """
    Full-text search over module names.  Returns paginated results.

    Parameters
    ----------
    q      : substring to search in display_name (case-insensitive)
    limit  : page size (max 100)
    offset : skip this many rows before returning results
    """
    df = _get_modules()
    limit = min(limit, 100)

    if q:
        mask = df["display_name"].str.contains(q, case=False, na=False)
        df = df[mask]

    total = len(df)
    page = df.iloc[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "pvlib_version": pvlib_version(),
        "items": [_row_to_module(row) for _, row in page.iterrows()],
    }


def get_module_by_id(module_id: str) -> dict | None:  # type: ignore[type-arg]
    """Look up a module from the CEC database by its key (exact match)."""
    df = _get_modules()
    matches = df[df["key"] == module_id]
    if matches.empty:
        return None
    return _row_to_module(matches.iloc[0])


# ---------------------------------------------------------------------------
# Inverter helpers
# ---------------------------------------------------------------------------


def _row_to_inverter(row: pd.Series) -> dict:  # type: ignore[type-arg]
    paco = float(row.get("Paco") or 0.0)
    pdco = float(row.get("Pdco") or max(paco, 1.0))
    eta = round(paco / max(pdco, 1.0), 4)
    return {
        "id": str(row["key"]),
        "name": _clean_name(str(row["key"])),
        "manufacturer": str(row.get("manufacturer") or ""),
        "model": str(row.get("model") or ""),
        "p_ac_max_w": round(paco, 1),
        "p_dc_max_w": round(pdco, 1),
        "v_dc_nom_v": round(float(row.get("Vdco") or 0), 1),
        "v_dc_max_v": round(float(row.get("Vdcmax") or 0), 1),
        "mppt_low_v": round(float(row.get("Mppt_low") or 0), 1),
        "mppt_high_v": round(float(row.get("Mppt_high") or 0), 1),
        "eta_max": eta,
        "cec_type": str(row.get("CEC_Type") or ""),
    }


def search_inverters(
    q: str = "",
    limit: int = 20,
    offset: int = 0,
) -> dict:  # type: ignore[type-arg]
    """Full-text search over inverter names.  Returns paginated results."""
    df = _get_inverters()
    limit = min(limit, 100)

    if q:
        mask = df["key"].str.contains(q.replace(" ", "_"), case=False, na=False) | df[
            "key"
        ].str.contains(q, case=False, na=False)
        df = df[mask]

    total = len(df)
    page = df.iloc[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "pvlib_version": pvlib_version(),
        "items": [_row_to_inverter(row) for _, row in page.iterrows()],
    }
