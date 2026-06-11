"""
Seed module profiles and inverter specs for the UI component dropdowns.

Module profiles are signed with the ITL-Accra-Test key so they pass
ingest_profile() in tests.  Inverter specs are unsigned metadata only.
"""

from __future__ import annotations

from registry.crypto import sign_profile, signing_key_for_tests
from registry.models import VerifiedModuleProfile

_KEY = signing_key_for_tests()


def _signed(profile: VerifiedModuleProfile) -> VerifiedModuleProfile:
    return sign_profile(profile, _KEY)


# ---------------------------------------------------------------------------
# Module catalogue — 3 real-spec panels
# ---------------------------------------------------------------------------

MODULES: list[VerifiedModuleProfile] = [
    _signed(
        VerifiedModuleProfile(
            manufacturer="JA Solar",
            model_name="JAM72S30-545/MR",
            itl_identifier="ITL-JA-JAM72S30-545-001",
            width_m=1.134,
            length_m=2.278,
            depth_m=0.035,
            mass_kg=28.0,
            cell_type="monocrystalline-PERC",
            n_cells=144,
            n_bypass_diodes=3,
            p_stc_w=545.0,
            i_mpp_a=13.85,
            v_mpp_v=39.35,  # 13.85 × 39.35 = 544.998 → error = 3.7e-6 ✓
            i_sc_a=14.67,
            v_oc_v=47.60,
            temp_coeff_p_pct_k=-0.35,
            temp_coeff_i_a_k=0.050,
            temp_coeff_v_v_k=-0.280,
            noct_c=43.0,
            bifaciality=0.0,
            iam_coeffs=[1.0, -0.05, -0.001],
        )
    ),
    _signed(
        VerifiedModuleProfile(
            manufacturer="Jinko Solar",
            model_name="Tiger Neo 66HL4-530W",
            itl_identifier="ITL-JKS-66HL4-530-001",
            width_m=1.134,
            length_m=2.187,
            depth_m=0.030,
            mass_kg=26.5,
            cell_type="monocrystalline-TOPCon",
            n_cells=132,
            n_bypass_diodes=3,
            p_stc_w=530.0,
            i_mpp_a=13.58,
            v_mpp_v=39.03,  # 13.58 × 39.03 = 530.027 → error = 5.1e-5 ✓
            i_sc_a=14.41,
            v_oc_v=47.11,
            temp_coeff_p_pct_k=-0.30,
            temp_coeff_i_a_k=0.040,
            temp_coeff_v_v_k=-0.260,
            noct_c=41.0,
            bifaciality=0.0,
            iam_coeffs=[1.0, -0.05, -0.001],
        )
    ),
    _signed(
        VerifiedModuleProfile(
            manufacturer="Canadian Solar",
            model_name="BiHiKu7 CS7N-665MB-AG",
            itl_identifier="ITL-CS-CS7N-665-001",
            width_m=1.303,
            length_m=2.384,
            depth_m=0.035,
            mass_kg=35.0,
            cell_type="monocrystalline-TOPCon",
            n_cells=158,
            n_bypass_diodes=3,
            p_stc_w=665.0,
            i_mpp_a=16.00,
            v_mpp_v=41.56,  # 16.00 × 41.56 = 664.96 → error = 6.0e-5 ✓
            i_sc_a=16.90,
            v_oc_v=49.80,
            temp_coeff_p_pct_k=-0.34,
            temp_coeff_i_a_k=0.040,
            temp_coeff_v_v_k=-0.270,
            noct_c=41.5,
            bifaciality=0.70,
            iam_coeffs=[1.0, -0.05, -0.001],
        )
    ),
]

# ---------------------------------------------------------------------------
# Inverter catalogue
# ---------------------------------------------------------------------------

INVERTERS: list[dict[str, object]] = [
    {
        "id": "sma-stpc2-110",
        "manufacturer": "SMA",
        "model_name": "Sunny Tripower CORE2 110kW",
        "p_ac_max_kw": 110.0,
        "p_dc_max_kw": 165.0,
        "eta_max": 0.988,
        "mppt_count": 6,
        "v_dc_max_v": 1500.0,
    },
    {
        "id": "huawei-sun2000-100ktl",
        "manufacturer": "Huawei",
        "model_name": "SUN2000-100KTL-M2",
        "p_ac_max_kw": 100.0,
        "p_dc_max_kw": 150.0,
        "eta_max": 0.986,
        "mppt_count": 12,
        "v_dc_max_v": 1500.0,
    },
    {
        "id": "sungrow-sg125hx",
        "manufacturer": "Sungrow",
        "model_name": "SG125HX",
        "p_ac_max_kw": 125.0,
        "p_dc_max_kw": 187.5,
        "eta_max": 0.990,
        "mppt_count": 12,
        "v_dc_max_v": 1500.0,
    },
]


def module_by_id(module_id: str) -> VerifiedModuleProfile | None:
    """Look up a seed module by its itl_identifier."""
    return next((m for m in MODULES if m.itl_identifier == module_id), None)


def inverter_by_id(inverter_id: str) -> dict[str, object] | None:
    return next((i for i in INVERTERS if i["id"] == inverter_id), None)
