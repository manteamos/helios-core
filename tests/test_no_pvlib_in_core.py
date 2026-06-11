"""
Guard: pvlib must never be imported inside core/ or registry/ (ADR 0001).

The restriction per CLAUDE.md is "never import it inside `core/`" — the physics
solver directories must remain independent of pvlib so their implementations can
be validated *against* pvlib rather than *delegating to* it.

api/catalog.py is explicitly exempt: it uses pvlib exclusively to access the
bundled NREL CEC/SAM database CSV files (pvlib.pvsystem.retrieve_sam), not for
any physics calculation.  This is consistent with pvlib's role as reference data.
"""

from pathlib import Path

# Directories where pvlib physics must never appear
PHYSICS_DIRS = ["core", "registry"]

# api/ files that are intentionally allowed to import pvlib for data access only
# app.py mentions pvlib only in docstrings (catalog endpoint descriptions)
_ALLOWED_API_FILES = {"api/catalog.py", "api/app.py"}


def test_no_pvlib_in_physics_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders = [
        str(p.relative_to(root))
        for d in PHYSICS_DIRS
        for p in (root / d).rglob("*.py")
        if "pvlib" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"pvlib imported in physics core: {offenders}"


def test_no_pvlib_in_api_except_catalog() -> None:
    """pvlib must not leak into other api/ modules beyond catalog.py."""
    root = Path(__file__).resolve().parents[1]
    offenders = [
        str(p.relative_to(root))
        for p in (root / "api").rglob("*.py")
        if "pvlib" in p.read_text(encoding="utf-8")
        and str(p.relative_to(root)).replace("\\", "/") not in _ALLOWED_API_FILES
    ]
    assert not offenders, f"pvlib imported outside allowed api files: {offenders}"
