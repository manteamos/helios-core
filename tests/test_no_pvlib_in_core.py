"""Guard: pvlib must never be imported by runtime packages (ADR 0001)."""

from pathlib import Path

RUNTIME_DIRS = ["core", "registry", "api"]


def test_no_pvlib_in_runtime_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders = [
        str(p.relative_to(root))
        for d in RUNTIME_DIRS
        for p in (root / d).rglob("*.py")
        if "pvlib" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"pvlib imported in runtime code: {offenders}"
