"""Sanity: packages import and required project files exist."""

from pathlib import Path

import core
import registry


def test_packages_import() -> None:
    assert core is not None and registry is not None


def test_required_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    for f in ["CLAUDE.md", "docs/SPEC.md", "pyproject.toml"]:
        assert (root / f).exists(), f"missing {f}"
