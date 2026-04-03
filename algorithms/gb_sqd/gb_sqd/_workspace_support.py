"""Helpers for importing local qcsc-prefect packages during development."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_SRC_DIRS = {
    "qcsc-prefect-core": _PROJECT_ROOT / "packages" / "qcsc-prefect-core" / "src",
    "qcsc-prefect-adapters": _PROJECT_ROOT / "packages" / "qcsc-prefect-adapters" / "src",
    "qcsc-prefect-blocks": _PROJECT_ROOT / "packages" / "qcsc-prefect-blocks" / "src",
    "qcsc-prefect-executor": _PROJECT_ROOT / "packages" / "qcsc-prefect-executor" / "src",
}


def ensure_workspace_packages(*package_names: str) -> None:
    """Add local monorepo package sources to ``sys.path`` when available."""

    if not (_PROJECT_ROOT / "packages").exists():
        return

    pending_paths: list[str] = []
    for package_name in package_names:
        package_path = _PACKAGE_SRC_DIRS.get(package_name)
        if package_path and package_path.exists():
            pending_paths.append(str(package_path))

    for package_path in reversed(pending_paths):
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
