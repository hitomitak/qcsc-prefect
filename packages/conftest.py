from __future__ import annotations

import sys
from pathlib import Path


def _add_src_path(path: Path) -> None:
    if path.exists():
        sys.path.insert(0, str(path))


_ROOT = Path(__file__).resolve().parent
_add_src_path(_ROOT / "hpc-prefect-core" / "src")
_add_src_path(_ROOT / "hpc-prefect-adapters" / "src")
_add_src_path(_ROOT / "hpc-prefect-executor" / "src")
