"""Helpers for generating Prefect artifact keys."""

from __future__ import annotations

import re


def bulk_metrics_artifact_key(mode: str) -> str:
    normalized_mode = re.sub(r"[^a-z0-9]+", "-", mode.lower()).strip("-")
    if not normalized_mode:
        normalized_mode = "unknown"
    return f"gb-sqd-bulk-{normalized_mode}-metrics"
