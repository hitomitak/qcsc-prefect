"""Helpers for per-target parameter overrides in bulk GB-SQD flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


def normalize_relative_target_path(path: str | Path) -> str:
    """Normalize a relative target path into the canonical `a/b/c` form."""

    raw = str(path).strip().replace("\\", "/")
    if not raw or raw == ".":
        return ""

    parts = [part for part in raw.split("/") if part and part != "."]
    return "/".join(parts)


def prepare_target_overrides(
    *,
    discovered_relative_paths: Iterable[str],
    target_overrides: Mapping[str, Mapping[str, Any]] | None,
    allowed_parameter_names: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Validate and normalize user-provided per-target parameter overrides."""

    if not target_overrides:
        return {}

    discovered = {normalize_relative_target_path(path) for path in discovered_relative_paths}
    allowed_names = set(allowed_parameter_names)
    prepared: dict[str, dict[str, Any]] = {}

    for raw_target_path, raw_override in target_overrides.items():
        normalized_target_path = normalize_relative_target_path(raw_target_path)
        if normalized_target_path not in discovered:
            raise ValueError(
                "target_overrides contains an unknown target path: "
                f"{raw_target_path!r}. Use a discovered relative path such as "
                f"'13_18MO_Wat/atom_10129'."
            )
        if not isinstance(raw_override, Mapping):
            raise TypeError(
                "Each target_overrides value must be a mapping of parameter names to override values. "
                f"Got {type(raw_override).__name__} for {raw_target_path!r}."
            )

        override = dict(raw_override)
        unknown_parameter_names = sorted(
            parameter_name for parameter_name in override if parameter_name not in allowed_names
        )
        if unknown_parameter_names:
            raise ValueError(
                "target_overrides contains unsupported GB-SQD parameter names for "
                f"{raw_target_path!r}: {', '.join(unknown_parameter_names)}"
            )
        prepared[normalized_target_path] = override

    return prepared


def merge_target_job_parameters(
    *,
    base_job_parameters: Mapping[str, Any],
    target_overrides: Mapping[str, Mapping[str, Any]] | None,
    relative_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge shared job parameters with the overrides for one target."""

    normalized_relative_path = normalize_relative_target_path(relative_path)
    override = dict((target_overrides or {}).get(normalized_relative_path, {}))
    merged = dict(base_job_parameters)
    merged.update(override)
    return merged, override
