from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from qcsc_prefect_executor.bulk.models import BulkJobRecord


@dataclass(frozen=True)
class NativeBulkManifestGroup:
    """Manifest files prepared for one native scheduler bulk group."""

    bulk_group_dir: Path
    manifest_dir: Path
    manifest_paths: list[Path]

    @property
    def bulk_count(self) -> int:
        return len(self.manifest_paths)


def create_native_bulk_group_manifests(
    *,
    bulk_group_dir: Path,
    jobs: Sequence[BulkJobRecord],
) -> NativeBulkManifestGroup:
    """Create one manifest directory for one native bulk submit group."""

    if not jobs:
        raise ValueError("Native bulk manifest generation requires at least one job.")

    resolved_group_dir = Path(bulk_group_dir).expanduser()
    manifest_dir = resolved_group_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: list[Path] = []
    used_indices: set[int] = set()
    for fallback_index, job in enumerate(jobs):
        if job.stage_id is None or not str(job.stage_id).strip():
            raise ValueError(f"Bulk job {job.job_key!r} requires stage_id.")

        bulk_index = fallback_index if job.bulk_index is None else int(job.bulk_index)
        if bulk_index < 0:
            raise ValueError(f"Bulk job {job.job_key!r} has negative bulk_index.")
        if bulk_index in used_indices:
            raise ValueError(f"Duplicate bulk_index {bulk_index} in native bulk group.")
        used_indices.add(bulk_index)

        manifest_path = manifest_dir / f"{bulk_index}.json"
        manifest = {
            "job_key": job.job_key,
            "stage_id": job.stage_id,
            "wave_id": job.wave_id,
            "target_id": job.target_id,
            "work_dir": str(job.work_dir),
            "command_args": dict(job.command_args),
            "expected_outputs": [str(path) for path in job.expected_outputs],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        manifest_paths.append(manifest_path)

    return NativeBulkManifestGroup(
        bulk_group_dir=resolved_group_dir,
        manifest_dir=manifest_dir,
        manifest_paths=manifest_paths,
    )
