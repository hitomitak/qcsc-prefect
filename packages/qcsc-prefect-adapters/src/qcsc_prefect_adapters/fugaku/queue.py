from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from qcsc_prefect_core.queue import QueueCapacity

_SECTION_RE = re.compile(r"^\s*(USER|GROUP|ALL)(?::\s*(\S+)?)?\s*$")
_LIMIT_ROW_RE = re.compile(r"^\s*(ru-\S+)\s+(\S+)\s+(\d+)\s*$")
_QUEUE_LIMIT_NAME = "ru-accept"
_NATIVE_BULK_LIMIT_NAMES = (
    "ru-accept-bulksubjob",
    "ru-accept-allsubjob",
    _QUEUE_LIMIT_NAME,
)


@dataclass(frozen=True)
class FugakuLimitRecord:
    """Parsed ``pjstat --limit`` row for one Fugaku limit item."""

    limit_name: str
    limit: int | None
    alloc: int


def _parse_limit_value(text: str) -> int | None:
    value = text.strip().lower()
    if value == "unlimited":
        return None
    return int(value)


def parse_pjstat_limit_records(
    stdout: str,
    *,
    group: str | None = None,
) -> dict[str, FugakuLimitRecord]:
    """Parse the target ``GROUP`` section from ``pjstat --limit`` output.

    Args:
        stdout: Raw output from ``pjstat --limit --group <group>``.
        group: Optional group name. When omitted, the first ``GROUP`` section is
            used.

    Returns:
        Mapping from Fugaku limit name, such as ``"ru-accept"``, to parsed
        limit and allocation values.

    Raises:
        ValueError: If no matching group section is found.
    """

    in_target_group = False
    found_group = False
    records: dict[str, FugakuLimitRecord] = {}

    for line in stdout.splitlines():
        section_match = _SECTION_RE.match(line)
        if section_match:
            section_kind = section_match.group(1)
            section_value = section_match.group(2)
            in_target_group = section_kind == "GROUP" and (group is None or section_value == group)
            if in_target_group:
                found_group = True
            elif found_group:
                break
            continue

        if not in_target_group:
            continue

        row_match = _LIMIT_ROW_RE.match(line)
        if not row_match:
            continue

        limit_name, limit_text, alloc_text = row_match.groups()
        records[limit_name] = FugakuLimitRecord(
            limit_name=limit_name,
            limit=_parse_limit_value(limit_text),
            alloc=int(alloc_text),
        )

    if not found_group:
        label = group if group is not None else "<first GROUP section>"
        raise ValueError(f"pjstat --limit output did not contain GROUP section {label!r}")

    return records


def _limit_names_for_capacity_mode(
    *,
    capacity_mode: str,
    limit_name: str | None,
) -> tuple[str, ...]:
    if limit_name:
        return (limit_name,)

    normalized_mode = capacity_mode.strip().lower().replace("-", "_")
    if normalized_mode == "native_bulk":
        return _NATIVE_BULK_LIMIT_NAMES
    if normalized_mode == "single":
        return (_QUEUE_LIMIT_NAME,)
    raise ValueError(
        f"Unsupported Fugaku capacity_mode {capacity_mode!r}; expected 'single' or 'native_bulk'."
    )


def _select_limit_record(
    records: dict[str, FugakuLimitRecord],
    *,
    capacity_mode: str,
    limit_name: str | None,
) -> FugakuLimitRecord | None:
    for candidate_name in _limit_names_for_capacity_mode(
        capacity_mode=capacity_mode,
        limit_name=limit_name,
    ):
        record = records.get(candidate_name)
        if record is not None:
            return record
    return None


def estimate_capacity_from_pjstat_limit(
    stdout: str,
    *,
    max_active_jobs: int,
    group: str | None = None,
    capacity_mode: str = "single",
    limit_name: str | None = None,
) -> QueueCapacity:
    """Estimate group-wide queue capacity from ``pjstat --limit`` output."""

    records = parse_pjstat_limit_records(stdout, group=group)
    record = _select_limit_record(
        records,
        capacity_mode=capacity_mode,
        limit_name=limit_name,
    )
    if record is None:
        expected_names = ", ".join(
            repr(name)
            for name in _limit_names_for_capacity_mode(
                capacity_mode=capacity_mode,
                limit_name=limit_name,
            )
        )
        raise ValueError(f"pjstat --limit output did not contain any of {expected_names}")

    limit = int(max_active_jobs) if record.limit is None else record.limit
    available_slots = max(0, limit - record.alloc)
    return QueueCapacity(
        max_active_jobs=limit,
        current_active_jobs=record.alloc,
        available_slots=available_slots,
        raw_output=stdout,
    )


def _run_pjstat_limit(*, group: str | None) -> str:
    args = ["pjstat", "--limit"]
    if group:
        args.extend(["--group", group])

    proc = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


@dataclass(frozen=True)
class FugakuQueueProbe:
    """Queue capacity probe for Fugaku PJM group-wide queue limits.

    Fugaku queue pressure must be estimated from ``pjstat --limit --group``
    because ``pjstat -v`` only exposes the caller's jobs. The ``project`` field
    maps to the PJM group name used by the existing Fugaku block/request model.
    """

    max_active_jobs: int = 1000
    safety_margin: int = 20
    project: str | None = None
    user: str | None = None
    queue: str | None = None
    capacity_mode: str = "single"
    limit_name: str | None = None

    def get_capacity(self) -> QueueCapacity:
        """Return a conservative capacity estimate from ``pjstat --limit``."""

        try:
            stdout = _run_pjstat_limit(group=self.project)
        except subprocess.CalledProcessError as exc:
            raw_output = "\n".join(
                part
                for part in [
                    str(exc.stdout or "").strip(),
                    str(exc.stderr or "").strip(),
                ]
                if part
            )
            return self._zero_capacity(raw_output=raw_output or str(exc))
        except Exception as exc:
            return self._zero_capacity(raw_output=str(exc))

        try:
            return estimate_capacity_from_pjstat_limit(
                stdout,
                max_active_jobs=self.max_active_jobs,
                group=self.project,
                capacity_mode=self.capacity_mode,
                limit_name=self.limit_name,
            )
        except Exception:
            return self._zero_capacity(raw_output=stdout)

    def _zero_capacity(self, *, raw_output: str | None) -> QueueCapacity:
        return QueueCapacity(
            max_active_jobs=int(self.max_active_jobs),
            current_active_jobs=int(self.max_active_jobs),
            available_slots=0,
            raw_output=raw_output,
        )
