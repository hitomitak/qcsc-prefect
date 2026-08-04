"""Bounded Prefect Cloud logging policies for executor job output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

MAX_CLOUD_LOG_CHARS = 10_000

CloudLogMode = Literal["legacy", "none", "summary", "tail", "full"]


class CloudLogger(Protocol):
    """Logger surface used by :func:`emit_cloud_job_logs`."""

    def info(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...


@dataclass(frozen=True)
class CloudLogPolicy:
    """Control how executor output is sent to Prefect Cloud.

    ``legacy`` preserves the historical first-10,000-character log messages
    and caller-specific artifact behavior. ``none`` sends no result logs,
    ``summary`` sends scheduler metadata plus bounded tails, ``tail`` sends
    only bounded tails, and ``full`` sends complete stdout/stderr.

    ``create_artifact=None`` preserves artifacts only in ``legacy`` mode.
    Non-legacy modes therefore avoid artifact events unless a caller opts in
    explicitly with ``create_artifact=True``.
    """

    mode: CloudLogMode = "legacy"
    tail_lines: int = 20
    create_artifact: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"legacy", "none", "summary", "tail", "full"}:
            raise ValueError("cloud log mode must be one of: legacy, none, summary, tail, full.")
        if isinstance(self.tail_lines, bool) or not isinstance(self.tail_lines, int):
            raise TypeError("tail_lines must be an integer.")
        if self.tail_lines < 0:
            raise ValueError("tail_lines must be non-negative.")
        if self.create_artifact is not None and not isinstance(self.create_artifact, bool):
            raise TypeError("create_artifact must be a bool or None.")

    def should_create_artifact(self, *, legacy_default: bool) -> bool:
        """Resolve artifact creation while preserving the caller's old default."""

        if self.create_artifact is not None:
            return self.create_artifact
        return legacy_default if self.mode == "legacy" else False


@dataclass(frozen=True)
class CloudJobSummary:
    """Small scheduler result payload suitable for a Cloud log event."""

    job_id: str
    state: str | None = None
    exit_code: str | int | None = None
    elapsed: str | None = None
    node: str | None = None
    stdout_path: str | Path | None = None
    stderr_path: str | Path | None = None


def resolve_cloud_log_policy(policy: CloudLogPolicy | None) -> CloudLogPolicy:
    """Return the backward-compatible policy for an omitted argument."""

    return policy if policy is not None else CloudLogPolicy()


def truncate_log(text: str) -> str:
    """Apply the historical prefix truncation exactly."""

    if len(text) > MAX_CLOUD_LOG_CHARS:
        return text[:MAX_CLOUD_LOG_CHARS] + (
            f"... (truncated {len(text) - MAX_CLOUD_LOG_CHARS} chars)"
        )
    return text


def tail_log(text: str, *, lines: int) -> str:
    """Return at most ``lines`` final lines, bounded by the legacy char limit."""

    if lines == 0 or not text:
        return ""
    tail = "".join(text.splitlines(keepends=True)[-lines:])
    if len(tail) <= MAX_CLOUD_LOG_CHARS:
        return tail
    omitted = len(tail) - MAX_CLOUD_LOG_CHARS
    return f"... (truncated {omitted} leading chars)\n" + tail[-MAX_CLOUD_LOG_CHARS:]


def read_log_text(path: str | Path | None) -> str:
    """Read a UTF-8 log if it exists, replacing invalid byte sequences."""

    if path is None:
        return ""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    return file_path.read_text(encoding="utf-8", errors="replace")


def _display(value: object | None) -> str:
    return "-" if value is None or value == "" else str(value)


def format_cloud_job_summary(summary: CloudJobSummary, *, stdout_tail: str = "") -> str:
    """Render scheduler metadata and an optional stdout tail as one message."""

    message = "\n".join(
        [
            "HPC job summary",
            f"job_id={_display(summary.job_id)}",
            f"state={_display(summary.state)}",
            f"exit_code={_display(summary.exit_code)}",
            f"elapsed={_display(summary.elapsed)}",
            f"node={_display(summary.node)}",
            f"stdout_path={_display(summary.stdout_path)}",
            f"stderr_path={_display(summary.stderr_path)}",
        ]
    )
    if stdout_tail:
        message += f"\nstdout_tail:\n{stdout_tail}"
    return message


def emit_cloud_job_logs(
    *,
    logger: CloudLogger,
    policy: CloudLogPolicy,
    summary: CloudJobSummary,
    stdout: str = "",
    stderr: str = "",
) -> None:
    """Emit job output according to one normalized Cloud policy."""

    if policy.mode == "none":
        return
    if policy.mode == "legacy":
        if stdout:
            logger.info(truncate_log(stdout))
        if stderr:
            logger.error(truncate_log(stderr))
        return
    if policy.mode == "full":
        if stdout:
            logger.info(stdout)
        if stderr:
            logger.error(stderr)
        return

    stdout_tail = tail_log(stdout, lines=policy.tail_lines)
    stderr_tail = tail_log(stderr, lines=policy.tail_lines)
    if policy.mode == "tail":
        if stdout_tail:
            logger.info(stdout_tail)
        if stderr_tail:
            logger.error(stderr_tail)
        return

    logger.info(format_cloud_job_summary(summary, stdout_tail=stdout_tail))
    if stderr_tail:
        logger.error(f"HPC job stderr tail (job_id={summary.job_id}):\n{stderr_tail}")


__all__ = [
    "MAX_CLOUD_LOG_CHARS",
    "CloudJobSummary",
    "CloudLogMode",
    "CloudLogPolicy",
    "emit_cloud_job_logs",
    "format_cloud_job_summary",
    "read_log_text",
    "resolve_cloud_log_policy",
    "tail_log",
    "truncate_log",
]
