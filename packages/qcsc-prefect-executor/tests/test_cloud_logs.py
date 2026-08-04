from __future__ import annotations

from pathlib import Path

import pytest
from qcsc_prefect_executor.cloud_logs import (
    MAX_CLOUD_LOG_CHARS,
    CloudJobSummary,
    CloudLogPolicy,
    emit_cloud_job_logs,
    read_log_text,
    tail_log,
    truncate_log,
)


class _LoggerStub:
    def __init__(self) -> None:
        self.info_lines: list[str] = []
        self.error_lines: list[str] = []

    def info(self, message: str) -> None:
        self.info_lines.append(message)

    def error(self, message: str) -> None:
        self.error_lines.append(message)


SUMMARY = CloudJobSummary(
    job_id="12345",
    state="SUCCEEDED",
    exit_code=0,
    elapsed="00:00:12",
    node="node001",
    stdout_path="/shared/job/output.out",
    stderr_path="/shared/job/output.err",
)


@pytest.mark.parametrize(
    ("mode", "expected_info", "expected_error"),
    [
        ("none", [], []),
        ("legacy", ["first\nsecond\nthird\n"], ["err-one\nerr-two\n"]),
        ("tail", ["second\nthird\n"], ["err-one\nerr-two\n"]),
        ("full", ["first\nsecond\nthird\n"], ["err-one\nerr-two\n"]),
    ],
)
def test_cloud_log_modes(mode: str, expected_info: list[str], expected_error: list[str]):
    logger = _LoggerStub()

    emit_cloud_job_logs(
        logger=logger,
        policy=CloudLogPolicy(mode=mode, tail_lines=2),
        summary=SUMMARY,
        stdout="first\nsecond\nthird\n",
        stderr="err-one\nerr-two\n",
    )

    assert logger.info_lines == expected_info
    assert logger.error_lines == expected_error


def test_summary_contains_scheduler_fields_and_only_requested_tail():
    logger = _LoggerStub()

    emit_cloud_job_logs(
        logger=logger,
        policy=CloudLogPolicy(mode="summary", tail_lines=1),
        summary=SUMMARY,
        stdout="do-not-send\nsend-this\n",
        stderr="old-error\nnew-error\n",
    )

    assert len(logger.info_lines) == 1
    info = logger.info_lines[0]
    assert "job_id=12345" in info
    assert "state=SUCCEEDED" in info
    assert "exit_code=0" in info
    assert "elapsed=00:00:12" in info
    assert "node=node001" in info
    assert "stdout_path=/shared/job/output.out" in info
    assert "send-this" in info
    assert "do-not-send" not in info
    assert logger.error_lines == ["HPC job stderr tail (job_id=12345):\nnew-error\n"]


def test_legacy_truncate_boundary_is_unchanged():
    at_limit = "x" * MAX_CLOUD_LOG_CHARS
    over_limit = at_limit + "yz"

    assert truncate_log(at_limit) == at_limit
    assert truncate_log(over_limit) == at_limit + "... (truncated 2 chars)"


def test_tail_boundaries_and_character_limit():
    assert tail_log("one\ntwo\n", lines=0) == ""
    assert tail_log("one\ntwo\n", lines=1) == "two\n"
    assert tail_log("one\ntwo", lines=8) == "one\ntwo"

    long_line = "x" * (MAX_CLOUD_LOG_CHARS + 3)
    bounded = tail_log(long_line, lines=1)
    assert bounded.startswith("... (truncated 3 leading chars)\n")
    assert bounded.endswith("x" * MAX_CLOUD_LOG_CHARS)


def test_read_log_text_replaces_invalid_utf8(tmp_path: Path):
    log_path = tmp_path / "binary.log"
    log_path.write_bytes(b"valid\xfftail\x80")

    assert read_log_text(log_path) == "valid\ufffdtail\ufffd"
    assert read_log_text(tmp_path / "missing.log") == ""
    assert read_log_text(tmp_path) == ""


def test_artifact_defaults_preserve_only_legacy_behavior():
    assert CloudLogPolicy().should_create_artifact(legacy_default=True) is True
    assert CloudLogPolicy().should_create_artifact(legacy_default=False) is False
    assert CloudLogPolicy(mode="summary").should_create_artifact(legacy_default=True) is False
    assert (
        CloudLogPolicy(mode="summary", create_artifact=True).should_create_artifact(
            legacy_default=False
        )
        is True
    )
    assert (
        CloudLogPolicy(mode="legacy", create_artifact=False).should_create_artifact(
            legacy_default=True
        )
        is False
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"mode": "unexpected"}, ValueError),
        ({"tail_lines": -1}, ValueError),
        ({"tail_lines": True}, TypeError),
        ({"create_artifact": "yes"}, TypeError),
    ],
)
def test_policy_rejects_invalid_settings(kwargs: dict[str, object], error: type[Exception]):
    with pytest.raises(error):
        CloudLogPolicy(**kwargs)
