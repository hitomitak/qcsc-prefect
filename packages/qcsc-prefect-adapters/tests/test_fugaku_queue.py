from __future__ import annotations

import subprocess

from qcsc_prefect_adapters.fugaku import queue as queue_mod


def _limit_row(limit_name: str, limit: str, alloc: int) -> str:
    return f"  {limit_name:<38} {limit:>10} {alloc:>13}"


_PJSTAT_LIMIT_SAMPLE = "\n".join(
    [
        " System Resource Information:",
        " RSCUNIT: rscunit_ft01",
        " USER: u13450",
        "  LIMIT-NAME                           LIMIT         ALLOC",
        _limit_row("ru-accept", "unlimited", 0),
        _limit_row("ru-run-job", "unlimited", 0),
        " GROUP: ra010014",
        "  LIMIT-NAME                           LIMIT         ALLOC",
        _limit_row("ru-accept", "1000", 7),
        _limit_row("ru-accept-allsubjob", "unlimited", 7),
        _limit_row("ru-accept-bulksubjob", "2000", 0),
        _limit_row("ru-run-job", "unlimited", 6),
        " ALL:",
        "  LIMIT-NAME                           LIMIT         ALLOC",
        _limit_row("ru-accept", "40000", 3870),
        _limit_row("ru-accept-bulksubjob", "42000", 108),
        _limit_row("ru-run-job", "unlimited", 3443),
    ]
) + "\n"


def test_parse_pjstat_limit_records_parses_group_section():
    records = queue_mod.parse_pjstat_limit_records(
        _PJSTAT_LIMIT_SAMPLE,
        group="ra010014",
    )

    assert records["ru-accept"].limit_name == "ru-accept"
    assert records["ru-accept"].limit == 1000
    assert records["ru-accept"].alloc == 7
    assert records["ru-run-job"].limit is None
    assert records["ru-run-job"].alloc == 6


def test_estimate_capacity_from_pjstat_limit_uses_group_ru_accept():
    capacity = queue_mod.estimate_capacity_from_pjstat_limit(
        _PJSTAT_LIMIT_SAMPLE,
        max_active_jobs=1000,
        group="ra010014",
    )

    assert capacity.max_active_jobs == 1000
    assert capacity.current_active_jobs == 7
    assert capacity.available_slots == 993
    assert capacity.raw_output == _PJSTAT_LIMIT_SAMPLE


def test_estimate_capacity_from_pjstat_limit_uses_configured_max_for_unlimited_limit():
    output = "\n".join(
        [
            " GROUP: ra010014",
            "  LIMIT-NAME                           LIMIT         ALLOC",
            _limit_row("ru-accept", "unlimited", 7),
        ]
    )

    capacity = queue_mod.estimate_capacity_from_pjstat_limit(
        output,
        max_active_jobs=1000,
        group="ra010014",
    )

    assert capacity.max_active_jobs == 1000
    assert capacity.current_active_jobs == 7
    assert capacity.available_slots == 993


def test_fugaku_queue_probe_reads_pjstat_limit(monkeypatch):
    calls: list[str | None] = []

    def fake_run_pjstat_limit(*, group: str | None) -> str:
        calls.append(group)
        return _PJSTAT_LIMIT_SAMPLE

    monkeypatch.setattr(queue_mod, "_run_pjstat_limit", fake_run_pjstat_limit)
    probe = queue_mod.FugakuQueueProbe(
        max_active_jobs=1000,
        safety_margin=20,
        project="ra010014",
    )

    capacity = probe.get_capacity()

    assert calls == ["ra010014"]
    assert capacity.max_active_jobs == 1000
    assert capacity.current_active_jobs == 7
    assert capacity.available_slots == 993
    assert capacity.raw_output == _PJSTAT_LIMIT_SAMPLE


def test_run_pjstat_limit_uses_group_argument(monkeypatch):
    calls: list[list[str]] = []

    class _CompletedProcess:
        stdout = "ok"

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> _CompletedProcess:
        calls.append(args)
        assert check is True
        assert capture_output is True
        assert text is True
        return _CompletedProcess()

    monkeypatch.setattr(queue_mod.subprocess, "run", fake_run)

    assert queue_mod._run_pjstat_limit(group="ra010014") == "ok"
    assert calls == [["pjstat", "--limit", "--group", "ra010014"]]


def test_fugaku_queue_probe_failure_path_returns_zero_capacity(monkeypatch):
    def fake_run_pjstat_limit(*, group: str | None) -> str:
        raise subprocess.CalledProcessError(
            1,
            ["pjstat", "--limit", "--group", group or ""],
            output="partial stdout",
            stderr="pjstat failed",
        )

    monkeypatch.setattr(queue_mod, "_run_pjstat_limit", fake_run_pjstat_limit)
    probe = queue_mod.FugakuQueueProbe(max_active_jobs=10, safety_margin=1, project="ra010014")

    capacity = probe.get_capacity()

    assert capacity.max_active_jobs == 10
    assert capacity.current_active_jobs == 10
    assert capacity.available_slots == 0
    assert "pjstat failed" in str(capacity.raw_output)
