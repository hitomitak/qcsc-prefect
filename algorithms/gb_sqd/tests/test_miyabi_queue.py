from __future__ import annotations

import asyncio

import pytest

from gb_sqd import miyabi_queue


QSTAT_OUTPUT = """\
Job Id: 12345.miyabi
    Job_Name = gbsqd-bulk-ext-a
    Job_Owner = user1@miyabi
    job_state = Q
    queue = regular-c

Job Id: 12346.miyabi
    Job_Name = unrelated-job
    Job_Owner = user1@miyabi
    job_state = R
    queue = regular-c

Job Id: 12347.miyabi
    Job_Name = gbsqd-bulk-ext-b
    Job_Owner = user1@miyabi
    job_state = F
    queue = regular-c

Job Id: 12348.miyabi
    Job_Name = gbsqd-bulk-ext-c
    Job_Owner = user2@miyabi
    job_state = Q
    queue = regular-c

Job Id: 12349.miyabi
    Job_Name = gbsqd-bulk-ext-d
    Job_Owner = user1@miyabi
    job_state = Q
    queue = regular-g
"""


def test_parse_qstat_listing_parses_multiple_rows():
    rows = miyabi_queue.parse_qstat_listing(QSTAT_OUTPUT)

    assert [row["Job_Id"] for row in rows] == [
        "12345.miyabi",
        "12346.miyabi",
        "12347.miyabi",
        "12348.miyabi",
        "12349.miyabi",
    ]


def test_filter_active_jobs_uses_terminal_state_and_scope():
    rows = miyabi_queue.parse_qstat_listing(QSTAT_OUTPUT)

    filtered = miyabi_queue.filter_active_jobs(
        rows,
        user="user1",
        queue_name="regular-c",
        job_name_prefix="gbsqd-bulk-",
    )

    assert [row["Job_Id"] for row in filtered] == ["12345.miyabi"]


@pytest.mark.asyncio
async def test_count_active_jobs_filters_by_scope(monkeypatch):
    async def fake_run_command(*args: str, **kwargs):
        assert args == ("qstat", "-f")
        return QSTAT_OUTPUT

    monkeypatch.setattr(miyabi_queue, "run_command", fake_run_command)

    count = await miyabi_queue.count_active_jobs(
        queue_name="regular-c",
        scope="flow_jobs_only",
        job_name_prefix="gbsqd-bulk-",
        user="user1",
    )

    assert count == 1


@pytest.mark.asyncio
async def test_wait_for_queue_slot_retries_until_capacity_available(monkeypatch):
    counts = iter([2, 1])
    sleep_calls: list[float] = []

    async def fake_count_active_jobs(**kwargs):
        return next(counts)

    async def fake_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setattr(miyabi_queue, "count_active_jobs", fake_count_active_jobs)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await miyabi_queue.wait_for_queue_slot(
        queue_name="regular-c",
        max_jobs_in_queue=2,
        scope="user_queue",
        poll_interval_seconds=30.0,
        user="user1",
    )

    assert result == 1
    assert sleep_calls == [30.0]
