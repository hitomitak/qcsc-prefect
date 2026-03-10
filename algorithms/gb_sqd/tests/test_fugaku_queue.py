from __future__ import annotations

import asyncio

import pytest

from gb_sqd import fugaku_queue


PJSTAT_OUTPUT = """\
JOB_ID JOB_NAME MD ST USER GROUP START_DATE ELAPSE_TIM ELAPSE_LIM NODE_REQUIRE VNODE CORE V_MEM V_POL E_POL RANK LST EC PC SN PRI ACCEPT RSC_GRP REASON
43607195 other-job NM RUN user1 grp 2026-03-10T00:00:00 00:01:00 00:10:00 1 1 48 0M N N 1 - 0 - - - - small -
43607196 gbsqd-ext-abc NM QUE user1 grp 2026-03-10T00:00:00 00:00:00 00:10:00 1 1 48 0M N N 1 - 0 - - - - small -
43607197 gbsqd-ext-def NM EXT user1 grp 2026-03-10T00:00:00 00:10:00 00:10:00 1 1 48 0M N N 1 - 0 - - - - small -
43607198 gbsqd-trim-ghi NM RUN user2 grp 2026-03-10T00:00:00 00:01:00 00:10:00 1 1 48 0M N N 1 - 0 - - - - small -
"""


def test_parse_pjstat_listing_parses_multiple_rows():
    rows = fugaku_queue.parse_pjstat_listing(PJSTAT_OUTPUT)

    assert [row["JOB_ID"] for row in rows] == ["43607195", "43607196", "43607197", "43607198"]


def test_filter_active_jobs_uses_terminal_state_and_scope():
    rows = fugaku_queue.parse_pjstat_listing(PJSTAT_OUTPUT)

    filtered = fugaku_queue.filter_active_jobs(
        rows,
        user="user1",
        resource_group="small",
        job_name_prefix="gbsqd-",
    )

    assert [row["JOB_ID"] for row in filtered] == ["43607196"]


@pytest.mark.asyncio
async def test_count_active_jobs_filters_by_scope(monkeypatch):
    async def fake_run_command(*args: str, **kwargs):
        assert args == ("pjstat",)
        return PJSTAT_OUTPUT

    monkeypatch.setattr(fugaku_queue, "run_command", fake_run_command)

    count = await fugaku_queue.count_active_jobs(
        resource_group="small",
        scope="flow_jobs_only",
        job_name_prefix="gbsqd-",
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

    monkeypatch.setattr(fugaku_queue, "count_active_jobs", fake_count_active_jobs)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await fugaku_queue.wait_for_queue_slot(
        resource_group="small",
        max_jobs_in_queue=2,
        scope="user_queue",
        poll_interval_seconds=30.0,
        user="user1",
    )

    assert result == 1
    assert sleep_calls == [30.0]
