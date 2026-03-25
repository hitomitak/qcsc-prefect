from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gb_sqd.tasks.bulk_target_run import NonRetryableBulkError, bulk_target_run_task
import gb_sqd.tasks.bulk_target_run as bulk_target_run_module


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


def _write_input_case(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "count_dict.txt").write_text("count")
    (path / "fci_dump.txt").write_text("fcidump")


def _ext_job_parameters() -> dict[str, object]:
    return {
        "num_recovery": 2,
        "num_batches": 3,
        "num_samples_per_batch": 1000,
        "iteration": 2,
        "block": 10,
        "tolerance": 1.0e-4,
        "max_time": 300.0,
        "adet_comm_size": 1,
        "bdet_comm_size": 1,
        "task_comm_size": 1,
        "adet_comm_size_final": 2,
        "bdet_comm_size_final": 1,
        "task_comm_size_final": 1,
        "do_carryover_in_recovery": True,
        "carryover_threshold": 1.0e-5,
        "carryover_ratio": 0.5,
        "with_hf": True,
        "verbose": True,
    }


@pytest.mark.asyncio
async def test_bulk_target_run_task_retries_then_succeeds(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "input" / "case_a" / "atom_1"
    output_root_dir = tmp_path / "output"
    _write_input_case(input_dir)

    monkeypatch.setattr(bulk_target_run_module, "get_run_logger", lambda: _FakeLogger())
    monkeypatch.setattr(
        bulk_target_run_module.HPCProfileBlock,
        "load",
        staticmethod(lambda _: SimpleNamespace(queue_cpu="small", hpc_target="fugaku")),
    )

    queue_calls: list[dict[str, object]] = []

    async def fake_wait_for_queue_slot(**kwargs):
        queue_calls.append(kwargs)
        return len(queue_calls) - 1

    run_calls: list[dict[str, object]] = []

    async def fake_run_job_from_blocks(**kwargs):
        run_calls.append(kwargs)
        if len(run_calls) == 1:
            raise RuntimeError("simulated scheduler failure")

        energy_log_file = Path(kwargs["work_dir"]) / "energy_log.json"
        energy_log_file.write_text(json.dumps({"energy_final": -123.456}))
        return SimpleNamespace(exit_status=0, job_id="job-0002", state="EXT")

    monkeypatch.setattr(bulk_target_run_module, "wait_for_queue_slot", fake_wait_for_queue_slot)
    monkeypatch.setattr(bulk_target_run_module, "run_job_from_blocks", fake_run_job_from_blocks)

    result = await bulk_target_run_task.fn(
        target_name="case_a__atom_1",
        mode="ext_sqd",
        input_dir=str(input_dir),
        relative_path="case_a/atom_1",
        output_root_dir=str(output_root_dir),
        count_dict_filename="count_dict.txt",
        fcidump_filename="fci_dump.txt",
        command_block_name="cmd-gb-sqd-ext",
        execution_profile_block_name="exec-gb-sqd-ext-fugaku",
        hpc_profile_block_name="hpc-fugaku-gb-sqd",
        max_jobs_in_queue=2,
        queue_limit_scope="user_queue",
        queue_poll_interval_seconds=30.0,
        job_name_prefix="gbsqd-bulk",
        skip_completed=True,
        max_attempts=2,
        job_parameters=_ext_job_parameters(),
        parameter_overrides={"max_time": 600.0},
    )

    assert result["status"] == "success"
    assert result["latest_attempt"] == 2
    assert result["parameter_overrides"] == {"max_time": 600.0}
    assert Path(result["latest_output_dir"]).name == "attempt_002"
    assert len(queue_calls) == 2
    assert all(call["resource_group"] == "small" for call in queue_calls)
    assert len(run_calls) == 2

    status_file = output_root_dir / "case_a" / "atom_1" / "target_status.json"
    status = json.loads(status_file.read_text())
    assert status["status"] == "success"
    assert status["latest_attempt"] == 2
    assert status["latest_parameter_overrides"] == {"max_time": 600.0}
    assert [attempt["status"] for attempt in status["attempts"]] == ["failed", "success"]


@pytest.mark.asyncio
async def test_bulk_target_run_task_skips_completed_target(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "input" / "case_b" / "atom_2"
    output_root_dir = tmp_path / "output"
    _write_input_case(input_dir)

    target_output_root = output_root_dir / "case_b" / "atom_2"
    success_dir = target_output_root / "attempt_003"
    success_dir.mkdir(parents=True, exist_ok=True)
    energy_log_file = success_dir / "energy_log.json"
    energy_log_file.write_text(json.dumps({"energy_final": -10.0}))
    (target_output_root / "target_status.json").write_text(
        json.dumps(
            {
                "status": "success",
                "latest_attempt": 3,
                "latest_output_dir": str(success_dir),
                "latest_job_id": "job-0003",
                "energy_final": -10.0,
                "energy_log_file": str(energy_log_file),
                "attempts": [{"attempt": 3, "status": "success"}],
            }
        )
    )

    monkeypatch.setattr(bulk_target_run_module, "get_run_logger", lambda: _FakeLogger())
    monkeypatch.setattr(
        bulk_target_run_module.HPCProfileBlock,
        "load",
        staticmethod(lambda _: (_ for _ in ()).throw(AssertionError("HPC block should not be loaded"))),
    )
    monkeypatch.setattr(
        bulk_target_run_module,
        "run_job_from_blocks",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Job should not be submitted")),
    )

    result = await bulk_target_run_task.fn(
        target_name="case_b__atom_2",
        mode="ext_sqd",
        input_dir=str(input_dir),
        relative_path="case_b/atom_2",
        output_root_dir=str(output_root_dir),
        count_dict_filename="count_dict.txt",
        fcidump_filename="fci_dump.txt",
        command_block_name="cmd-gb-sqd-ext",
        execution_profile_block_name="exec-gb-sqd-ext-fugaku",
        hpc_profile_block_name="hpc-fugaku-gb-sqd",
        max_jobs_in_queue=2,
        queue_limit_scope="user_queue",
        queue_poll_interval_seconds=30.0,
        job_name_prefix="gbsqd-bulk",
        skip_completed=True,
        max_attempts=2,
        job_parameters=_ext_job_parameters(),
        parameter_overrides={"max_time": 600.0},
    )

    assert result["status"] == "skipped"
    assert result["latest_attempt"] == 3
    assert result["latest_job_id"] == "job-0003"


@pytest.mark.asyncio
async def test_bulk_target_run_task_raises_non_retryable_for_missing_input_file(
    tmp_path: Path, monkeypatch
):
    input_dir = tmp_path / "input" / "case_c" / "atom_3"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "count_dict.txt").write_text("count")

    monkeypatch.setattr(bulk_target_run_module, "get_run_logger", lambda: _FakeLogger())

    with pytest.raises(NonRetryableBulkError, match="FCIDUMP file not found"):
        await bulk_target_run_task.fn(
            target_name="case_c__atom_3",
            mode="ext_sqd",
            input_dir=str(input_dir),
            relative_path="case_c/atom_3",
            output_root_dir=str(tmp_path / "output"),
            count_dict_filename="count_dict.txt",
            fcidump_filename="fci_dump.txt",
            command_block_name="cmd-gb-sqd-ext",
            execution_profile_block_name="exec-gb-sqd-ext-fugaku",
            hpc_profile_block_name="hpc-fugaku-gb-sqd",
            max_jobs_in_queue=2,
            queue_limit_scope="user_queue",
            queue_poll_interval_seconds=30.0,
            job_name_prefix="gbsqd-bulk",
            skip_completed=True,
            max_attempts=2,
            job_parameters=_ext_job_parameters(),
        )


@pytest.mark.asyncio
async def test_bulk_target_run_task_runs_on_miyabi_without_fugaku_queue_gate(
    tmp_path: Path, monkeypatch
):
    input_dir = tmp_path / "input" / "case_m" / "atom_4"
    output_root_dir = tmp_path / "output"
    _write_input_case(input_dir)

    monkeypatch.setattr(bulk_target_run_module, "get_run_logger", lambda: _FakeLogger())
    monkeypatch.setattr(
        bulk_target_run_module.HPCProfileBlock,
        "load",
        staticmethod(lambda _: SimpleNamespace(queue_cpu="regular-c", hpc_target="miyabi")),
    )

    async def fake_wait_for_queue_slot(**kwargs):
        raise AssertionError("Miyabi path should not call Fugaku queue throttling")

    run_calls: list[dict[str, object]] = []

    async def fake_run_job_from_blocks(**kwargs):
        run_calls.append(kwargs)
        energy_log_file = Path(kwargs["work_dir"]) / "energy_log.json"
        energy_log_file.write_text(json.dumps({"energy_final": -7.5}))
        return SimpleNamespace(exit_status=0, job_id="12345.miyabi", job_status={"Exit_status": "0"})

    monkeypatch.setattr(bulk_target_run_module, "wait_for_queue_slot", fake_wait_for_queue_slot)
    monkeypatch.setattr(bulk_target_run_module, "run_job_from_blocks", fake_run_job_from_blocks)

    result = await bulk_target_run_task.fn(
        target_name="case_m__atom_4",
        mode="ext_sqd",
        input_dir=str(input_dir),
        relative_path="case_m/atom_4",
        output_root_dir=str(output_root_dir),
        count_dict_filename="count_dict.txt",
        fcidump_filename="fci_dump.txt",
        command_block_name="cmd-gb-sqd-ext",
        execution_profile_block_name="exec-gb-sqd-ext-miyabi",
        hpc_profile_block_name="hpc-miyabi-gb-sqd",
        max_jobs_in_queue=2,
        queue_limit_scope="user_queue",
        queue_poll_interval_seconds=30.0,
        job_name_prefix="gbsqd-bulk",
        skip_completed=True,
        max_attempts=1,
        job_parameters=_ext_job_parameters(),
    )

    assert result["status"] == "success"
    assert result["latest_job_id"] == "12345.miyabi"
    assert len(run_calls) == 1
    assert run_calls[0]["script_filename"] == "gb_sqd_ext.pbs"
    assert run_calls[0]["fugaku_job_name"] is None
