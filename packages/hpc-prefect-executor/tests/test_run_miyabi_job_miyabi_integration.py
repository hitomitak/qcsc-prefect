from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from hpc_prefect_adapters.miyabi.builder import MiyabiJobRequest
from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_executor.miyabi import run as run_mod


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_work_dir(tmp_path: Path) -> Path:
    raw = os.getenv("MIYABI_TEST_WORK_DIR", "").strip()
    if not raw:
        return tmp_path
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.mark.miyabi_integration
def test_run_miyabi_job_real_qsub(tmp_path: Path, monkeypatch):
    if not _env_enabled("MIYABI_INTEGRATION"):
        pytest.skip("Set MIYABI_INTEGRATION=1 to run Miyabi qsub integration test.")
    if shutil.which("qsub") is None or shutil.which("qstat") is None:
        pytest.skip("qsub/qstat not found. Run on Miyabi login node.")

    queue_name = os.getenv("MIYABI_PBS_QUEUE")
    project = os.getenv("MIYABI_PBS_PROJECT")
    if not queue_name or not project:
        pytest.skip("Set MIYABI_PBS_QUEUE and MIYABI_PBS_PROJECT for Miyabi integration test.")

    timeout_seconds = int(os.getenv("MIYABI_TEST_TIMEOUT", "600"))
    artifact_calls: list[dict[str, Any]] = []

    class _LoggerStub:
        def info(self, message: str) -> None:  # pragma: no cover
            pass

        def error(self, message: str) -> None:  # pragma: no cover
            pass

    async def fake_create_table_artifact(*, table: list[list[Any]], key: str) -> None:
        artifact_calls.append({"table": table, "key": key})

    monkeypatch.setattr(run_mod, "get_run_logger", lambda: _LoggerStub())
    monkeypatch.setattr(run_mod, "create_table_artifact", fake_create_table_artifact)

    work_dir = _resolve_work_dir(tmp_path)
    executable = work_dir / "hello.sh"
    executable.write_text("#!/bin/sh\necho miyabi-integration-ok\n")
    executable.chmod(0o755)

    profile = ExecutionProfile(
        command_key="integration",
        num_nodes=1,
        launcher="single",
        walltime="00:05:00",
    )
    req = MiyabiJobRequest(
        queue_name=queue_name,
        project=project,
        executable=str(executable),
    )

    result = asyncio.run(
        run_mod.run_miyabi_job(
            work_dir=work_dir,
            script_filename="integration_job.pbs",
            exec_profile=profile,
            req=req,
            watch_poll_interval=5.0,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key="miyabi-integration-metrics",
        )
    )

    assert result.job_id
    assert result.exit_status == 0
    assert str(result.job_status.get("Exit_status", "")).strip() == "0"
    assert len(artifact_calls) == 1
