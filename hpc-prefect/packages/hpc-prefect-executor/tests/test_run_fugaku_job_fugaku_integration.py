from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from hpc_prefect_adapters.fugaku.builder import FugakuJobRequest
from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_executor.fugaku import run as run_mod


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_work_dir(tmp_path: Path) -> Path:
    raw = os.getenv("FUGAKU_TEST_WORK_DIR", "").strip()
    if not raw:
        return tmp_path
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.mark.fugaku_integration
def test_run_fugaku_job_real_pjsub(tmp_path: Path, monkeypatch):
    if not _env_enabled("FUGAKU_INTEGRATION"):
        pytest.skip("Set FUGAKU_INTEGRATION=1 to run Fugaku pjsub integration test.")
    if shutil.which("pjsub") is None or shutil.which("pjstat") is None:
        pytest.skip("pjsub/pjstat not found. Run on Fugaku login node.")

    queue_name = os.getenv("FUGAKU_RSCGRP")
    project = os.getenv("FUGAKU_PROJECT")
    if not queue_name or not project:
        pytest.skip("Set FUGAKU_RSCGRP and FUGAKU_PROJECT for Fugaku integration test.")

    timeout_seconds = int(os.getenv("FUGAKU_TEST_TIMEOUT", "900"))
    gfscache = os.getenv("FUGAKU_GFSCACHE", "/vol0002")
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
    executable = work_dir / "hello_fugaku.sh"
    executable.write_text("#!/bin/sh\necho fugaku-integration-ok\nhostname\n")
    executable.chmod(0o755)

    profile = ExecutionProfile(
        command_key="fugaku-integration",
        num_nodes=1,
        launcher="single",
        walltime="00:05:00",
    )
    req = FugakuJobRequest(
        queue_name=queue_name,
        project=project,
        executable=str(executable),
        job_name="fugaku-integration",
        gfscache=gfscache,
    )

    result = asyncio.run(
        run_mod.run_fugaku_job(
            work_dir=work_dir,
            script_filename="integration_job.pjm",
            exec_profile=profile,
            req=req,
            watch_poll_interval=10.0,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key="fugaku-integration-metrics",
        )
    )

    assert result.job_id
    assert result.state in {"EXT", "CCL"}
    assert isinstance(result.exit_status, int)
    assert len(artifact_calls) == 1
