"""
Prefect tasks: generate PBS script and submit via qsub.
"""

from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

from prefect import task

from .blocks import CommandBlock, ExecutionProfileBlock, MiyabiHPCProfileBlock
from .models import Tuning
from .pbs import write_pbs_script
from .resolver import resolve_run


@task
def generate_script(
    *,
    work_root: str,
    job_name: str,
    cmd: CommandBlock,
    profile: ExecutionProfileBlock,
    hpc: MiyabiHPCProfileBlock,
    tuning: Tuning | None = None,
    user_args: list[str] | None = None,
) -> str:
    resolved = resolve_run(cmd=cmd, profile=profile, hpc=hpc, tuning=tuning, user_args=user_args)
    script_path = write_pbs_script(resolved, out_dir=Path(work_root), job_name=job_name)
    return str(script_path)


@task
def submit_script(script_path: str) -> str:
    if shutil.which("qsub") is None:
        raise RuntimeError("qsub was not found in PATH. Run on Miyabi login node where PBS is available.")
    proc = subprocess.run(["qsub", script_path], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"qsub failed: {proc.stderr.strip()}")
    return proc.stdout.strip()
