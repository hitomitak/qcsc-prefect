from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_adapters.base.jinja_env import make_env

_ENV = make_env("hpc_prefect_adapters.fugaku")
_TEMPLATE = "batch.pjm.j2"


@dataclass(frozen=True)
class FugakuJobRequest:
    queue_name: str
    project: str
    executable: str
    job_name: str = "prefect_job"
    gfscache: str | None = None
    mpi_options_for_pjm: list[str] | None = None
    spack_modules: list[str] | None = None


def to_fugaku_template_kwargs(
    *,
    work_dir: Path,
    exec_profile: ExecutionProfile,
    req: FugakuJobRequest,
    script_basename: str = "batch.pjm",
) -> dict:
    stdout_path = work_dir / f"{script_basename}.{req.job_name}.out"
    stderr_path = work_dir / f"{script_basename}.{req.job_name}.err"
    stat_path = work_dir / f"{script_basename}.{req.job_name}.stats"

    kw: dict = {
        "resource_group": req.queue_name,
        "group_name": req.project,
        "job_name": req.job_name,
        "num_nodes": exec_profile.num_nodes,
        "elapse_time": exec_profile.walltime,
        "launcher": exec_profile.launcher,
        "executable": req.executable,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stat_path": str(stat_path),
    }

    if req.mpi_options_for_pjm:
        kw["mpi_options_for_pjm"] = list(req.mpi_options_for_pjm)
    if req.gfscache:
        kw["gfscache"] = req.gfscache
    if req.spack_modules:
        kw["spack_modules"] = list(req.spack_modules)
    if exec_profile.environments:
        kw["environments"] = dict(exec_profile.environments)
    if exec_profile.mpi_options:
        kw["mpi_options"] = list(exec_profile.mpi_options)
    if exec_profile.arguments:
        kw["arguments"] = list(exec_profile.arguments)
    return kw


def render_script(
    *,
    work_dir: Path,
    exec_profile: ExecutionProfile,
    req: FugakuJobRequest,
    script_basename: str = "batch.pjm",
) -> str:
    template = _ENV.get_template(_TEMPLATE)
    kwargs = to_fugaku_template_kwargs(
        work_dir=work_dir,
        exec_profile=exec_profile,
        req=req,
        script_basename=script_basename,
    )
    return template.render(**kwargs)


def write_script_file(*, work_dir: Path, filename: str, text: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / filename
    path.write_text(text)
    return path
