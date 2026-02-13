from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_adapters.base.jinja_env import make_env

_ENV = make_env("hpc_prefect_adapters.miyabi")
_TEMPLATE = "batch.pbs.j2"


@dataclass(frozen=True)
class MiyabiJobRequest:
    queue_name: str
    project: str
    executable: str


def to_miyabi_template_kwargs(*, exec_profile: ExecutionProfile, req: MiyabiJobRequest) -> dict:
    kw: dict = {
        "queue_name": req.queue_name,
        "project": req.project,
        "num_nodes": exec_profile.num_nodes,
        "launcher": exec_profile.launcher,
        "executable": req.executable,
    }
    if exec_profile.mpiprocs is not None:
        kw["mpiprocs"] = exec_profile.mpiprocs
    if exec_profile.ompthreads is not None:
        kw["ompthreads"] = exec_profile.ompthreads
    if exec_profile.walltime is not None:
        kw["walltime"] = exec_profile.walltime
    if exec_profile.modules:
        kw["modules"] = list(exec_profile.modules)
    if exec_profile.environments:
        kw["environments"] = dict(exec_profile.environments)
    if exec_profile.mpi_options:
        kw["mpi_options"] = list(exec_profile.mpi_options)
    if exec_profile.arguments:
        kw["arguments"] = list(exec_profile.arguments)
    return kw


def render_script(*, work_dir: Path, exec_profile: ExecutionProfile, req: MiyabiJobRequest) -> str:
    template = _ENV.get_template(_TEMPLATE)
    kwargs = to_miyabi_template_kwargs(exec_profile=exec_profile, req=req)
    return template.render(work_dir=str(work_dir), **kwargs)


def write_script_file(*, work_dir: Path, filename: str, text: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / filename
    path.write_text(text)
    return path