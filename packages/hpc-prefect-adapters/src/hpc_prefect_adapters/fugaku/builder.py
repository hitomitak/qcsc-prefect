from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from hpc_prefect_core.models.execution_profile import ExecutionProfile
from hpc_prefect_adapters.base.jinja_env import make_env

_ENV = make_env("hpc_prefect_adapters.fugaku")
_TEMPLATE = "batch.pjm.j2"


@dataclass(frozen=True)
class FugakuJobRequest:
    """Target-specific request fields required to build a Fugaku PJM job."""

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
    """Build template variables for the Fugaku PJM script.

    Args:
        work_dir: Base working directory used to derive output paths.
        exec_profile: Scheduler-independent execution profile.
        req: Fugaku-specific scheduler request fields.
        script_basename: Script basename used for output/stat filenames.

    Returns:
        A dictionary that can be passed to the Fugaku Jinja template.
    """

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
        "mpi_options_for_pjm": list(req.mpi_options_for_pjm or []),
        "gfscache": req.gfscache,
        "spack_modules": list(req.spack_modules or []),
        "environments": dict(exec_profile.environments or {}),
        "mpi_options": list(exec_profile.mpi_options or []),
        "arguments": list(exec_profile.arguments or []),
    }
    return kw


def render_script(
    *,
    work_dir: Path,
    exec_profile: ExecutionProfile,
    req: FugakuJobRequest,
    script_basename: str = "batch.pjm",
) -> str:
    """Render a Fugaku job script text from the configured Jinja template.

    .. note::
        The template file is configured by module constant ``_TEMPLATE`` and
        is expected to be a ``.j2`` file.

    Args:
        work_dir: Base working directory used in template variables.
        exec_profile: Scheduler-independent execution profile.
        req: Fugaku-specific scheduler request fields.
        script_basename: Script basename used for output/stat filenames.

    Returns:
        Rendered PJM script text.
    """

    template = _ENV.get_template(_TEMPLATE)
    kwargs = to_fugaku_template_kwargs(
        work_dir=work_dir,
        exec_profile=exec_profile,
        req=req,
        script_basename=script_basename,
    )
    return template.render(**kwargs)


def write_script_file(*, work_dir: Path, filename: str, text: str) -> Path:
    """Write a rendered job script into the work directory.

    .. note::
        This function is expected to be called inside
        :func:`hpc_prefect_executor.fugaku.run.run_fugaku_job`.
        Workflow authors normally do not need to call it directly.

    .. note::
        The ``text`` argument is expected to come from :func:`render_script`,
        which renders the ``.j2`` template specified by ``_TEMPLATE``.

    Args:
        work_dir: Base working directory where the script file is created.
        filename: Script file name (for example ``batch.pjm``).
        text: Rendered script text.

    Returns:
        Absolute path to the created job script file.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / filename
    path.write_text(text)
    return path
