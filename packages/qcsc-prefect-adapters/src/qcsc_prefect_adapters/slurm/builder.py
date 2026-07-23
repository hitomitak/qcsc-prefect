from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from qcsc_prefect_adapters.base.jinja_env import make_env
from qcsc_prefect_core.models.execution_profile import ExecutionProfile

_ENV = make_env("qcsc_prefect_adapters.slurm")
_TEMPLATE = "batch.slurm.j2"

# This is intentionally a conservative library default, not a claim about every
# Slurm installation's configured MaxJobName.  Operators must confirm their
# cluster's limit before relying on a longer value.
DEFAULT_SLURM_JOB_NAME_MAX_LENGTH = 64
DEFAULT_SLURM_JOB_NAME_PREFIX = "qcsc"
_IDENTITY_SCHEMA_VERSION = "qcsc-prefect-slurm-identity-v1"
_IDENTITY_DIGEST_SUFFIX_LENGTH = 24
_JOB_NAME_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SLURM_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SLURM_COMMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


@dataclass(frozen=True)
class SlurmJobIdentity:
    """Deterministic scheduler identity used to rediscover a Slurm job.

    ``job_name`` is short enough for the conservative default length while
    preserving a digest suffix. ``comment`` holds the complete digest used for
    later identity matching.
    """

    job_name: str
    comment: str


def _normalized_job_name_component(value: str) -> str:
    normalized = _JOB_NAME_COMPONENT_RE.sub("-", value).strip(".-")
    return normalized or "job"


def build_slurm_job_identity(
    *,
    job_key: str,
    spec_hash: str,
    prefix: str = DEFAULT_SLURM_JOB_NAME_PREFIX,
    max_job_name_length: int = DEFAULT_SLURM_JOB_NAME_MAX_LENGTH,
) -> SlurmJobIdentity:
    """Build a deterministic, scheduler-safe identity for one logical job.

    The name contains readable normalized ``prefix``/``job_key`` text followed
    by a fixed 96-bit digest suffix.  Truncation occurs before the suffix, so
    long or similar keys cannot remove the collision-resistant portion.  The
    comment stores the complete SHA-256 identity digest without exposing either
    input string.

    Args:
        job_key: Stable logical job key.
        spec_hash: Canonical immutable resolved-spec hash.
        prefix: Human-readable name prefix.
        max_job_name_length: Target-approved maximum job-name length. The
            default is deliberately conservative.

    Returns:
        A scheduler-safe job name and a full-digest Slurm comment.

    Raises:
        ValueError: If an input is empty or the requested length cannot retain
            the digest suffix.
    """

    normalized_job_key = str(job_key).strip()
    normalized_spec_hash = str(spec_hash).strip()
    if not normalized_job_key:
        raise ValueError("job_key must be non-empty when building Slurm identity.")
    if not normalized_spec_hash:
        raise ValueError("spec_hash must be non-empty when building Slurm identity.")

    max_length = int(max_job_name_length)
    suffix_length = _IDENTITY_DIGEST_SUFFIX_LENGTH + 1
    if max_length <= suffix_length:
        raise ValueError(
            "max_job_name_length must leave room for the Slurm identity digest suffix."
        )

    digest_input = "\0".join((_IDENTITY_SCHEMA_VERSION, normalized_job_key, normalized_spec_hash))
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    suffix = f"-{digest[:_IDENTITY_DIGEST_SUFFIX_LENGTH]}"

    readable_prefix = "-".join(
        (
            _normalized_job_name_component(str(prefix)),
            _normalized_job_name_component(normalized_job_key),
        )
    )
    job_name = readable_prefix[: max_length - len(suffix)].rstrip(".-") + suffix
    return SlurmJobIdentity(
        job_name=job_name,
        comment=f"{_IDENTITY_SCHEMA_VERSION}:sha256:{digest}",
    )


def _validated_slurm_directive_value(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    pattern = _SLURM_JOB_NAME_RE if field_name == "job_name" else _SLURM_COMMENT_RE
    if not pattern.fullmatch(normalized):
        raise ValueError(f"Slurm {field_name} must use only safe scheduler directive characters.")
    return normalized


@dataclass(frozen=True)
class SlurmJobRequest:
    """Target-specific request fields required to build a Slurm batch job.

    Attributes:
        partition: Slurm partition name passed to ``#SBATCH --partition``.
        executable: Absolute or scheduler-visible command path to execute.
        account: Optional Slurm account passed to ``#SBATCH --account``.
        qpu: Optional QPU resource selector emitted by the Slurm template.
        memory: Optional memory request passed to ``#SBATCH --mem``.
        ntasks: Optional task count passed to ``#SBATCH --ntasks``.
        job_name: Optional scheduler-safe name passed to ``#SBATCH --job-name``.
        comment: Optional scheduler-safe identity passed to ``#SBATCH --comment``.
    """

    partition: str
    executable: str
    account: str | None = None
    qpu: str | None = None
    memory: str | None = None
    ntasks: int | None = None
    job_name: str | None = None
    comment: str | None = None


def to_slurm_template_kwargs(*, exec_profile: ExecutionProfile, req: SlurmJobRequest) -> dict:
    """Build template variables for the Slurm job script.

    Args:
        exec_profile: Scheduler-independent execution profile.
        req: Slurm-specific scheduler request fields.

    Returns:
        A dictionary that can be passed to the Slurm Jinja template.
    """

    kw: dict = {
        "partition": req.partition,
        "executable": req.executable,
        "num_nodes": exec_profile.num_nodes,
        "launcher": exec_profile.launcher,
    }
    if req.account:
        kw["account"] = req.account
    if req.qpu:
        kw["qpu"] = req.qpu
    if req.memory:
        kw["memory"] = req.memory
    if req.ntasks is not None:
        kw["ntasks"] = req.ntasks
    if req.job_name is not None:
        kw["job_name"] = _validated_slurm_directive_value(
            req.job_name,
            field_name="job_name",
        )
    if req.comment is not None:
        kw["comment"] = _validated_slurm_directive_value(
            req.comment,
            field_name="comment",
        )
    if exec_profile.mpiprocs is not None:
        kw["mpiprocs"] = exec_profile.mpiprocs
    if exec_profile.ompthreads is not None:
        kw["ompthreads"] = exec_profile.ompthreads
    if exec_profile.walltime is not None:
        kw["walltime"] = exec_profile.walltime
    if exec_profile.modules:
        kw["modules"] = list(exec_profile.modules)
    if exec_profile.pre_commands:
        kw["pre_commands"] = list(exec_profile.pre_commands)
    if exec_profile.environments:
        kw["environments"] = dict(exec_profile.environments)
    if exec_profile.mpi_options:
        kw["mpi_options"] = list(exec_profile.mpi_options)
    if exec_profile.arguments:
        kw["arguments"] = list(exec_profile.arguments)
    return kw


def render_script(*, work_dir: Path, exec_profile: ExecutionProfile, req: SlurmJobRequest) -> str:
    """Render Slurm job script text from the configured Jinja template.

    Args:
        work_dir: Working directory injected into the template.
        exec_profile: Scheduler-independent execution profile.
        req: Slurm-specific scheduler request fields.

    Returns:
        Rendered Slurm script text.
    """

    template = _ENV.get_template(_TEMPLATE)
    kwargs = to_slurm_template_kwargs(exec_profile=exec_profile, req=req)
    return template.render(work_dir=str(work_dir), **kwargs)


def write_script_file(*, work_dir: Path, filename: str, text: str) -> Path:
    """Write a rendered Slurm script into the work directory.

    Args:
        work_dir: Base working directory where the script file is created.
        filename: Script file name, for example ``batch.slurm``.
        text: Rendered script text.

    Returns:
        Absolute path to the created job script file.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / filename
    path.write_text(text)
    return path
