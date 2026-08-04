# QCSC Prefect Executor

Prefect executor integrations for local processes and QCSC HPC scheduler workflows.

## Local execution

Set `HPCProfileBlock.hpc_target` to `"local"` and map the command's executable
key to a path or command available on the Prefect worker. Local execution does
not generate a job script and does not call a scheduler. It invokes the command
directly with `cwd=work_dir`.

With `launcher="single"`, the command is:

```text
executable default_args... user_args...
```

With another launcher, it is:

```text
launcher mpi_options... executable default_args... user_args...
```

`environments` are merged into the worker process environment. Values are
passed literally without shell expansion. `modules` and `pre_commands` are
explicitly unsupported for local execution and cause `ValueError` before the
process starts.

## Prefect Cloud log policy

Use `CloudLogPolicy` to control how scheduler stdout/stderr and metrics
artifacts are sent to Prefect Cloud. Scheduler log files remain in the job's
shared-filesystem working directory; the policy only changes Cloud output.

| Mode | Cloud logs | Default artifact behavior |
| --- | --- | --- |
| `legacy` | Historical first 10,000 characters of each stream | Preserves each API's historical behavior |
| `none` | No job result logs | Disabled |
| `summary` | Job ID, state, exit code, elapsed time, node, log paths, and bounded tails | Disabled |
| `tail` | The final `tail_lines` of each stream | Disabled |
| `full` | Complete stdout/stderr | Disabled |

`legacy` is the library default for backward compatibility. For single-job
APIs it retains the existing metrics artifact; for bulk APIs it retains the
existing behavior of producing no per-job logs or artifacts. Set
`create_artifact=True` to opt a non-legacy policy into an artifact, or set it to
`False` to suppress a legacy artifact.

Prefect Cloud counts logs and artifact-related events against workspace plan
limits. `summary` is the recommended explicit choice for a high-volume Hobby
workflow because it sends at most one compact summary and one stderr-tail log
per status transition. Repeated bulk polls of an unchanged state emit nothing.
`full` should be reserved for small diagnostic jobs; keep the authoritative
complete logs on the shared filesystem.

ROQUO callers should select `summary` explicitly rather than relying on a
future library default:

```python
from qcsc_prefect_executor import CloudLogPolicy, run_jobs_from_blocks_bulk

cloud_logs = CloudLogPolicy(
    mode="summary",
    tail_lines=20,
    create_artifact=False,
)

result = await run_jobs_from_blocks_bulk(
    jobs=jobs,
    command_block="command-block",
    execution_profile_block="slurm-cpu",
    hpc_profile_block="roquo-slurm",
    registry_path=registry_path,
    cloud_log_policy=cloud_logs,
)
```

The same `cloud_log_policy` argument is accepted by `run_job_from_blocks`,
`submit_job_from_blocks`, `monitor_jobs_many`, and `GlobalFugakuBulkRunner`.

## Bulk execution

`BulkJobSpec` supports optional per-job `execution_profile_block` and
`hpc_profile_block` overrides. `None` uses the runner/API default blocks. The
single-submit bulk paths use these effective blocks for submission and group
monitoring by effective `hpc_profile_block`; native PJM bulk mode rejects per-job
block overrides because one generated script/profile is shared by all subjobs.

### Bounded Slurm submission

For a Slurm runner using one common HPC/execution profile scope,
`run_jobs_from_blocks_bulk` creates its Slurm queue probe when `queue_probe` is
omitted:

```python
result = await run_jobs_from_blocks_bulk(
    jobs=jobs,
    command_block="command-block",
    execution_profile_block="slurm-cpu",
    hpc_profile_block="slurm-site",
    registry_path=registry_path,
    max_active_jobs=50,
    safety_margin=5,
    max_submit_per_refill=10,
    slurm_user="scheduler-user",
)
```

`max_active_jobs` is the configured workflow ceiling; it is not a Slurm quota
discovered from `squeue`. The default probe filters by `slurm_user` (or the
current process user), account/project, and partition resolved from the blocks.
For each refill, the effective allowance is:

```text
min(
    configured_max_active_jobs - scoped_current_active_jobs - safety_margin,
    max_submit_per_refill,
    remaining_submit_candidates,
)
```

Negative allowance is clamped to zero. Scheduler errors, timeouts, and malformed
probe output fail closed with zero submissions for that refill. If per-job block
overrides span different user/account/partition scopes, pass an explicit
composite `QueueProbe` whose capacity contract covers those scopes.

### Explicit Slurm cancellation

Stopping a Prefect run or cancelling its waiting coroutine does not cancel the
external Slurm allocation. Persist an explicit operator intent in the bulk
registry, then run the cancel executor:

```python
from pathlib import Path

from qcsc_prefect_executor.bulk import BulkJobRegistry, execute_cancel_requests

registry = BulkJobRegistry(Path("/shared/work/bulk.sqlite"))
registry.request_cancel(
    "logical-job-key",
    requested_by="operator-identity",
    reason="operator-approved cancellation",
)
await execute_cancel_requests(
    registry=registry,
    hpc_profile_block="hpc-slurm",
)
```

`request_cancel()` is atomic and idempotent: the first request preserves its
actor, timestamp, and reason. A pending job with no scheduler side effect is
cancelled locally. For a known Slurm job ID, a durable dispatch claim permits
at most one automatic `scancel` call. Accepted, already-terminal, not-found,
temporary-failure, and rejected outcomes are recorded separately. Ambiguous
`DISPATCHING` outcomes are not retried automatically. The low-level
`SlurmRuntime.cancel()` primitive rejects calls that do not explicitly confirm
that durable intent has already been recorded.
