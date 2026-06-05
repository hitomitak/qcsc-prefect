# Queue-Aware Bulk HPC Jobs

Use the Bulk HPC API when a workflow needs to submit and monitor a large batch
of independent scheduler jobs without creating one Prefect task per scheduler
job. The API keeps the existing block separation:

- `CommandBlock`: what to run
- `ExecutionProfileBlock`: how to run
- `HPCProfileBlock`: where to run

`run_job_from_blocks()` remains the single-job API. It submits one scheduler
job and waits for that job to finish. `run_jobs_from_blocks_bulk()` manages a
pool of `BulkJobSpec` records, submits only when queue capacity is available,
monitors active jobs in batches, and refills the scheduler queue as jobs leave
the active set.

## Registry and Restart Safety

Bulk state is persisted in a local SQLite registry. Each `BulkJobSpec.job_key`
is an idempotency key, so it should be stable and unique for one logical job.
After a restart, already completed jobs are not resubmitted, and submitted or
running jobs are monitored again from their scheduler job IDs.

`SUBMIT_DEFERRED` means submission was attempted but should be retried later,
for example because the scheduler queue was full or temporarily unavailable. It
is not a terminal status.

## Queue-Aware Refill

The bulk loop uses a `QueueProbe` and `QueueAwareSubmitGate` to decide how many
pending jobs may be submitted in a refill cycle. If the probe cannot safely
determine capacity, the gate returns zero and the loop waits for a later cycle.
Queue-full errors are recorded as `SUBMIT_DEFERRED`, not `FAILED`.

For Fugaku-like PJM systems, use `FugakuQueueProbe` or let the bulk API create
the default Fugaku probe from the `HPCProfileBlock` and `ExecutionProfileBlock`.
For other schedulers, pass an explicit scheduler-specific `QueueProbe`.

## Optional Fugaku Native Bulk Mode

The default and recommended integration path is `submit_mode="single"`, which
submits one scheduler job per logical `BulkJobSpec`. Fugaku native PJM bulk
submission through `pjsub --bulk --sparam` is available as an experimental,
opt-in mode. It is not used unless `submit_mode="native_bulk"` is passed to
`run_jobs_from_blocks_bulk()`.

Use native bulk only when you specifically want multiple logical jobs submitted
as PJM subjobs in fewer scheduler calls. The registry still tracks each
`BulkJobSpec` independently, and native bulk metadata remains nullable for
backward compatibility with single-submit registries.

Native bulk mode uses logical subjob slots for queue capacity. When creating a
Fugaku probe yourself, set `capacity_mode="native_bulk"` so `pjstat --limit`
prefers `ru-accept-bulksubjob`, then `ru-accept-allsubjob`, then `ru-accept`.
`max_bulk_group_size` only controls how many logical jobs go into one
`pjsub --bulk` call; it does not increase queue allowance.

Each native bulk group gets a manifest directory:

```text
<bulk_group_dir>/manifests/0.json
<bulk_group_dir>/manifests/1.json
...
```

The generated Fugaku script selects the manifest with `PJM_BULKNUM`:

```sh
MANIFEST="${QCSC_BULK_MANIFEST_DIR}/${PJM_BULKNUM}.json"
export QCSC_BULK_MANIFEST="${MANIFEST}"
export QCSC_BULK_NUM="${PJM_BULKNUM}"
```

The command block should define a generic application command that reads
`$QCSC_BULK_MANIFEST`, for example:

```sh
python -m your_app.batch_entry --manifest "$QCSC_BULK_MANIFEST"
```

For Fugaku `EXT` means the scheduler says the subjob exited; it is not enough
to mark a logical job `SUCCEEDED`. Configure `expected_outputs` on each
`BulkJobSpec`; the registry marks the job `SUCCEEDED` only when all expected
outputs exist. Workflow-level rerun and application-specific recovery should be
handled by the calling workflow. qcsc-prefect provides stable `job_key`
idempotency and registry reset primitives.

## Waves

`wave_id` is registry metadata for downstream workflows. It is not the submit
unit. `run_jobs_from_blocks_bulk()` treats all jobs as one pending pool and does
not submit wave by wave. Use registry methods such as `is_wave_ready()` or
`get_ready_waves()` when downstream work needs to wait for all jobs in a wave.

## Minimal Example

```python
from pathlib import Path

from qcsc_prefect_adapters.fugaku.queue import FugakuQueueProbe
from qcsc_prefect_executor.from_blocks import run_jobs_from_blocks_bulk
from qcsc_prefect_executor.bulk import BulkJobSpec


jobs = [
    BulkJobSpec(
        job_key=f"batch-{index:04d}",
        work_dir=Path("work") / f"batch-{index:04d}",
        command_args={"index": index},
        wave_id="wave-0",
        expected_outputs=[Path("done.marker")],
    )
    for index in range(1000)
]

result = await run_jobs_from_blocks_bulk(
    jobs=jobs,
    command_block="cmd-large-batch",
    execution_profile_block="exec-large-batch",
    hpc_profile_block="hpc-fugaku",
    registry_path=Path("work") / "bulk-jobs.sqlite",
    queue_probe=FugakuQueueProbe(project="your-group"),
    max_active_jobs=1000,
    safety_margin=20,
    max_submit_per_refill=100,
    poll_interval_seconds=60,
    refill_interval_seconds=60,
)

print(result.status_counts)
```

## Experimental Native Bulk Example

```python
from pathlib import Path

from qcsc_prefect_adapters.fugaku.queue import FugakuQueueProbe
from qcsc_prefect_executor.bulk import BulkJobSpec
from qcsc_prefect_executor.from_blocks import run_jobs_from_blocks_bulk


jobs = [
    BulkJobSpec(
        job_key=f"qpy-{index:04d}",
        stage_id="qpy",
        work_dir=Path("work") / f"qpy-{index:04d}",
        command_args={"index": index},
        expected_outputs=[Path("done.marker")],
    )
    for index in range(1000)
]

result = await run_jobs_from_blocks_bulk(
    jobs=jobs,
    command_block="cmd-manifest-batch",
    execution_profile_block="exec-fugaku-bulk",
    hpc_profile_block="hpc-fugaku",
    registry_path=Path("work") / "bulk-jobs.sqlite",
    queue_probe=FugakuQueueProbe(
        project="your-group",
        capacity_mode="native_bulk",
    ),
    submit_mode="native_bulk",
    initial_submit_count=200,
    max_submit_per_refill=100,
    max_bulk_group_size=50,
    poll_interval_seconds=60,
    refill_interval_seconds=60,
)

print(result.status_counts)
```
