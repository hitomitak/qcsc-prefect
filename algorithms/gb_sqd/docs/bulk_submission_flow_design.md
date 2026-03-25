# GB-SQD Recursive Bulk Submission Flow Specification

## 1. Goal

Add a new GB-SQD flow that scans a user-specified parent directory, finds target directories that contain both `count_dict.txt` and `fci_dump.txt`, and runs one monolithic `gb-demo` job per target directory.

This flow is intended for Fugaku first.

## 2. Background

The current GB-SQD implementation already supports two execution styles:

- `ext_sqd_simple_flow` / `trim_sqd_simple_flow`
  - one Prefect flow run submits one `gb-demo` job and waits for completion
- `ext_sqd_flow` / `trim_sqd_flow`
  - one logical GB-SQD run is split into `init`, `recovery`, and `finalize` Prefect tasks

The new flow must use the monolithic execution style for each input directory.
It must not split `gb-demo` into `init` / `recovery` / `finalize` tasks.

## 3. Scope

In scope:

- recursive discovery of input directories
- one target directory = one Prefect task = one Fugaku job submission
- `gb-demo` runs to completion inside that single task
- retry / resubmission when a target task fails
- queue-aware throttling on Fugaku, including both running and waiting jobs
- final summary of all submitted targets

Out of scope for the first implementation:

- checkpoint resume inside one `gb-demo` run
- splitting one target into multiple Prefect tasks
- Miyabi-specific queue throttling
- changing the `run_job_from_blocks()` executor contract

## 4. Fixed Assumptions

- Inputs live under a shared filesystem visible from both the Prefect worker and Fugaku nodes.
- Generated outputs should not overwrite the input directories directly.
- The queue limit should count jobs already present in the target Fugaku queue, including jobs not submitted by the current flow run.
- The first implementation should continue processing other targets even if one target fails, then fail the flow at the end if any target still failed after retries.
- The intended Fugaku launch shape is one MPI rank per node via `mpirun -n <num_nodes>`.

## 4.1 Actual Fugaku runtime requirements

The user-provided reference scripts define the target runtime contract more precisely than the current generic GB-SQD examples.

Required scheduler/runtime characteristics:

- `#PJM -g <project>`
- `#PJM -L "node=<num_nodes>"`
- `#PJM -L "rscgrp=<queue>"`
- `#PJM -L "elapse=<walltime>"`
- `#PJM -L "freq=2000,eco_state=2"`
- `#PJM --mpi "max-proc-per-node=1"`
- `#PJM -x PJM_LLIO_GFSCACHE=/vol0004:/vol0002`
- `#PJM -s`
- `module load LLVM/llvmorg-21.1.0`
- `export OMP_NUM_THREADS=48`
- `export UTOFU_SWAP_PROTECT=1`
- `export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH`
- `mpirun -n <num_nodes> <gb-demo> ...`

These are not optional documentation examples.
They should be treated as the intended Fugaku baseline for this GB-SQD bulk flow.

## 5. Proposed Flow

### 5.1 Flow name

Primary API:

```python
@flow(name="GB-SQD-Bulk")
async def bulk_gb_sqd_flow(...)
```

Optional convenience wrappers:

- `bulk_ext_sqd_flow(...)`
- `bulk_trim_sqd_flow(...)`

The wrappers are thin aliases around `bulk_gb_sqd_flow(mode="ext_sqd")` and `bulk_gb_sqd_flow(mode="trim_sqd")`.

### 5.2 Reused blocks

This flow should reuse the existing simple-flow blocks created by `create_blocks.py`.

For `ext_sqd`:

- `CommandBlock`: `cmd-gb-sqd-ext`
- `ExecutionProfileBlock`: `exec-gb-sqd-ext-<target>`

For `trim_sqd`:

- `CommandBlock`: `cmd-gb-sqd-trim`
- `ExecutionProfileBlock`: `exec-gb-sqd-trim-<target>`

Shared:

- `HPCProfileBlock`: `hpc-<target>-gb-sqd`

No new executor block type is required.

However, the Fugaku path still needs targeted extensions.

- environment variables can already be passed through `ExecutionProfileBlock.environments`
- `gfscache` and PJM MPI options can already be passed through `HPCProfileBlock`
- the missing pieces are mainly:
  - additional PJM resource directives such as `freq=2000,eco_state=2`
  - Fugaku support for `module load ...`
  - GB-SQD flow signatures that expose all required CLI parameters

## 6. Proposed Parameters

### 6.1 Bulk-specific parameters

```python
mode: Literal["ext_sqd", "trim_sqd"]
input_root_dir: str
output_root_dir: str
count_dict_filename: str = "count_dict.txt"
fcidump_filename: str = "fci_dump.txt"
leaf_only: bool = True
skip_completed: bool = True
fail_fast: bool = False
max_jobs_in_queue: int = 10
queue_limit_scope: Literal["user_queue", "flow_jobs_only"] = "user_queue"
queue_poll_interval_seconds: float = 120.0
max_target_task_retries: int = 1
max_prefect_concurrency: int | None = None
job_name_prefix: str = "gbsqd-bulk"
command_block_name: str | None = None
execution_profile_block_name: str | None = None
hpc_profile_block_name: str = "hpc-fugaku-gb-sqd"
```

### 6.2 Mode-specific GB-SQD parameters

The flow should accept the same algorithm parameters that already exist in:

- `ext_sqd_simple_flow`
- `trim_sqd_simple_flow`

In addition, the flow must support the full parameter sets used in the reference Fugaku scripts.

ExtSQD-specific required parameters:

```python
num_samples_per_batch: int
num_batches: int
num_recovery: int
iteration: int
adet_comm_size: int
bdet_comm_size: int
task_comm_size: int
adet_comm_size_final: int
bdet_comm_size_final: int
task_comm_size_final: int
do_carryover_in_recovery: bool = True
carryover_ratio: float
carryover_threshold: float
max_time: float
with_hf: bool = False
verbose: bool = True
```

TrimSQD-specific required parameters:

```python
num_samples_per_recovery: int
num_batches: int
num_recovery: int
iteration: int
adet_comm_size: int
bdet_comm_size: int
task_comm_size: int
adet_comm_size_combined: int
bdet_comm_size_combined: int
task_comm_size_combined: int
adet_comm_size_final: int
bdet_comm_size_final: int
task_comm_size_final: int
carryover_ratio_batch: float
carryover_ratio_combined: float
carryover_threshold: float
max_time: float
with_hf: bool = False
verbose: bool = True
```

The bulk flow therefore cannot simply reuse the current simple-flow Python signatures unchanged.
Those signatures are missing some parameters that are required by the reference Fugaku scripts.

## 7. Discovery Contract

### 7.1 Target directory definition

A directory is a target when it contains both:

- `count_dict.txt`
- `fci_dump.txt`

The check should be based on exact file names by default, but the file names remain configurable.

### 7.2 Recursive scan rules

1. Start from `input_root_dir`.
2. Walk descendants recursively.
3. Collect directories that satisfy the target definition.
4. Sort the resulting relative paths lexicographically for deterministic execution order.

### 7.3 `leaf_only=True`

When `leaf_only=True`, a matched directory is ignored if one of its descendants also matches.
This matches the intended "go to the deepest directory" behavior.

### 7.4 No target found

If no target directories are found, the flow should raise `ValueError` and submit no jobs.

## 8. Per-Target Task Contract

Each discovered target directory becomes one Prefect task.

Task inputs:

- target input directory
- relative path from `input_root_dir`
- derived output directory
- selected GB-SQD mode and algorithm parameters
- block names

Task behavior:

1. Validate that both required files exist.
2. Derive the target output root from `output_root_dir / relative_target_path`.
3. If `skip_completed=True` and the latest recorded status is successful, skip submission.
4. Wait for a queue slot on Fugaku.
5. Submit one monolithic `gb-demo` job using `run_job_from_blocks(...)`.
6. Wait for the scheduler result.
7. Validate `exit_status == 0`.
8. Validate that `energy_log.json` exists in the attempt work directory.
9. Write per-target metadata.
10. Return a structured result object.

The task must raise on failure, and the implementation must distinguish retryable failures from non-retryable failures.

## 9. Output Layout

Outputs should mirror the relative path under `input_root_dir`.

Example:

- input root: `algorithms/gb_sqd/data/ligand`
- target input: `algorithms/gb_sqd/data/ligand/19_26MO_Lig/atom_4`
- output root: `/shared/gb_sqd_runs/ligand`
- target output root: `/shared/gb_sqd_runs/ligand/19_26MO_Lig/atom_4`

Recommended layout:

```text
output_root_dir/
  _bulk_summary/
    run_summary.json
  19_26MO_Lig/
    atom_4/
      target_status.json
      attempt_001/
        gb_sqd_ext.pjm
        gb_sqd_ext.pjm.gbsqd-bulk-....out
        gb_sqd_ext.pjm.gbsqd-bulk-....err
        gb_sqd_ext.pjm.gbsqd-bulk-....stats
        energy_log.json
      attempt_002/
        ...
```

Notes:

- One retry must create a new `attempt_NNN` directory.
- `target_status.json` stores the latest known status for skip/restart decisions.
- `run_summary.json` stores the full flow-level summary.

## 10. Queue Slot Control on Fugaku

### 10.1 Requirement

Before submitting a target job, the task must ensure that the number of current jobs in the Fugaku queue does not exceed `max_jobs_in_queue`.

The count must include:

- running jobs
- waiting / queued jobs

The count must exclude:

- terminal jobs
- completed job history

### 10.2 Counting scope

`queue_limit_scope="user_queue"`:

- count all non-terminal jobs of the current user in the same `RSC_GRP`

`queue_limit_scope="flow_jobs_only"`:

- count only non-terminal jobs of the current user in the same `RSC_GRP`
- additionally filter by `JOB_NAME.startswith(job_name_prefix)`

Default behavior should be `user_queue`, because it respects jobs that the user already has in the same queue outside this flow.

### 10.3 Terminal states

For the first implementation, the terminal-state check should follow the same rule already used by the Fugaku runtime:

- `EXT`
- `CCL`

Everything else returned by `pjstat` should be treated as active for queue-limit purposes.

This keeps the queue gate aligned with the existing executor behavior instead of introducing a second terminal-state definition.

### 10.4 Polling behavior

If the queue is full:

1. do not submit the new job
2. sleep for `queue_poll_interval_seconds`
3. check the queue again
4. submit only after the active count becomes `< max_jobs_in_queue`

This makes the queue limit include jobs that were already present before the flow started.

## 10.5 Queue counting command

The first implementation should use `pjstat` output to count active jobs.

The counter must inspect at least:

- `USER`
- `RSC_GRP`
- `JOB_NAME`
- `ST`

If the plain `pjstat` view is insufficient on Fugaku, the implementation may use `pjstat -v` or another stable machine-readable view, but the filtering rules above remain the same.

## 11. Fugaku Job Naming

Each submitted job should set `fugaku_job_name` explicitly instead of using the generic default.

Recommended format:

```text
{job_name_prefix}-{mode}-{short_hash}
```

Required properties:

- unique enough to avoid collisions inside one bulk run
- short enough for Fugaku job-name limits
- traceable back to the target directory through metadata

The full relative target path should be stored in `target_status.json`, even if the actual PJM job name uses a shortened hash.

## 12. Retry and Re-run Semantics

### 12.1 Prefect retry behavior

Each per-target task should use Prefect retries.

`max_target_task_retries=1` means:

- first submission attempt
- one retry submission if the task fails

The retry should resubmit the target as a fresh Fugaku job and write to a new attempt directory.

### 12.2 What counts as a retryable failure

Retryable:

- `pjsub` submission failure
- queue-status query failure
- scheduler result with non-zero exit status
- missing `energy_log.json`
- corrupted or unreadable final metadata file

Non-retryable:

- target input files are missing
- invalid user parameters detected before submission

Non-retryable failures should surface immediately without consuming the retry budget.

### 12.3 Re-run of the whole flow

When the whole flow is started again with the same `input_root_dir` and `output_root_dir`:

- successful targets are skipped if `skip_completed=True`
- failed or incomplete targets are submitted again
- the scan is performed again, so newly added target directories are included automatically

## 13. Flow-Level Result

The flow should always aggregate all target results into a final summary object with:

- total discovered targets
- skipped targets
- succeeded targets
- failed targets
- per-target metadata:
  - relative input path
  - latest attempt number
  - final status
  - latest job id
  - latest output directory
  - final energy if available

The flow should:

- succeed if all targets either succeeded or were skipped
- fail at the end if at least one target still failed after retries

## 14. Recommended Execution Model

Recommended topology:

1. discovery task
2. per-target execution tasks
3. summary task

Recommended task runner:

- `ConcurrentTaskRunner`

Recommended concurrency rule:

- if `max_prefect_concurrency` is not set, use `max_jobs_in_queue`

This prevents the Prefect worker from creating an excessive number of simultaneously waiting target tasks.

## 14.1 Concurrency semantics

`max_prefect_concurrency` and the Fugaku queue limit solve different problems.

`max_prefect_concurrency`:

- limits how many per-target Prefect tasks are active on the worker at once
- protects the worker from creating too many long-lived waiting tasks
- is a local orchestration limit only

Fugaku queue gate based on `pjstat`:

- decides whether a new job may actually be submitted
- must count jobs already in the queue before this flow started
- must include jobs submitted outside the current flow when `queue_limit_scope="user_queue"`

A local semaphore may still be used to implement `max_prefect_concurrency`, but it is not sufficient for queue control by itself.
The actual submit decision must remain based on the `pjstat` queue gate.

## 15. Command Arguments Per Mode

The per-target task should build `user_args` from the actual Fugaku CLI contract, not only from the current simple-flow helpers.

Required common arguments per target:

- `--fcidump <target_input_dir>/<fcidump_filename>`
- `--count_dict_file <target_input_dir>/<count_dict_filename>`
- `--output_dir <attempt_work_dir>`

Required ExtSQD arguments:

- `--mode ext_sqd`
- `--num_samples_per_batch <int>`
- `--num_batches <int>`
- `--num_recovery <int>`
- `--iteration <int>`
- `--adet_comm_size <int>`
- `--bdet_comm_size <int>`
- `--task_comm_size <int>`
- `--adet_comm_size_final <int>`
- `--bdet_comm_size_final <int>`
- `--task_comm_size_final <int>`
- `--do_carryover_in_recovery` when enabled
- `--carryover_ratio <float>`
- `--carryover_threshold <float>`
- `--max_time <float>`
- `--with_hf` when enabled
- `-v` when enabled

Required TrimSQD arguments:

- `--mode trim_sqd`
- `--num_samples_per_recovery <int>`
- `--num_batches <int>`
- `--num_recovery <int>`
- `--iteration <int>`
- `--adet_comm_size <int>`
- `--bdet_comm_size <int>`
- `--task_comm_size <int>`
- `--adet_comm_size_combined <int>`
- `--bdet_comm_size_combined <int>`
- `--task_comm_size_combined <int>`
- `--adet_comm_size_final <int>`
- `--bdet_comm_size_final <int>`
- `--task_comm_size_final <int>`
- `--carryover_ratio_batch <float>`
- `--carryover_ratio_combined <float>`
- `--carryover_threshold <float>`
- `--max_time <float>`
- `--with_hf` when enabled
- `-v` when enabled

This keeps the new bulk flow aligned with the actual command lines already used on Fugaku.

## 16. Validation Rules

Before any submission, validate:

- `input_root_dir` exists
- `output_root_dir` is writable
- `max_jobs_in_queue >= 1`
- `queue_poll_interval_seconds > 0`
- selected block names resolve correctly
- the loaded `HPCProfileBlock.hpc_target` is `fugaku` for the first implementation
- the Fugaku launcher resolves to `mpirun`
- the generated runtime can express the required PJM directives and environment setup from section 4.1

If `command_block_name` or `execution_profile_block_name` is omitted, the flow should resolve mode-specific defaults automatically.

## 16.1 Current implementation gaps to close

The current qcsc-prefect Fugaku path does not yet fully represent the reference scripts.

Known gaps:

- current GB-SQD simple-flow Python signatures do not expose:
  - `adet_comm_size_final`
  - `bdet_comm_size_final`
  - `task_comm_size_final`
  - `do_carryover_in_recovery`
- current TrimSQD simple-flow signature also lacks the `*_final` communicator parameters
- the Fugaku PJM template does not currently emit `#PJM -L "freq=2000,eco_state=2"`
- the Fugaku template currently supports `spack load ...`, but the reference script requires `module load LLVM/llvmorg-21.1.0`
- `ExecutionProfileBlock.modules` exists, but the Fugaku template does not currently render it
- queue gating requires a new `pjstat` parsing helper for active-job counting

Already supported by the current model:

- `OMP_NUM_THREADS`, `UTOFU_SWAP_PROTECT`, and `LD_LIBRARY_PATH` can be passed via `ExecutionProfileBlock.environments`
- `#PJM --mpi "max-proc-per-node=1"` can be passed via `HPCProfileBlock.mpi_options_for_pjm`
- `#PJM -x PJM_LLIO_GFSCACHE=...` can be passed via `HPCProfileBlock.gfscache`

The bulk-flow implementation should either extend the common block/template model to support these items, or introduce GB-SQD-specific overrides for Fugaku before the bulk flow is considered complete.

## 16.2 Recommended implementation order

The highest-signal implementation order is:

1. extend `ext_sqd_simple_flow` and `trim_sqd_simple_flow` with the missing GB-SQD CLI parameters while preserving backward compatibility
2. extend the Fugaku request/template path for the missing scheduler and module-loading behavior
3. implement `gb_sqd/fugaku_queue.py` for `pjstat`-based queue gating
4. add worker-side concurrency limits as a separate concern from queue gating

## 17. Example Usage

```python
result = await bulk_gb_sqd_flow(
    mode="ext_sqd",
    input_root_dir="algorithms/gb_sqd/data/ligand",
    output_root_dir="/shared/gb_sqd_runs/ligand",
    hpc_profile_block_name="hpc-fugaku-gb-sqd",
    execution_profile_block_name="exec-gb-sqd-ext-fugaku",
    command_block_name="cmd-gb-sqd-ext",
    max_jobs_in_queue=8,
    queue_limit_scope="user_queue",
    max_target_task_retries=1,
    num_recovery=3,
    num_batches=8,
    num_samples_per_batch=1000,
    adet_comm_size_final=2,
    bdet_comm_size_final=1,
    task_comm_size_final=1,
    do_carryover_in_recovery=True,
)
```

## 18. Implementation Notes

Suggested implementation units:

- `gb_sqd/discovery.py`
  - recursive target discovery
- `gb_sqd/fugaku_queue.py`
  - active-job counting and wait-for-slot helper
- `gb_sqd/tasks/bulk_target_run.py`
  - one target directory task
- `gb_sqd/main.py`
  - new bulk flow entrypoints

This keeps the new behavior separate from the existing task-splitting code path.
