# GB-SQD Bulk Submission Flow Runbook

This document explains how to run the new bulk GB-SQD flow on Fugaku.

The bulk flow scans a parent directory, finds leaf directories that contain both
`count_dict.txt` and `fci_dump.txt`, and runs one monolithic `gb-demo` job per
discovered target directory.

## 1. Preconditions

- Run from an environment where Prefect and Fugaku scheduler commands are available.
- The `gb-demo` executable must already be built for Fugaku.
- The input tree and output root must be on a filesystem visible from both:
  - the Prefect worker host
  - Fugaku compute nodes

Relevant implementation files:

- `algorithms/gb_sqd/gb_sqd/bulk.py`
- `algorithms/gb_sqd/gb_sqd/tasks/bulk_target_run.py`
- `algorithms/gb_sqd/gb_sqd/discovery.py`
- `algorithms/gb_sqd/gb_sqd/fugaku_queue.py`

## 2. Prepare the input tree

The flow expects a parent directory like:

```text
data/
  ligand/
    19_26MO_Lig/
      atom_1/
        count_dict.txt
        fci_dump.txt
      atom_2/
        count_dict.txt
        fci_dump.txt
    27_35MO_Lig/
      atom_0/
        count_dict.txt
        fci_dump.txt
```

Each deepest directory that contains both files becomes one GB-SQD target.

## 3. Create Fugaku blocks

Copy and edit the example config:

```bash
cd /Users/hitomi/Project/qcsc-prefect/algorithms/gb_sqd
cp gb_sqd_blocks.example.toml gb_sqd_blocks.toml
```

Recommended Fugaku settings:

```toml
hpc_target = "fugaku"
group = "ra010014"
queue = "small"
work_dir = "/shared/gb_sqd_runs"

num_nodes = 4
ompthreads = 48
walltime = "00:15:00"
launcher = "mpirun"
modules = ["LLVM/llvmorg-21.1.0"]
mpi_options = ["-n", "4"]

fugaku_gfscache = "/vol0004:/vol0002"
fugaku_mpi_options_for_pjm = ["max-proc-per-node=1"]
fugaku_pjm_resources = ["freq=2000,eco_state=2"]
```

Create the blocks:

```bash
cd /Users/hitomi/Project/qcsc-prefect/algorithms/gb_sqd
python create_blocks.py --config gb_sqd_blocks.toml
```

Expected default block names for Fugaku:

- `cmd-gb-sqd-ext`
- `cmd-gb-sqd-trim`
- `exec-gb-sqd-ext-fugaku`
- `exec-gb-sqd-trim-fugaku`
- `hpc-fugaku-gb-sqd`

## 4. Run the bulk flow

### 4.1 ExtSQD example

```python
from gb_sqd.bulk import bulk_gb_sqd_flow

result = bulk_gb_sqd_flow(
    mode="ext_sqd",
    input_root_dir="/Users/hitomi/Project/qcsc-prefect/algorithms/gb_sqd/data/ligand",
    output_root_dir="/shared/gb_sqd_runs/ligand_ext",
    command_block_name="cmd-gb-sqd-ext",
    execution_profile_block_name="exec-gb-sqd-ext-fugaku",
    hpc_profile_block_name="hpc-fugaku-gb-sqd",
    max_jobs_in_queue=8,
    queue_limit_scope="user_queue",
    max_target_task_retries=1,
    max_prefect_concurrency=8,
    num_recovery=2,
    num_batches=2,
    num_samples_per_batch=1000,
    iteration=2,
    adet_comm_size=1,
    bdet_comm_size=1,
    task_comm_size=1,
    adet_comm_size_final=2,
    bdet_comm_size_final=1,
    task_comm_size_final=1,
    do_carryover_in_recovery=True,
    carryover_ratio=0.50,
    carryover_threshold=1e-5,
    max_time=300,
    with_hf=True,
    verbose=True,
)
print(result)
```

### 4.2 TrimSQD example

```python
from gb_sqd.bulk import bulk_gb_sqd_flow

result = bulk_gb_sqd_flow(
    mode="trim_sqd",
    input_root_dir="/Users/hitomi/Project/qcsc-prefect/algorithms/gb_sqd/data/ligand",
    output_root_dir="/shared/gb_sqd_runs/ligand_trim",
    command_block_name="cmd-gb-sqd-trim",
    execution_profile_block_name="exec-gb-sqd-trim-fugaku",
    hpc_profile_block_name="hpc-fugaku-gb-sqd",
    max_jobs_in_queue=8,
    queue_limit_scope="user_queue",
    max_target_task_retries=1,
    max_prefect_concurrency=8,
    num_recovery=2,
    num_batches=2,
    num_samples_per_recovery=1000,
    iteration=2,
    adet_comm_size=1,
    bdet_comm_size=1,
    task_comm_size=1,
    adet_comm_size_combined=2,
    bdet_comm_size_combined=1,
    task_comm_size_combined=1,
    adet_comm_size_final=2,
    bdet_comm_size_final=1,
    task_comm_size_final=1,
    carryover_ratio_batch=0.10,
    carryover_ratio_combined=0.50,
    carryover_threshold=1e-5,
    max_time=300,
    with_hf=True,
    verbose=True,
)
print(result)
```

### 4.3 Run from a one-shot command

```bash
cd /Users/hitomi/Project/qcsc-prefect/algorithms/gb_sqd
uv run python -c "from gb_sqd.bulk import bulk_gb_sqd_flow; bulk_gb_sqd_flow(mode='ext_sqd', input_root_dir='./data/ligand', output_root_dir='/shared/gb_sqd_runs/ligand_ext', command_block_name='cmd-gb-sqd-ext', execution_profile_block_name='exec-gb-sqd-ext-fugaku', hpc_profile_block_name='hpc-fugaku-gb-sqd', max_jobs_in_queue=8, queue_limit_scope='user_queue', max_target_task_retries=1, max_prefect_concurrency=8, num_recovery=2, num_batches=2, num_samples_per_batch=1000, iteration=2, adet_comm_size=1, bdet_comm_size=1, task_comm_size=1, adet_comm_size_final=2, bdet_comm_size_final=1, task_comm_size_final=1, do_carryover_in_recovery=True, carryover_ratio=0.5, carryover_threshold=1e-5, max_time=300, with_hf=True, verbose=True)"
```

## 5. Queue control behavior

The flow uses two limits:

- `max_prefect_concurrency`
  - limits how many per-target Prefect tasks are active at once
  - when one target finishes, the flow submits the next discovered target without
    waiting for the rest of the current wave to finish
- `max_jobs_in_queue`
  - uses `pjstat` to decide whether another Fugaku job may be submitted

Recommended starting point:

- `max_prefect_concurrency = max_jobs_in_queue`

If you already have unrelated jobs in the same queue and want those counted too,
keep:

- `queue_limit_scope="user_queue"`

If you want only jobs submitted by this flow to be counted, use:

- `queue_limit_scope="flow_jobs_only"`

## 6. Output layout

For each discovered target directory, the flow creates:

```text
<output_root_dir>/
  _bulk_summary/
    run_summary.json
  <relative_target_path>/
    target_status.json
    attempt_001/
      gb_sqd_ext.pjm or gb_sqd_trim.pjm
      *.out
      *.err
      *.stats
      energy_log.json
    attempt_002/
      ...
```

Important files:

- `target_status.json`
  - latest status for that target
- `_bulk_summary/run_summary.json`
  - flow-level summary across all targets

## 7. Re-run behavior

If you rerun the same bulk flow with the same `input_root_dir` and
`output_root_dir`:

- successful targets are skipped by default
- failed targets are retried in a new `attempt_NNN` directory
- newly added input directories are discovered automatically

If you want to force rerun successful targets too, set:

- `skip_completed=False`

## 8. Override parameters for failed targets only

If some targets failed and you want to rerun only those targets with different
GB-SQD parameters:

- keep the same `input_root_dir`
- keep the same `output_root_dir`
- keep `skip_completed=True`
- pass `target_overrides` keyed by the discovered relative target path

Example:

```python
from gb_sqd.bulk import bulk_gb_sqd_flow

result = bulk_gb_sqd_flow(
    mode="ext_sqd",
    input_root_dir="/Users/hitomi/Project/qcsc-prefect/algorithms/gb_sqd/data/ligand",
    output_root_dir="/shared/gb_sqd_runs/ligand_ext",
    command_block_name="cmd-gb-sqd-ext",
    execution_profile_block_name="exec-gb-sqd-ext-fugaku",
    hpc_profile_block_name="hpc-fugaku-gb-sqd",
    skip_completed=True,
    max_jobs_in_queue=8,
    target_overrides={
        "13_18MO_Wat/atom_10129": {
            "max_time": 600,
            "num_samples_per_batch": 500,
        }
    },
    num_recovery=2,
    num_batches=2,
    num_samples_per_batch=1000,
    iteration=2,
)
```

In this example:

- already successful targets are skipped
- failed targets without an override rerun with the shared parameters
- `13_18MO_Wat/atom_10129` reruns with `max_time=600` and `num_samples_per_batch=500`

Rules:

- keys must use the discovered relative path under `input_root_dir`
- unknown target paths raise `ValueError`
- unknown parameter names inside `target_overrides` also raise `ValueError`

## 9. Current limitations

- The queue gate has unit tests, but it still needs confirmation against the
  real `pjstat` listing format on a Fugaku login node.
- The bulk flow has helper-level unit tests, but not a full Prefect integration
  test yet.
- The task-based `init/recovery/finalize` flows are still separate from the new
  monolithic bulk flow.
