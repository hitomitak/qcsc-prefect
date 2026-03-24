# GB SQD Algorithm Integration

Prefect workflow integration for GB SQD (ExtSQD and TrimSQD) algorithms.

## Overview

This package provides Prefect workflows for running ExtSQD and TrimSQD algorithms on HPC systems (Fugaku, Miyabi).

### Supported Workflows

- **ExtSQD**: Extended Subspace Quantum Diagonalization
- **TrimSQD**: Trimmed Subspace Quantum Diagonalization

Additional design and operations documents:

- Bulk flow design: `docs/bulk_submission_flow_design.md`
- Bulk flow runbook: `docs/bulk_submission_flow_runbook.md`
- Miyabi bulk rerun runbook: `docs/miyabi_bulk_rerun_runbook.md`
- Bulk flow manual test plan: `docs/bulk_submission_manual_test_plan.md`

## Installation

```bash
cd qcsc-prefect/algorithms/gb_sqd
uv pip install -e .
```

## Native Binary

The C++ implementation is maintained in a separate repository:
- Repository: [gb_demo_2026](https://github.com/your-org/gb_demo_2026)
- Build instructions: See `native/README.md`

## Quick Start

### 1. Create Configuration File

Copy the example configuration and customize it:

```bash
cp gb_sqd_blocks.example.toml gb_sqd_blocks.toml
vim gb_sqd_blocks.toml
```

Edit the following required fields:
- `hpc_target`: "miyabi" or "fugaku"
- `project`: Miyabi project name
- `group`: Fugaku Group ID (preferred in Fugaku mode; `project` is kept as a legacy alias)
- `queue`: Queue/resource group name
- `work_dir`: Working directory for job outputs

### 2. Create Prefect Blocks

Using configuration file (recommended):

```bash
python create_blocks.py --config gb_sqd_blocks.toml
```

Or specify parameters directly:

```bash
python create_blocks.py \
    --hpc-target miyabi \
    --project gz00 \
    --queue regular-c \
    --work-dir ~/work/gb_sqd
```

You can also override config file values with CLI arguments:

```bash
python create_blocks.py \
    --config gb_sqd_blocks.toml \
    --num-nodes 4 \
    --walltime 00:15:00
```

### 3. Run Workflow

#### Task-Based Workflow (Recommended)

The task-based workflows provide improved visibility and step-level failure isolation:

```python
from gb_sqd.main import ext_sqd_flow, trim_sqd_flow

# Run ExtSQD workflow with task-based execution
result = await ext_sqd_flow(
    init_command_block_name="cmd-gb-sqd-init",
    recovery_command_block_name="cmd-gb-sqd-recovery",
    finalize_command_block_name="cmd-gb-sqd-finalize",
    init_execution_profile_block_name="exec-gb-sqd-init-miyabi",
    recovery_execution_profile_block_name="exec-gb-sqd-recovery-miyabi",
    finalize_execution_profile_block_name="exec-gb-sqd-finalize-miyabi",
    hpc_profile_block_name="hpc-miyabi-gb-sqd",
    fcidump_file="./data/fci_dump.txt",
    count_dict_file="./data/count_dict.txt",
    work_dir="./results",
    num_recovery=3,
    num_iters_per_recovery=1,
    num_batches=8,
)

# Run TrimSQD workflow with task-based execution
result = await trim_sqd_flow(
    init_command_block_name="cmd-gb-sqd-init",
    recovery_command_block_name="cmd-gb-sqd-recovery",
    finalize_command_block_name="cmd-gb-sqd-finalize",
    init_execution_profile_block_name="exec-gb-sqd-init-miyabi",
    recovery_execution_profile_block_name="exec-gb-sqd-recovery-miyabi",
    finalize_execution_profile_block_name="exec-gb-sqd-finalize-miyabi",
    hpc_profile_block_name="hpc-miyabi-gb-sqd",
    fcidump_file="./data/fci_dump.txt",
    count_dict_file="./data/count_dict.txt",
    work_dir="./results",
    num_recovery=3,
    num_iters_per_recovery=1,
    num_batches=8,
)
```

**Benefits of Task-Based Workflows:**
- ✅ Progress visibility in Prefect dashboard for each recovery task/checkpoint
- ✅ Failure localization by step (`init` / each `recovery` / `finalize`)
- ✅ Detailed telemetry and logging for each step
- ✅ Better debugging and monitoring

#### Simple Workflow (Legacy)

For backward compatibility, simple single-task workflows are also available:

```python
from gb_sqd.main import ext_sqd_simple_flow, trim_sqd_simple_flow

# Run ExtSQD workflow (single task)
result = await ext_sqd_simple_flow(
    command_block_name="cmd-gb-sqd-ext",
    execution_profile_block_name="exec-gb-sqd-ext-miyabi",
    hpc_profile_block_name="hpc-miyabi-gb-sqd",
    fcidump_file="./data/fci_dump.txt",
    count_dict_file="./data/count_dict.txt",
    work_dir="./results",
)
```

#### Bulk Submission Workflow (Fugaku)

The new bulk flow scans a parent directory and submits one monolithic `gb-demo`
job per discovered target directory.

```python
from gb_sqd.bulk import bulk_gb_sqd_flow

result = bulk_gb_sqd_flow(
    mode="ext_sqd",
    input_root_dir="./data/ligand",
    output_root_dir="/shared/gb_sqd_runs/ligand_ext",
    command_block_name="cmd-gb-sqd-ext",
    execution_profile_block_name="exec-gb-sqd-ext-fugaku",
    hpc_profile_block_name="hpc-fugaku-gb-sqd",
    max_jobs_in_queue=8,
    queue_limit_scope="user_queue",
    num_recovery=2,
    num_batches=2,
    num_samples_per_batch=1000,
)
```

For the full Fugaku run procedure, see `docs/bulk_submission_flow_runbook.md`.

For a Miyabi CPU bulk run using the standard block names created by
`create_blocks.py`, set `hpc_target="miyabi"`:

```python
from gb_sqd.bulk import bulk_gb_sqd_flow

result = bulk_gb_sqd_flow(
    mode="ext_sqd",
    hpc_target="miyabi",
    input_root_dir="./data/ligand",
    output_root_dir="/shared/gb_sqd_runs/ligand_ext_miyabi",
    command_block_name="cmd-gb-sqd-ext",
    hpc_profile_block_name="hpc-miyabi-gb-sqd",
    max_jobs_in_queue=8,
    max_prefect_concurrency=8,
    num_recovery=2,
    num_batches=2,
    num_samples_per_batch=1000,
)
```

For a Miyabi GPU bulk run, create GPU-specific blocks and either pass them
explicitly or set `resource_class="gpu"` to use the default `*-miyabi-gpu`
block names:

```python
from gb_sqd.bulk import bulk_gb_sqd_flow

result = bulk_gb_sqd_flow(
    mode="trim_sqd",
    hpc_target="miyabi",
    resource_class="gpu",
    input_root_dir="./data/ligand",
    output_root_dir="/shared/gb_sqd_runs/ligand_trim_miyabi_gpu",
    command_block_name="cmd-gb-sqd-trim",
    execution_profile_block_name="exec-gb-sqd-trim-miyabi-gpu",
    hpc_profile_block_name="hpc-miyabi-gpu-gb-sqd",
    max_jobs_in_queue=8,
    max_prefect_concurrency=8,
    num_recovery=1,
    num_batches=1,
    num_samples_per_recovery=100,
)
```

If you want to run the bulk flow once and then rerun only the failed targets
with different parameters, use the rerun helper:

```python
from gb_sqd import bulk_gb_sqd_flow_with_failed_target_rerun

result = bulk_gb_sqd_flow_with_failed_target_rerun(
    mode="ext_sqd",
    input_root_dir="./data/ligand",
    output_root_dir="/shared/gb_sqd_runs/ligand_ext",
    command_block_name="cmd-gb-sqd-ext",
    execution_profile_block_name="exec-gb-sqd-ext-fugaku",
    hpc_profile_block_name="hpc-fugaku-gb-sqd",
    max_jobs_in_queue=8,
    max_prefect_concurrency=8,
    num_recovery=2,
    num_batches=2,
    num_samples_per_batch=1000,
    failed_target_override_parameters={
        "carryover_threshold": 1e-3,
    },
)
```

If you want to retry failed targets multiple times while changing the
parameters at each rerun stage, use the staged rerun helper:

```python
from gb_sqd import bulk_gb_sqd_flow_with_failed_target_rerun_plan

result = bulk_gb_sqd_flow_with_failed_target_rerun_plan(
    mode="ext_sqd",
    input_root_dir="./data/ligand",
    output_root_dir="/shared/gb_sqd_runs/ligand_ext",
    command_block_name="cmd-gb-sqd-ext",
    execution_profile_block_name="exec-gb-sqd-ext-fugaku",
    hpc_profile_block_name="hpc-fugaku-gb-sqd",
    max_jobs_in_queue=8,
    max_prefect_concurrency=8,
    num_recovery=2,
    num_batches=2,
    num_samples_per_batch=1000,
    failed_target_override_sequence=[
        {"carryover_threshold": 1e-3},
        {"carryover_threshold": 1e-2, "max_time": 1800},
        {"carryover_threshold": 1e-1, "max_time": 2400},
    ],
)
```

The same staged rerun pattern works on Miyabi CPU with the standard Miyabi
block names:

```python
from gb_sqd import bulk_gb_sqd_flow_with_failed_target_rerun_plan

result = bulk_gb_sqd_flow_with_failed_target_rerun_plan(
    mode="ext_sqd",
    hpc_target="miyabi",
    input_root_dir="./data/ligand",
    output_root_dir="/shared/gb_sqd_runs/ligand_ext_miyabi",
    command_block_name="cmd-gb-sqd-ext",
    execution_profile_block_name="exec-gb-sqd-ext-miyabi",
    hpc_profile_block_name="hpc-miyabi-gb-sqd",
    max_jobs_in_queue=8,
    max_prefect_concurrency=8,
    num_recovery=2,
    num_batches=2,
    num_samples_per_batch=1000,
    failed_target_override_sequence=[
        {"carryover_threshold": 1e-3},
        {"carryover_threshold": 1e-2, "max_time": 1800},
        {"carryover_threshold": 1e-1, "max_time": 2400},
    ],
)
```

For Miyabi GPU, switch to the GPU block names created with
`create_blocks.py --resource-class gpu`:

```python
from gb_sqd import bulk_gb_sqd_flow_with_failed_target_rerun_plan

result = bulk_gb_sqd_flow_with_failed_target_rerun_plan(
    mode="trim_sqd",
    hpc_target="miyabi",
    resource_class="gpu",
    input_root_dir="./data/ligand",
    output_root_dir="/shared/gb_sqd_runs/ligand_trim_miyabi_gpu",
    command_block_name="cmd-gb-sqd-trim",
    execution_profile_block_name="exec-gb-sqd-trim-miyabi-gpu",
    hpc_profile_block_name="hpc-miyabi-gpu-gb-sqd",
    max_jobs_in_queue=8,
    max_prefect_concurrency=8,
    num_recovery=1,
    num_batches=1,
    num_samples_per_recovery=100,
    failed_target_override_sequence=[
        {"carryover_threshold": 1e-3},
        {"carryover_threshold": 1e-2, "max_time": 1800},
    ],
)
```

## Workflow Architecture

### Task-Based Workflow Structure

The task-based workflows split execution into multiple Prefect tasks.
`num_recovery` controls how many recovery tasks/checkpoints Prefect creates.
`num_iters_per_recovery` controls how many `gb-demo recovery` iterations run inside each task.
Total recovery iterations = `num_recovery * num_iters_per_recovery`.

```
Flow: GB-SQD-ExtSQD / GB-SQD-TrimSQD
├─ Task 1: initialize
│  └─ Run `gb-demo init` and produce `init/state_iter_000.json`
│
├─ Task 2-N: recovery_iteration_0..N (sequential)
│  └─ Run `gb-demo recovery` with `--num-iters <num_iters_per_recovery>`
│     └─ Uses MPI parallelization internally (gb-demo binary)
│
├─ Task Final: final_diagonalization
│  └─ Run `gb-demo finalize` on the latest state file
│
└─ Task Output: output_results
   └─ Publish summary + Prefect artifact from energy_log.json
```

**Key Points:**
- Recovery iterations run **sequentially** (each depends on previous carryover)
- Each recovery task executes exactly one recovery iteration (`--num-iters 1`)
- MPI parallelization happens inside gb-demo binary (not at Prefect level)
- Prefect provides visibility and retries; automatic checkpoint resume is not yet enabled

### File Structure

```
work_dir/
├── init/                       # initialize task work_dir
│   └── state_iter_000.json
├── init_data.json
├── recovery_0/
│   └── state_iter_001.json
├── recovery_1/
│   └── state_iter_002.json
├── recovery_N/
│   └── state_iter_{N+1}.json
├── energy_log.json             # generated by gb-demo finalize
└── final_result.json           # summarized final metadata
```

Scheduler script files and scheduler logs (for example `*.pbs`, `*.pjm`, `*.out`, `*.err`) are also created under each task directory by `run_job_from_blocks`.

### Failure Handling and Re-run

- Task-level retries are configured (`initialize`: 2, `recovery_iteration`: 1, `final_diagonalization`: 1).
- If retries are exhausted, rerun the flow with the same parameters after addressing the root cause.
- Automatic "skip completed tasks and resume from failed iteration" is not implemented in the current GB-SQD flow (no task cache policy yet).

### Shared Filesystem Requirement

`work_dir` must be a path visible from both:
- Prefect worker host (for checking `state_iter_*.json`, `energy_log.json`)
- HPC compute nodes (for job execution outputs)

Using a non-shared local path causes post-job file checks in Prefect tasks to fail.

## Configuration

### Block Types

1. **CommandBlock**: Defines the command to execute
   - `cmd-gb-sqd-ext`: ExtSQD mode
   - `cmd-gb-sqd-trim`: TrimSQD mode
   - `cmd-gb-sqd-init`: `gb-demo init` subcommand
   - `cmd-gb-sqd-recovery`: `gb-demo recovery` subcommand
   - `cmd-gb-sqd-finalize`: `gb-demo finalize` subcommand

2. **ExecutionProfileBlock**: Execution parameters
   - Number of nodes, MPI processes, OMP threads
   - Walltime, modules, environment variables
   - Must be prepared per command (`init` / `recovery` / `finalize`) for task-based flows
   - `run_job_from_blocks` validates `ExecutionProfileBlock.command_name == CommandBlock.command_name`; mismatch raises `ValueError`

3. **HPCProfileBlock**: HPC-specific settings
   - Queue/resource group
   - Project/group
   - Executable paths

## Deployment

### Deploy Workflow

To deploy the workflow with a local Prefect worker:

```bash
# Deploy ExtSQD workflow
python deploy.py

# Or deploy TrimSQD workflow
python deploy.py trim
```

This will start a Prefect worker that serves the workflow. The workflow can then be triggered from:
- Prefect UI
- Prefect CLI
- Python API

### Example: Trigger from CLI

```bash
# After deployment, trigger a flow run
prefect deployment run 'GB-SQD-ExtSQD/gb-sqd-ext-sqd' \
    --param init_command_block_name="cmd-gb-sqd-init" \
    --param recovery_command_block_name="cmd-gb-sqd-recovery" \
    --param finalize_command_block_name="cmd-gb-sqd-finalize" \
    --param init_execution_profile_block_name="exec-gb-sqd-init-miyabi" \
    --param recovery_execution_profile_block_name="exec-gb-sqd-recovery-miyabi" \
    --param finalize_execution_profile_block_name="exec-gb-sqd-finalize-miyabi" \
    --param hpc_profile_block_name="hpc-miyabi-gb-sqd" \
    --param fcidump_file="./data/fci_dump.txt" \
    --param count_dict_file="./data/count_dict.txt" \
    --param work_dir="./results" \
    --param num_recovery=3 \
    --param num_batches=8
```

## Development

### Running Tests

```bash
pytest tests/
```

### Building Native Binary

```bash
cd native
./build_gb_sqd.sh
```

## License

Apache License 2.0
