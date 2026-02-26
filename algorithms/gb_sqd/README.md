# GB SQD Algorithm Integration

Prefect workflow integration for GB SQD (ExtSQD and TrimSQD) algorithms.

## Overview

This package provides Prefect workflows for running ExtSQD and TrimSQD algorithms on HPC systems (Fugaku, Miyabi).

### Supported Workflows

- **ExtSQD**: Extended Subspace Quantum Diagonalization
- **TrimSQD**: Trimmed Subspace Quantum Diagonalization

## Installation

```bash
cd hpc-prefect/algorithms/gb_sqd
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
- `project`: Your project/group name
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
    --walltime 02:00:00
```

### 3. Run Workflow

#### Task-Based Workflow (Recommended)

The task-based workflows provide improved visibility and restart capability:

```python
from gb_sqd.main import ext_sqd_flow, trim_sqd_flow

# Run ExtSQD workflow with task-based execution
result = await ext_sqd_flow(
    command_block_name="cmd-gb-sqd-ext",
    execution_profile_block_name="exec-gb-sqd-miyabi",
    hpc_profile_block_name="hpc-miyabi-gb-sqd",
    fcidump_file="./data/fci_dump.txt",
    count_dict_file="./data/count_dict.txt",
    work_dir="./results",
    num_recovery=3,
    num_batches=8,
)

# Run TrimSQD workflow with task-based execution
result = await trim_sqd_flow(
    command_block_name="cmd-gb-sqd-trim",
    execution_profile_block_name="exec-gb-sqd-miyabi",
    hpc_profile_block_name="hpc-miyabi-gb-sqd",
    fcidump_file="./data/fci_dump.txt",
    count_dict_file="./data/count_dict.txt",
    work_dir="./results",
    num_recovery=3,
    num_batches=8,
)
```

**Benefits of Task-Based Workflows:**
- ✅ Progress visibility in Prefect dashboard for each recovery iteration
- ✅ Ability to restart from failed iteration
- ✅ Detailed telemetry and logging for each step
- ✅ Better debugging and monitoring

#### Simple Workflow (Legacy)

For backward compatibility, simple single-task workflows are also available:

```python
from gb_sqd.main import ext_sqd_simple_flow, trim_sqd_simple_flow

# Run ExtSQD workflow (single task)
result = await ext_sqd_simple_flow(
    command_block_name="cmd-gb-sqd-ext",
    execution_profile_block_name="exec-gb-sqd-miyabi",
    hpc_profile_block_name="hpc-miyabi-gb-sqd",
    fcidump_file="./data/fci_dump.txt",
    count_dict_file="./data/count_dict.txt",
    work_dir="./results",
)
```

## Workflow Architecture

### Task-Based Workflow Structure

The task-based workflows split execution into multiple Prefect tasks:

```
Flow: GB-SQD-ExtSQD / GB-SQD-TrimSQD
├─ Task 1: initialize
│  └─ Validate inputs, prepare workspace
│
├─ Task 2-N: recovery_iteration_0..N (sequential)
│  └─ Execute one recovery iteration with all batches
│     └─ Uses MPI parallelization internally (gb-demo binary)
│
├─ Task Final: final_diagonalization
│  └─ Aggregate results from all iterations
│
└─ Task Output: output_results
   └─ Generate energy_log.json and telemetry
```

**Key Points:**
- Recovery iterations run **sequentially** (each depends on previous carryover)
- Each iteration processes all batches internally via MPI
- MPI parallelization happens inside gb-demo binary (not at Prefect level)
- Prefect provides visibility and restart capability, not parallel execution

### File Structure

```
work_dir/
├── init_data.json              # Initialization data
├── recovery_0/
│   ├── energy_log.json         # Iteration 0 results
│   ├── carryover_bitstrings.txt
│   └── recovery_0.pbs.log
├── recovery_1/
│   └── ...
├── recovery_N/
│   └── ...
├── final_result.json           # Aggregated results
└── energy_log.json             # Final output
```

### Restarting from Failed Iteration

If a recovery iteration fails, you can restart the workflow from that point:

1. Check Prefect dashboard to identify failed iteration
2. Resume the flow run:
   ```bash
   prefect flow-run resume <flow-run-id>
   ```
3. The workflow will automatically skip completed iterations and restart from the failed one

## Configuration

### Block Types

1. **CommandBlock**: Defines the command to execute
   - `cmd-gb-sqd-ext`: ExtSQD mode
   - `cmd-gb-sqd-trim`: TrimSQD mode

2. **ExecutionProfileBlock**: Execution parameters
   - Number of nodes, MPI processes, OMP threads
   - Walltime, modules, environment variables

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
    --param command_block_name="cmd-gb-sqd-ext" \
    --param execution_profile_block_name="exec-gb-sqd-miyabi" \
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