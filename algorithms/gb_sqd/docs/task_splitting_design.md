# GB-SQD Task Splitting Design Document

## Table of Contents

1. [Background and Objectives](#background-and-objectives)
2. [Current Challenges](#current-challenges)
3. [Proposed Design](#proposed-design)
4. [Subcommand Structure](#subcommand-structure)
5. [State Management](#state-management)
6. [File Reference Specification](#file-reference-specification)
7. [Usage Examples](#usage-examples)
8. [Backward Compatibility](#backward-compatibility)

---

## Background and Objectives

### Current Situation

The GB-SQD (Generalized Bitstring Subspace Quantum Diagonalization) algorithm currently runs as a single Prefect Task, which presents the following challenges:

- **Lack of visibility**: Cannot track detailed progress in the Prefect dashboard
- **No restart capability**: Must restart from the beginning if execution fails
- **Difficult debugging**: Hard to identify which step caused the problem

### Objectives

Split the GB-SQD workflow into multiple Prefect Tasks to achieve:

1. **Improved visibility**: Track progress of each step in the Prefect dashboard
2. **Restart capability**: Resume from failed steps
3. **Flexible granularity control**: Fine-grained for debugging, coarse-grained for production
4. **HPC scheduler efficiency**: Optimize the number of job submissions

---

## Current Challenges

### gb-demo Binary Structure

The current [`gb-demo`](../gb_demo_2026/src/main.cpp) executes recovery loops internally:

```cpp
// ext_sqd.hpp (current implementation)
for (uint64_t i_recovery = 0; i_recovery < opt.num_recovery; ++i_recovery) {
    // 1. Configuration Recovery
    recover_configurations(...);
    
    // 2. Subsampling
    subsample(...);
    
    // 3. SBD (diagonalization)
    sbd_main(...);
    
    // 4. Update occupancies and carryover
    update_state(...);
}
```

With this structure, even if we split at the Prefect Task level, **the loop still executes inside gb-demo**, so it's not a true split.

### Required Changes

We need to split the gb-demo binary itself so that each step can be invoked via CLI.

---

## Proposed Design

### Design Principles

1. **Single binary with subcommands**: For maintainability, implement subcommands within a single binary rather than creating three separate binaries
2. **File references for state**: Use references to binary files instead of embedding large data in JSON
3. **Range execution**: Support executing multiple iterations with `--num-iters` parameter
4. **Backward compatibility**: Continue to support existing `--mode ext_sqd/trim_sqd` usage

### Architecture

```mermaid
graph LR
    A[gb-demo init] --> B[state_iter_000.json]
    B --> C[gb-demo recovery]
    C --> D[state_iter_005.json]
    D --> E[gb-demo recovery]
    E --> F[state_iter_010.json]
    F --> G[gb-demo finalize]
    G --> H[energy_log.json]
```

---

## Subcommand Structure

### Command List

```bash
gb-demo <subcommand> [options]
```

| Subcommand | Description | Main Role |
|------------|-------------|-----------|
| `init` | Initialization | Load count dictionary, prepare initial occupancies |
| `recovery` | Recovery iteration | Configuration recovery → Subsampling → SBD |
| `finalize` | Finalization | Aggregate results, generate energy_log.json |

### 1. `gb-demo init`

**Purpose**: Initialize the workflow

**Inputs**:
- `--mode <ext_sqd|trim_sqd>`: Execution mode
- `--count_dict_file <path>`: Count dictionary file
- `--fcidump <path>`: FCIDUMP file
- `--initial_occupancies <path>`: (Optional) Initial occupancies
- `--output_dir <path>`: Output directory
- `--state_out <path>`: Output state JSON file

**Outputs**:
- `state_iter_000.json`: Initial state
- `occupancies_iter_000.bin`: Initial occupancies (binary)

**Example**:
```bash
gb-demo init \
    --mode ext_sqd \
    --count_dict_file counts.json \
    --fcidump fcidump.txt \
    --output_dir output/ \
    --state_out output/state_iter_000.json
```

---

### 2. `gb-demo recovery`

**Purpose**: Execute recovery iterations (supports range execution)

**Inputs**:
- `--start-iter <N>`: Starting iteration number
- `--num-iters <K>`: Number of iterations to execute
- `--state-in <path>`: Input state JSON file
- `--state-out <path>`: Output state JSON file
- `--num_samples_per_batch <N>`: Number of samples per batch
- `--do_carryover_in_recovery`: Enable carryover

**Outputs**:
- `state_iter_<N>.json`: New state
- `carryover_alpha_iter_<N>.bin`: Alpha carryover bitstrings
- `carryover_beta_iter_<N>.bin`: Beta carryover bitstrings
- `AlphaDets_iter_<N>.bin`: Alpha determinants
- `BetaDets_iter_<N>.bin`: Beta determinants

**Examples**:
```bash
# Execute one iteration at a time (for debugging)
gb-demo recovery \
    --start-iter 0 \
    --num-iters 1 \
    --state-in output/state_iter_000.json \
    --state-out output/state_iter_001.json \
    --num_samples_per_batch 1000

# Execute 5 iterations together (for production)
gb-demo recovery \
    --start-iter 0 \
    --num-iters 5 \
    --state-in output/state_iter_000.json \
    --state-out output/state_iter_005.json \
    --num_samples_per_batch 1000 \
    --do_carryover_in_recovery
```

---

### 3. `gb-demo finalize`

**Purpose**: Aggregate final results and generate output

**Inputs**:
- `--state-in <path>`: Final iteration state JSON file
- `--output_dir <path>`: Output directory

**Outputs**:
- `energy_log.json`: Energy history
- `telemetry.json`: Telemetry data

**Example**:
```bash
gb-demo finalize \
    --state-in output/state_iter_010.json \
    --output_dir output/
```

---

## State Management

### State JSON Schema

```json
{
  "schema_version": "1.0.0",
  "metadata": {
    "created_at": "2026-02-26T06:00:00Z",
    "command": "gb-demo recovery",
    "git_commit": "abc123def456",
    "hostname": "fugaku-login01"
  },
  "state": {
    "mode": "ext_sqd",
    "current_iteration": 5,
    "total_iterations": 10,
    "norb": 10,
    "nelec_a": 5,
    "nelec_b": 5,
    
    "occupancies_path": "occupancies_iter_005.bin",
    "carryover_alpha_path": "carryover_alpha_iter_005.bin",
    "carryover_beta_path": "carryover_beta_iter_005.bin",
    "adet_path": "AlphaDets_iter_005.bin",
    "bdet_path": "BetaDets_iter_005.bin",
    
    "energy_history": [-123.456, -124.567, -125.678, -126.789, -127.890],
    "num_carryover_alpha": 5000,
    "num_carryover_beta": 5000
  },
  "checksum": "sha256:1234567890abcdef..."
}
```

### Field Descriptions

#### `schema_version`
- Schema version of the state JSON
- Used for compatibility checking
- Follows semantic versioning (e.g., "1.0.0")

#### `metadata`
- `created_at`: State file creation timestamp (ISO 8601 format)
- `command`: Executed command
- `git_commit`: gb-demo Git commit hash
- `hostname`: Execution hostname

#### `state`
- `mode`: Execution mode (`ext_sqd` or `trim_sqd`)
- `current_iteration`: Current iteration number
- `total_iterations`: Total number of iterations
- `norb`: Number of orbitals
- `nelec_a`, `nelec_b`: Number of alpha/beta electrons
- `*_path`: Relative paths to binary files
- `energy_history`: Energy history (lightweight data, stored in JSON)
- `num_carryover_*`: Number of carryover bitstrings

#### `checksum`
- For state file integrity checking
- SHA-256 hash

---

## File Reference Specification

### Why File References Are Needed

**Problem**: Carryover bitstrings and occupancies can contain tens of thousands to hundreds of thousands of data points. Embedding them directly in JSON causes:
- JSON files become huge (hundreds of MB to GB)
- Long parsing time
- High memory consumption

**Solution**: Store large data in binary files and record only paths in JSON

### Directory Structure

```
output/
├── state_iter_000.json              # Initial state (lightweight, few KB)
├── occupancies_iter_000.bin         # Initial occupancies (binary)
│
├── state_iter_005.json              # Iteration 5 state
├── occupancies_iter_005.bin
├── carryover_alpha_iter_005.bin
├── carryover_beta_iter_005.bin
├── AlphaDets_iter_005.bin
├── BetaDets_iter_005.bin
│
├── state_iter_010.json              # Final state
├── occupancies_iter_010.bin
├── carryover_alpha_iter_010.bin
├── carryover_beta_iter_010.bin
│
├── energy_log.json                  # Final results
└── telemetry.json
```

### Binary File Format

Each binary file is saved in the following format:

```
[Header: 16 bytes]
  - Magic number: 4 bytes (0x47425344 = "GBSD")
  - Version: 4 bytes
  - Data type: 4 bytes (0=double, 1=bitstring)
  - Count: 4 bytes (number of elements)

[Data: variable length]
  - Array of doubles or bitstrings
```

---

## Usage Examples

### Debugging (Fine-grained Execution)

```bash
# 1. Initialize
gb-demo init \
    --mode ext_sqd \
    --count_dict_file counts.json \
    --fcidump fcidump.txt \
    --output_dir output/ \
    --state_out output/state_iter_000.json

# 2. Recovery (one iteration at a time)
gb-demo recovery \
    --start-iter 0 --num-iters 1 \
    --state-in output/state_iter_000.json \
    --state-out output/state_iter_001.json \
    --num_samples_per_batch 1000

gb-demo recovery \
    --start-iter 1 --num-iters 1 \
    --state-in output/state_iter_001.json \
    --state-out output/state_iter_002.json \
    --num_samples_per_batch 1000

# ... repeat ...

# 3. Finalize
gb-demo finalize \
    --state-in output/state_iter_010.json \
    --output_dir output/
```

### Production (Coarse-grained Execution)

```bash
# 1. Initialize
gb-demo init \
    --mode ext_sqd \
    --count_dict_file counts.json \
    --fcidump fcidump.txt \
    --output_dir output/ \
    --state_out output/state_iter_000.json

# 2. Recovery (5 iterations at a time)
gb-demo recovery \
    --start-iter 0 --num-iters 5 \
    --state-in output/state_iter_000.json \
    --state-out output/state_iter_005.json \
    --num_samples_per_batch 1000 \
    --do_carryover_in_recovery

gb-demo recovery \
    --start-iter 5 --num-iters 5 \
    --state-in output/state_iter_005.json \
    --state-out output/state_iter_010.json \
    --num_samples_per_batch 1000 \
    --do_carryover_in_recovery

# 3. Finalize
gb-demo finalize \
    --state-in output/state_iter_010.json \
    --output_dir output/
```

### Python (Prefect) Usage

```python
from prefect import task, flow
from pathlib import Path
import json

@task
async def initialize_task(
    command_block_name: str,
    count_dict_file: str,
    fcidump_file: str,
    output_dir: str = "output",
) -> Path:
    """Initialize GB-SQD workflow"""
    output_state = Path(output_dir) / "state_iter_000.json"
    
    user_args = [
        "gb-demo", "init",
        "--mode", "ext_sqd",
        "--count_dict_file", count_dict_file,
        "--fcidump", fcidump_file,
        "--output_dir", output_dir,
        "--state_out", str(output_state),
    ]
    
    await run_job_from_blocks(
        command_block_name=command_block_name,
        user_args=user_args,
    )
    
    return output_state

@task
async def recovery_range_task(
    command_block_name: str,
    start_iter: int,
    num_iters: int,
    state_in: Path,
    num_samples_per_batch: int,
    do_carryover: bool,
    output_dir: str = "output",
) -> Path:
    """Execute a range of recovery iterations"""
    end_iter = start_iter + num_iters
    output_state = Path(output_dir) / f"state_iter_{end_iter:03d}.json"
    
    user_args = [
        "gb-demo", "recovery",
        "--start-iter", str(start_iter),
        "--num-iters", str(num_iters),
        "--state-in", str(state_in),
        "--state-out", str(output_state),
        "--num_samples_per_batch", str(num_samples_per_batch),
    ]
    
    if do_carryover:
        user_args.append("--do_carryover_in_recovery")
    
    await run_job_from_blocks(
        command_block_name=command_block_name,
        user_args=user_args,
    )
    
    return output_state

@task
async def finalize_task(
    command_block_name: str,
    state_in: Path,
    output_dir: str = "output",
) -> dict:
    """Finalize GB-SQD workflow"""
    user_args = [
        "gb-demo", "finalize",
        "--state-in", str(state_in),
        "--output_dir", output_dir,
    ]
    
    await run_job_from_blocks(
        command_block_name=command_block_name,
        user_args=user_args,
    )
    
    with open(Path(output_dir) / "energy_log.json") as f:
        return json.load(f)

@flow
async def ext_sqd_flow(
    command_block_name: str,
    count_dict_file: str,
    fcidump_file: str,
    num_recovery: int = 10,
    num_samples_per_batch: int = 1000,
    do_carryover: bool = False,
    output_dir: str = "output",
    iterations_per_job: int = 1,  # 1 for debugging, 3-5 for production
):
    """GB-SQD Ext-SQD workflow with flexible granularity"""
    
    # 1. Initialize
    state = await initialize_task(
        command_block_name=command_block_name,
        count_dict_file=count_dict_file,
        fcidump_file=fcidump_file,
        output_dir=output_dir,
    )
    
    # 2. Recovery iterations (chunked)
    current_iter = 0
    while current_iter < num_recovery:
        chunk_size = min(iterations_per_job, num_recovery - current_iter)
        
        state = await recovery_range_task(
            command_block_name=command_block_name,
            start_iter=current_iter,
            num_iters=chunk_size,
            state_in=state,
            num_samples_per_batch=num_samples_per_batch,
            do_carryover=do_carryover,
            output_dir=output_dir,
        )
        
        current_iter += chunk_size
    
    # 3. Finalize
    final_results = await finalize_task(
        command_block_name=command_block_name,
        state_in=state,
        output_dir=output_dir,
    )
    
    return final_results
```

---

## Backward Compatibility

### Existing Usage Continues to Work

```bash
# Existing method (monolithic execution)
gb-demo --mode ext_sqd \
    --count_dict_file counts.json \
    --fcidump fcidump.txt \
    --num_recovery 10 \
    --num_batches 8 \
    --num_samples_per_batch 1000
```

This usage continues to work. Internally, [`cmd::run_ext_sqd()`](../gb_demo_2026/src/commands/ext_sqd.hpp) is called.

### Implementation Method

In [`main.cpp`](../gb_demo_2026/src/main.cpp), check the first argument:

```cpp
int main(int argc, char *argv[]) {
    MPI_Init_thread(...);
    
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string arg1 = argv[1];
    
    // New subcommand approach
    if (arg1 == "init") {
        return cmd::run_init(argc - 1, argv + 1);
    } else if (arg1 == "recovery") {
        return cmd::run_recovery(argc - 1, argv + 1);
    } else if (arg1 == "finalize") {
        return cmd::run_finalize(argc - 1, argv + 1);
    }
    
    // Backward compatibility: --mode ext_sqd/trim_sqd
    Mode mode = detect_mode(argc, argv);
    if (mode == Mode::ExtSQD) {
        return cmd::run_ext_sqd(argc, argv);
    } else if (mode == Mode::TrimSQD) {
        return cmd::run_trim_sqd(argc, argv);
    }
    
    std::cerr << "Unknown subcommand or mode" << std::endl;
    return 1;
}
```

---