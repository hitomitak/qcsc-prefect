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
pip install -e .
```

## Native Binary

The C++ implementation is maintained in a separate repository:
- Repository: [gb_demo_2026](https://github.com/your-org/gb_demo_2026)
- Build instructions: See `native/README.md`

## Quick Start

### 1. Create Prefect Blocks

```bash
python create_blocks.py \
    --hpc-target miyabi \
    --project gz00 \
    --queue regular-c \
    --work-dir ~/work/gb_sqd
```

### 2. Run Workflow

```python
from gb_sqd.main import ext_sqd_simple_flow

# Run ExtSQD workflow
result = await ext_sqd_simple_flow(
    command_block_name="cmd-gb-sqd-ext",
    execution_profile_block_name="exec-gb-sqd-miyabi",
    hpc_profile_block_name="hpc-miyabi-gb-sqd",
    fcidump_file="./data/fci_dump.txt",
    count_dict_file="./data/count_dict.txt",
    work_dir="./results",
)
```

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