# Miyabi BitCount Tutorial (Wrapper-Compatible + Optimized)

This tutorial provides **two BitCount workflows** for Miyabi:

1. **Wrapper-compatible flow**: keeps the same structure as `create_qcsc_workflow.md` (`counter.get(bitstrings)` style).
2. **Optimized flow**: uses the current `hpc-prefect` block architecture (`CommandBlock` / `ExecutionProfileBlock` / `HPCProfileBlock`) and optimized MPI output handling.

The setup is script-driven so users mainly run scripts and choose block names at execution time.

## Directory

- `/Users/hitomi/Project/hpc-execution-profiles/hpc-prefect/examples/miyabi_prefect_bitcount_demo/build_binaries.sh`
- `/Users/hitomi/Project/hpc-execution-profiles/hpc-prefect/examples/miyabi_prefect_bitcount_demo/create_blocks.py`
- `/Users/hitomi/Project/hpc-execution-profiles/hpc-prefect/examples/miyabi_prefect_bitcount_demo/bitcount_blocks.example.toml`
- `/Users/hitomi/Project/hpc-execution-profiles/hpc-prefect/examples/miyabi_prefect_bitcount_demo/wrapper_block.py`
- `/Users/hitomi/Project/hpc-execution-profiles/hpc-prefect/examples/miyabi_prefect_bitcount_demo/flow_wrapper.py`
- `/Users/hitomi/Project/hpc-execution-profiles/hpc-prefect/examples/miyabi_prefect_bitcount_demo/flow_optimized.py`

## Prerequisites

- You can run `qsub` / `qstat` on Miyabi.
- Prefect API is reachable.
- A Quantum Runtime block (for example `ibm-runner`) already exists.
- Python environment includes:
  - `prefect`
  - `prefect-qiskit`
  - `qiskit`
  - this workspace packages (`hpc-prefect-*`)

## Step 1. Sync workspace dependencies

```bash
cd /Users/hitomi/Project/hpc-execution-profiles/hpc-prefect
uv sync
```

If needed, add Qiskit integration packages:

```bash
cd /Users/hitomi/Project/hpc-execution-profiles/hpc-prefect
uv pip install prefect-qiskit qiskit
```

## Step 2. Build BitCount executables

Run this on a Miyabi environment where `mpiicpx` is available:

```bash
cd /Users/hitomi/Project/hpc-execution-profiles/hpc-prefect
./examples/miyabi_prefect_bitcount_demo/build_binaries.sh
```

This creates:

- `examples/miyabi_prefect_bitcount_demo/bin/get_counts_json`
- `examples/miyabi_prefect_bitcount_demo/bin/get_counts_hist`

## Step 3. Create all Blocks and Variables by script

Use a config file (recommended):

```bash
cd /Users/hitomi/Project/hpc-execution-profiles/hpc-prefect
cp examples/miyabi_prefect_bitcount_demo/bitcount_blocks.example.toml \
   examples/miyabi_prefect_bitcount_demo/bitcount_blocks.toml
```

Edit `examples/miyabi_prefect_bitcount_demo/bitcount_blocks.toml` and set only these required keys:

- `project`
- `queue`
- `work_dir` (base directory; each run creates `job_xxxx` under this path)

Then run:

```bash
cd /Users/hitomi/Project/hpc-execution-profiles/hpc-prefect
uv run python examples/miyabi_prefect_bitcount_demo/create_blocks.py \
  --config examples/miyabi_prefect_bitcount_demo/bitcount_blocks.toml
```

You can still override individual values from CLI arguments:

```bash
cd /Users/hitomi/Project/hpc-execution-profiles/hpc-prefect
uv run python examples/miyabi_prefect_bitcount_demo/create_blocks.py \
  --config examples/miyabi_prefect_bitcount_demo/bitcount_blocks.toml \
  --shots 200000 \
  --num-nodes 4 \
  --mpiprocs 8
```

Legacy env vars are still supported for backward compatibility (`MIYABI_PBS_PROJECT`, `MIYABI_PBS_QUEUE`, etc.), but the config file approach is clearer for shared setup.

Defaulted keys (optional, no need to set unless you want to customize):

- `launcher="mpiexec.hydra"`
- `walltime="00:10:00"`
- `num_nodes=2`
- `mpiprocs=5`
- `ompthreads=1`
- `shots=100000`
- `modules=["intel/2023.2.0","impi/2021.10.0"]`
- `mpi_options=[]`

Optional advanced keys:

- `wrapper_executable`, `optimized_executable`
- `wrapper_block_name`, `command_block_name`, `execution_profile_block_name`, `hpc_profile_block_name`, `options_variable_name`

## Step 4A. Run Wrapper-compatible tutorial flow

This mirrors the legacy tutorial style with a wrapper block and `counter.get(bitstrings)`.

```bash
cd /Users/hitomi/Project/hpc-execution-profiles/hpc-prefect
uv run python examples/miyabi_prefect_bitcount_demo/flow_wrapper.py \
  --runtime-block ibm-runner \
  --counter-block bit-counter-wrapper-demo \
  --options-variable miyabi-bitcount-options
```

Generated artifacts include:

- `sampler-count-dict-wrapper`
- `miyabi-bitcount-wrapper-metrics`
- Job script and output files in `<work_dir>/job_xxxx/`

## Step 4B. Run Optimized tutorial flow

This version uses the current codebase design:

- reusable `CommandBlock` / `ExecutionProfileBlock` / `HPCProfileBlock`
- binary histogram output (`hist_u64.bin`) instead of JSON text
- `MPI_Scatterv` for non-even work distribution

```bash
cd /Users/hitomi/Project/hpc-execution-profiles/hpc-prefect
uv run python examples/miyabi_prefect_bitcount_demo/flow_optimized.py \
  --runtime-block ibm-runner \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-mpi \
  --hpc-profile-block hpc-miyabi-bitcount \
  --options-variable miyabi-bitcount-options
```

Generated artifacts include:

- `sampler-count-dict-optimized`
- `miyabi-bitcount-optimized-metrics`
- Job script and output files in `<work_dir>/job_xxxx/`

## What users still choose manually

After setup, users only need to select block names at run time:

- Wrapper flow: `--counter-block`
- Optimized flow: `--command-block`, `--execution-profile-block`, `--hpc-profile-block`

Everything else (block schema, block instances, sampler options variable) is created by `create_blocks.py`.

## Notes

- The wrapper-compatible MPI executable (`get_counts_json`) is intentionally close to the original tutorial behavior.
- The optimized executable (`get_counts_hist`) is preferred for larger shot counts.
- If you need custom executable paths, set `wrapper_executable` and `optimized_executable` in the config file or pass `--wrapper-executable` / `--optimized-executable`.
