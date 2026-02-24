# SBD Closed-Loop Workflow on hpc-prefect (Miyabi/Fugaku)

This workflow performs SQD closed-loop optimization and runs SBD diagonalization
jobs on Miyabi or Fugaku through the current `hpc-prefect` architecture.

The old `prefect-sbd` / `prefect-miyabi` runtime path is replaced by:

- `hpc_prefect_blocks` (`CommandBlock`, `ExecutionProfileBlock`, `HPCProfileBlock`)
- `hpc_prefect_executor.run_job_from_blocks(...)`
- local `SBD Solver Job` block (`sbd.solver_job.SBDSolverJob`)

Compatibility notes from old `prefect_sbd.sbd_job`:

- keeps the same block slug: `sbd_solver_job`
- keeps the same solver arguments (`task_comm_size`, `adet_comm_size`, `bdet_comm_size`, `block`, `iteration`, `tolerance`, `carryover_ratio`, `solver_mode`)
- keeps the same expected output files (`davidson_energy.txt`, `occ_a.txt`, `occ_b.txt`, `carryover.bin`)

## Prerequisites

- You can submit HPC jobs on your target system (`qsub` / `qstat` for Miyabi, `pjsub` / `pjstat` for Fugaku).
- Prefect API is reachable (On-Prem or Cloud profile is active).
- `QuantumRuntime` block exists (for example `ibm-runner`).
- SBD executable (`diag`) is already built on your target HPC system and you know its absolute path.

Build examples:

```bash
# Miyabi
cd /work/gz00/z12345/hpc-prefect/algorithms/sbd/native
bash ./build_sbd.sh
realpath ./diag

# Fugaku
cd /work/gz00/z12345/hpc-prefect/algorithms/sbd/native
bash ./build_sbd_fugaku.sh
realpath ./diag
```

## Install

From repository root:

```bash
cd /work/gz00/z12345/hpc-prefect
source ~/venv/prefect/bin/activate

# hpc-prefect local packages
uv pip install --no-deps \
  -e packages/hpc-prefect-core \
  -e packages/hpc-prefect-adapters \
  -e packages/hpc-prefect-blocks \
  -e packages/hpc-prefect-executor

# algorithm dependencies
uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/sbd
```

## Create SBD Blocks (Miyabi/Fugaku)

Use config file:

```bash
cd /work/gz00/z12345/hpc-prefect
cp algorithms/sbd/sbd_blocks.example.toml algorithms/sbd/sbd_blocks.toml
vim algorithms/sbd/sbd_blocks.toml
```

Run block generator:

Miyabi:

```bash
python algorithms/sbd/create_blocks.py --config algorithms/sbd/sbd_blocks.toml --hpc-target miyabi
```

Fugaku:

```bash
python algorithms/sbd/create_blocks.py --config algorithms/sbd/sbd_blocks.toml --hpc-target fugaku
```

`sbd_executable` should point to your built binary, for example:

```toml
sbd_executable = "/work/gz00/z12345/hpc-prefect/algorithms/sbd/native/diag"
```

For memory stability on Miyabi, start from:

```toml
mpiprocs = 4
mpi_options = ["-np", "4"]
task_comm_size = 1
adet_comm_size = 1
bdet_comm_size = 1
block = 4
iteration = 1
tolerance = 0.01
carryover_ratio = 0.1
```

Then increase only if your target case runs without OOM.

This creates:

- `CommandBlock` (default: `cmd-sbd-diag`)
- `ExecutionProfileBlock` (default: `exec-sbd-mpi` for Miyabi, `exec-sbd-fugaku` for Fugaku)
- `HPCProfileBlock` (default: `hpc-miyabi-sbd` for Miyabi, `hpc-fugaku-sbd` for Fugaku)
- `SBD Solver Job` block (default: `davidson-solver`)
- Prefect Variable `sqd_options`

## Run flow

Local execution from JSON parameters:

```bash
python algorithms/sbd/exec.py algorithms/sbd/default_params/test_n2.json
```

Deploy entrypoint:

```bash
sbd-deploy
```

## Optional integrations

- If `s3-sqd` block is configured (`prefect-aws` S3 block), intermediate arrays
  are saved to object storage.
- If not configured, data falls back to Prefect local storage.
