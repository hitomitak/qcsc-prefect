# Run SBD Closed-Loop Workflow on Miyabi (hpc-prefect)

This tutorial explains how to run the SBD closed-loop workflow with the current
`hpc-prefect` architecture.

Target scope in this version:

- Miyabi execution path only
- block creation by script (no manual solver block editing in UI)

---

## Prerequisites

Before starting, complete:

- [Create Your QCSC Workflow with Prefect](./create_qcsc_workflow.md)
- [How to Set Up IBM Quantum Access Credentials for Prefect](../howto/howto_setup_prefect_qiskit.md)

Also confirm:

- `qsub` / `qstat` are available on Miyabi
- a `QuantumRuntime` block exists (for example `ibm-runner`)
- SBD executable `diag` has been built on Miyabi and you know its absolute path

> [!IMPORTANT]
> Replace `g00` and `z12345` with your actual group/account.

---

## 0. Files used in this tutorial

- `../../algorithms/sbd/create_blocks.py`
- `../../algorithms/sbd/sbd_blocks.example.toml`
- `../../algorithms/sbd/default_params/test_n2.json`
- `../../algorithms/sbd/sbd/main.py`
- `../../algorithms/sbd/sbd/sqd.py`
- `../../algorithms/sbd/sbd/solver_job.py`

---

## 1. Log in to MDX workflow client

<img src="./images/icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh -A z12345@qii-kawasaki-miyabi-cli.cspp.cc.u-tokyo.ac.jp
```

Activate your environment:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
source ~/venv/prefect/bin/activate
```

---

## 2. Install required packages

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
cd /work/g00/z12345/hpc-prefect

uv pip install --no-deps \
  -e packages/hpc-prefect-core \
  -e packages/hpc-prefect-adapters \
  -e packages/hpc-prefect-blocks \
  -e packages/hpc-prefect-executor

uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/sbd
```

---

## 3. Prepare SBD executable path on Miyabi

Build `diag` under `algorithms/sbd/native`:

<img src="./images/icon-miyabi.png" alt="miyabi" width="50"/><br>
```bash
cd /work/g00/z12345/hpc-prefect/algorithms/sbd/native
bash ./build_sbd.sh
realpath ./diag
```

Use this absolute path in Step 4 (`sbd_executable`).

---

## 4. Generate Prefect blocks by script

Copy config template:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
cd /work/g00/z12345/hpc-prefect
mkdir -p /work/g00/z12345/sbd_jobs
cp algorithms/sbd/sbd_blocks.example.toml algorithms/sbd/sbd_blocks.toml
```

Edit `algorithms/sbd/sbd_blocks.toml` and set:

- `project`
- `queue`
- `work_dir`
- `sbd_executable`
- `mpiprocs = 4`
- `mpi_options = ["-np", "4"]`
- `task_comm_size = 1`
- `adet_comm_size = 1`
- `bdet_comm_size = 1`
- `block = 4`
- `iteration = 1`
- `tolerance = 0.01`
- `carryover_ratio = 0.1`

For Miyabi, start with low MPI parallelism (`-np 4`) to avoid memory OOM, then scale up only after confirming stability.

Run block creation:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
python algorithms/sbd/create_blocks.py --config algorithms/sbd/sbd_blocks.toml
```

This creates (default names):

- `CommandBlock`: `cmd-sbd-diag`
- `ExecutionProfileBlock`: `exec-sbd-mpi`
- `HPCProfileBlock`: `hpc-miyabi-sbd`
- `SBD Solver Job`: `davidson-solver`
- Prefect Variable: `sqd_options`

---

## 5. Check solver block and options

Check solver block:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
prefect block inspect sbd_solver_job/davidson-solver
```

Check options variable:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
prefect variable inspect sqd_options
```

Optional override example:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
prefect variable set sqd_options '{"params":{"shots":500000}}' --overwrite
```

---

## 6. Deploy the SBD flow

Start a detached session:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
screen -S sbd-workflow
```

Deploy:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
cd /work/g00/z12345/hpc-prefect
sbd-deploy
```

Detach screen: `<ctrl> + a`, then `d`.

---

## 7. Run from Prefect Web Portal

Open deployment `riken-sqd-de/riken_sqd_de`, then click:

- **Run**
- **Custom run**

Set at least:

- `FCIDump File`: path to FCIDUMP file (for example `/work/gz00/z12345/hpc-prefect/algorithms/sbd/data/fcidump_N2_MO.txt`)
- `SQD Subspace Dimension`: start from `200000` for a fast run
- `Differential Evolution Iterations`: start from `1` for a fast run
- `Solver Block Ref`: `sbd_solver_job/davidson-solver`

Submit the run.

---

## 8. What happens in this architecture

1. `walker_sqd` loads `SBDSolverJob` by name.
2. `SBDSolverJob.run(...)` prepares `fcidump.txt` and `AlphaDets.bin`.
3. Job is submitted by `run_job_from_blocks(...)` using:
   - `CommandBlock`
   - `ExecutionProfileBlock`
   - `HPCProfileBlock`
4. SBD output files (`davidson_energy.txt`, `occ_a.txt`, `occ_b.txt`, `carryover.bin`) are parsed.
5. Flow stores telemetry artifact (`sqd-telemetry`).

This keeps workflow code stable while HPC settings are controlled by block instances.

---

## 9. Cleanup

When finished:

- stop serving process (screen session)
- optionally follow [How to shutdown the workflow](../howto/howto_shutdown_workflow.md)
