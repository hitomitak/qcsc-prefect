# Run the UHF-SQD Closed-loop Workflow (qcsc-prefect)

This tutorial runs `uhf-sqd`, the closed-loop SQD workflow on a broken-symmetry UHF (BS-UHF)
reference, on the NH2 doublet that ships with the workflow. It runs on your workstation first,
with a random quantum source, and then submits the SBD diagonalizations to a Slurm cluster: on
CPUs, or on ROQUO on one or several GPUs per solve. The last step samples on an IBM Quantum QPU.

`uhf-sqd` is the unrestricted counterpart of the SBD closed loop (`sqd_de`). If you have run
[Run SBD Closed-loop Workflow on Slurm](run_sbd_closed_loop_workflow_slurm.md), most steps will
look familiar. The differences are in the reference, the solver binary and the block names.

> **Scope of this release**
>
> | Path | Status |
> |---|---|
> | Local CPU, `random` source (Steps 3 to 5) | verified bit for bit against the source workflow |
> | ROQUO, one GPU per solve (Step 7) | run end to end on 2026-11-25 |
> | ROQUO, several GPUs per solve on one node (Step 8) | **not yet run on ROQUO** |
> | IBM Quantum QPU through QRMI, `scheduler = "iqp"` (Step 9) | **not yet run on hardware** |
> | CPU Slurm (Step 6) | documented, **not yet run end to end** |
> | QRMI through Slurm and SPANK (end of Step 9) | described, not run for `uhf-sqd` |
>
> The native CPU driver is compared with FCI in CI on 1 and 2 MPI ranks. The whole workflow has
> been run with one rank per solve only, except as stated in Step 8. Other GPU sites, several
> nodes per solve, and running the flow and its solves inside one allocation are not supported.

## Prerequisites

- A clone of `qcsc-prefect` and a Python 3.12 environment, as in Steps 1 and 2 of the
  [Slurm tutorial](run_sbd_closed_loop_workflow_slurm.md).
- A running Prefect server, or Prefect Cloud, that your client is connected to. For a Slurm login
  node, see [How to Run and Access a Prefect Server on a Slurm Login Node](../howto/howto_setup_prefect_server_slurm.md).
- An MPI C++ compiler and OpenBLAS, to build the CPU SBD driver.
- For Step 6 only: a Slurm cluster where you can submit jobs. For Steps 7 and 8: an account on
  ROQUO.
- For Step 9: an IBM Cloud API key with access to an IBM Quantum instance, and a Linux workflow
  host that reaches IBM Quantum (QRMI publishes Linux wheels only).

## 0. What changes from the SBD closed loop?

| | `sqd_de` (restricted) | `uhf-sqd` (unrestricted) |
|---|---|---|
| Reference | RHF orbitals; alpha and beta share one orbital set | BS-UHF from the FCIDUMP: separate alpha and beta orbitals, spin-broken when that is lower |
| Ansatz | spin-balanced LUCJ | spin-unbalanced LUCJ |
| Subsampling | one determinant list for both spins | separate alpha and beta lists, each with its own carryover |
| SBD driver | `diag` | `diag_uhf`, built with `UHF=1` (SBD's `-D_UHF`) |
| Solver block | `SBDSolverJob` with restricted integrals | `SBDSolverJob(integrals="unrestricted")`, default name `sbd-uhf-solver` |
| Shots | the `sqd_options` Variable | the flow parameter `sampler.shots` |
| Default source Block | `quantum-source` | `uhf-sqd-random` |

The input is the same kind of file: a **restricted, spatial-orbital FCIDUMP** with all orbitals
active.

- `MS2` in its header sets `n_alpha - n_beta`.
- ECORE is a constant term, added once to the energies.

The workflow refuses an FCIDUMP that is already in a UHF layout (`IUHF` other than 0, or the SBD
interleaved marker) before reading its integrals.

The restricted and the unrestricted SBD drivers read different integral layouts. **Never point a
UHF block at `diag`, or a restricted block at `diag_uhf`.** Two checks catch a mismatch:

- **The driver.** A driver built from the current `main.cc` refuses an FCIDUMP in the other
  layout before any work.
- **`SBDSolverJob`.** After each job, it compares the layout the driver reported in
  `sbd_build_info.txt` with the input. An unrestricted run stops with an error if the report is
  missing or does not match.

A driver built before this check writes no report. For **restricted** input such a run only logs
a warning, and its energy is not verified. A `-D_UHF` binary of that age registered as the
restricted executable would even complete with a wrong Hamiltonian. Rebuild all SBD drivers from
the current scripts.

The UHF blocks have their own default names, so creating them does not overwrite the `sqd_de`
blocks.

## 1. Where things run

```text
workflow host (your workstation or the login node)
  ├─ Prefect flow uhf_sqd_de (ThreadPoolTaskRunner)
  │    ├─ BS-UHF reference, UCCSD amplitudes      (PySCF, in the flow process)
  │    ├─ DE walkers: sample → recover → subsample (in the flow process)
  │    └─ SBD solve                                 → local process or Slurm job (diag_uhf)
  └─ Prefect server / Cloud: blocks, flow runs, telemetry artifact
```

The blocks:

| Block | Name (default) | Content |
|---|---|---|
| `SBDSolverJob` | `sbd-uhf-solver` | Davidson settings; refers to the three blocks below |
| `CommandBlock` | `cmd-sbd-uhf-solver` | the `diag_uhf` command |
| `ExecutionProfileBlock` | `exec-sbd-uhf-<target>` | MPI ranks, launcher, wall time |
| `HPCProfileBlock` | `hpc-<target>-sbd-uhf` | scheduler settings and the executable path |
| `QuantumSourceBlock` | `uhf-sqd-random` | where the bitstrings come from (random here) |

## 2. Tutorial steps

### Step 1. Install the workflow

From the repository root, in your Python environment:

```bash
uv pip install -e algorithms/closed_loop/uhf-sqd
```

This installs the `uhf_sqd` package and its local dependencies: `qcsc-prefect-sbd`,
`qcsc-prefect-qiskit` and `qcsc_workflow_utility`. It also installs two commands,
`uhf-sqd-create-blocks` and `uhf-sqd-deploy`.

### Step 2. Build the unrestricted SBD driver

Build `diag_uhf` in an environment compatible with the machines that will run it: your
workstation for Step 3, the compute nodes for Step 6.

```bash
cd packages/qcsc-prefect-sbd/native
UHF=1 bash build_sbd_slurm.sh
realpath diag_uhf
```

The script fetches the pinned SBD revision if needed, compiles `main.cc` with `-D_UHF`, and ends
with `Build completed: .../native/diag_uhf`. Without `UHF=1`, the same script builds the
restricted `diag`; the two can live side by side. On Miyabi and Fugaku, use `build_sbd.sh` and
`build_sbd_fugaku.sh` with `UHF=1`. For ROQUO's GPUs, Steps 7 and 8 build `diag-gpu_uhf` and
`diag-gpu_uhf-mpi` with `build_sbd_gpu.sh`.

Record the absolute path of `diag_uhf`; it is the `sbd_executable` below.

> **Note**
>
> A driver reports its integral layout in `sbd_build_info.txt` in each job directory. A UHF
> driver built before that check, for example a `diag_uhf` from an older checkout, does not.
> The first solve then fails with
> `The SBD binary did not report its integral layout (sbd_build_info.txt is missing)`. Rebuild
> it with the current script.

### Step 3. Validate on your workstation with a random source

Create `uhf_sqd_local.toml`:

```toml
[solver]
hpc_target = "local"
root_dir = "/path/to/work/uhf-sqd-jobs"
sbd_executable = "/path/to/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag_uhf"

[quantum_sources.uhf-sqd-random]
config = { kind = "random" }
```

With `hpc_target = "local"`, each solve runs `diag_uhf` directly as one process.

Create the blocks:

```bash
uhf-sqd-create-blocks uhf_sqd_local.toml
```

The command prints the block names. Unless you set them in `[solver]`, they are the UHF
defaults: `sbd-uhf-solver`, `cmd-sbd-uhf-solver`, `exec-sbd-uhf-local` and `hpc-local-sbd-uhf`.

Two things about re-running it:

- The solver blocks are updated in place.
- An existing `QuantumSourceBlock` of the same name is not replaced, because another workflow may
  use it. Creation stops with a message. Remove the table to keep the Block, or add
  `overwrite = true` to replace it.

Run the flow on the NH2 doublet that ships with the workflow (7 orbitals, 5 + 4 electrons):

```python
from uhf_sqd.flow_params import FlowParameters
from uhf_sqd.main import uhf_sqd_de

energy = uhf_sqd_de(parameters=FlowParameters(
    fcidump="/path/to/qcsc-prefect/algorithms/closed_loop/uhf-sqd/data/fcidump_NH2_STO3G.txt",
))
print(energy)
```

The flow logs the reference first:
`Reference: E_UHF=-54.8302183653 <S^2>=0.7564 norb=7 nelec=(5, 4) key=v2|...`. The returned
value is the best total energy (ECORE included), about -54.8726 Eh; the FCI energy of this
FCIDUMP is -54.872572 Eh.

The defaults run 4 DE walkers for one trial. The subspace (`sqd_dim`, default 1 000 000) covers
the whole space of this small molecule.

> **Note: the default Davidson settings are loose.**
>
> `uhf-sqd-create-blocks` uses the source workflow's defaults: block 4, iteration 1, tolerance
> 1e-2, `carryover_ratio` 0.1, `carryover_type` 0. They stop the solve before convergence. For
> stretched H6, the result is 9.61 mEh above FCI.
>
> When you need converged subspace energies, add tighter settings to `[solver]`:
>
> ```toml
> block = 30
> iteration = 60
> tolerance = 1e-10
> ```

### Step 4. Read the results

In the Prefect UI, open the flow run. Its table artifact `uhf-sqd-telemetry` has one row per
walker and trial:

- `energy`, `net_subspace_dim` and `num_post_determinants` of the walker's best recovery pass;
- `recovery_trace`, one entry per recovery pass, with the per-batch energies, the alpha and beta
  occupancies and the excitation counts of the determinants;
- `sqd_data`, the stored arrays of that pass: the LUCJ parameters, the raw and recovered
  bitstrings, the alpha and beta determinants, the occupancies and the carryover.

The arrays are stored in the S3 block `s3-sqd` if it exists. Otherwise they are under
`$SBD_LOCAL_STORAGE_DIR` (default `$PREFECT_HOME/storage`), in
`sqd_data/uhf_sqd_de/<flow run name>/`.

Each SBD solve has its own job directory under `root_dir`: `fcidump.txt`, `AlphaDets.bin`,
`BetaDets.bin`, `davidson_energy.txt`, `occ_a.txt`, `occ_b.txt`, the carryover files and
`sbd_build_info.txt` (`integrals=SBD_UHF_INTERLEAVED`). The solver energy is electronic: the
workflow adds ECORE.

### Step 5. Tune the run

All options are flow parameters; unknown keys are rejected. The most used ones:

| Parameter | Default | Effect |
|---|---|---|
| `sqd_dim` | 1 000 000 | subspace dimension, about `sqrt(sqd_dim)` determinants per spin |
| `n_recovery_steps` | 1 | self-consistent recovery passes per walker, each fed by the previous pass's occupancies |
| `n_batches` | 1 | independent subspaces per pass (K-batch SQD); the lowest energy is kept, the mean occupancies feed the next pass |
| `de_params.iterations` | 1 | DE trials. Mutation needs at least 4 walkers (`de_params.num_walkers`) |
| `sampler.shots` | 10 000 | shots per walker |
| `reference.af_groups` | none | an atom-localized antiferromagnetic guess; see below |
| `reference.cache_dir` | `~/.cache/qcsc_uhf` | cache of converged BS-UHF orbitals, keyed by the FCIDUMP contents, the guess and the electron numbers; `null` disables it |

For example, a run with a truncated subspace, two recovery passes, two batches and two DE trials:

```python
FlowParameters(
    fcidump=".../fcidump_NH2_STO3G.txt",
    sqd_dim=400,
    n_recovery_steps=2,
    n_batches=2,
    de_params={"iterations": 2},
)
```

**Broken-symmetry guess.** By default, the reference is the lower of two solutions:

- the UHF solution reached by following internal instabilities;
- the solution from an energy-ordered antiferromagnetic (AF) guess.

For spin-coupled clusters, an atom-localized guess can reach a lower broken-symmetry state. Give
it as fragments of 0-based FCIDUMP orbitals, for example for two iron centers:

```python
reference={"af_groups": {
    "fe1": [2, 3, 4, 5, 6], "fe2": [7, 8, 9, 10, 11],
    "up": ["fe1"], "down": ["fe2"],
}}
```

- `up` and `down` fragments are seeded spin-up and spin-down. All other fragments are seeded
  doubly occupied.
- Optional keys: `free` lists fragments seeded half-filled, and `pol` (default 1.0) scales the
  polarization.

The guess is used only if its solution is lower. The log line `Reference: E_UHF=... <S^2>=...`
shows the reference that was used.

### Step 6. Run the solver on Slurm and deploy (not yet run end to end)

Build `diag_uhf` on the cluster (Step 2), then create `uhf_sqd_slurm.toml` from
[uhf_sqd_blocks.example.toml](https://github.com/qiskit-community/qcsc-prefect/blob/main/algorithms/closed_loop/uhf-sqd/uhf_sqd_blocks.example.toml):

```toml
[solver]
hpc_target = "slurm"
root_dir = "/shared/work/qcsc-prefect/uhf-sqd"
sbd_executable = "/path/to/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag_uhf"
queue = "<partition>"
num_nodes = 1
mpiprocs = 4
walltime = "01:00:00"
# Names of their own, so that the local blocks of Step 3 stay as they are.
solver_block_name = "sbd-uhf-solver-slurm"
command_block_name = "cmd-sbd-uhf-solver-slurm"
```

- **No `[quantum_sources]` table.** The `uhf-sqd-random` Block from Step 3 is reused by name.
  Listing it again would stop the creation, because that Block already exists.
- **Block names.** The execution and HPC profiles get `exec-sbd-uhf-slurm` and
  `hpc-slurm-sbd-uhf`, which already differ from the local ones. The solver and command blocks
  need the explicit names above. Their defaults are the same for every target, so without them
  the Slurm settings would replace the local `sbd-uhf-solver` of Step 3.
- **Scheduler settings.** `[solver]` takes the keyword arguments of
  `qcsc_prefect_sbd.create_sbd_blocks`, for example `project`, `modules`, `pre_commands` and
  `mpi_options`. The [Slurm tutorial](run_sbd_closed_loop_workflow_slurm.md) explains them.
- **`root_dir`** must be on a file system shared by the workflow host and the compute nodes.

Create the blocks, then start the deployment from the repository root. Use `screen`, `tmux` or a
service for a long-running deployment:

```bash
uhf-sqd-create-blocks uhf_sqd_slurm.toml
uhf-sqd-deploy
```

The deployment is `uhf_sqd_de/uhf_sqd_de`. From the Prefect UI, choose **Run → Custom run** and
set at least:

| Field | Value |
|---|---|
| FCIDump File | `/path/to/qcsc-prefect/algorithms/closed_loop/uhf-sqd/data/fcidump_NH2_STO3G.txt` |
| Quantum Source Block | `uhf-sqd-random` |
| Solver Block Ref | `sbd_solver_job/sbd-uhf-solver-slurm` |

Each walker then submits its solves as Slurm jobs; follow them with `squeue -u "$USER"`.

> **Not yet run end to end**
>
> This CPU Slurm procedure has not been run end to end. The CPU driver itself is compared with
> FCI in CI on 1 and 2 MPI ranks, but a workflow run with `mpiprocs = 4` as above has not been
> compared with the single-process result of Step 3. Run NH2 with the settings of Step 3 and
> compare the energies before relying on the Slurm results. The Slurm path itself (submission,
> monitoring, exit status and logs) was run on ROQUO with GPUs in Steps 7 and 8.

### Step 7. Run the solver on one GPU on ROQUO

On ROQUO, each solve can run on one GB200 GPU. The flow runs on the login node, as in Step 6,
and each solve is a one-GPU Slurm job (`hpc_target = "slurm"`).

**Build `diag-gpu_uhf` on a compute node.** ROQUO's compute nodes are aarch64; its login node
is x86_64. Follow "ROQUO" in
[native/README.md](https://github.com/qiskit-community/qcsc-prefect/blob/main/packages/qcsc-prefect-sbd/native/README.md):

```bash
srun --account=<account> --partition=roquo --gpus-per-node=1 --time=01:00:00 --pty bash
module load cuda/13.2 nvhpc/26.5 && module load hpcx/2.50
export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5/compilers/bin:$PATH
cd packages/qcsc-prefect-sbd/native && UHF=1 bash build_sbd_gpu.sh   # -> diag-gpu_uhf
exit
```

**Create the blocks** from
[uhf_sqd_blocks.roquo-gpu.example.toml](https://github.com/qiskit-community/qcsc-prefect/blob/main/algorithms/closed_loop/uhf-sqd/uhf_sqd_blocks.roquo-gpu.example.toml).
Copy it, then set `project` (your account), `root_dir` (under your home directory, which the
compute nodes see) and `sbd_executable` (the absolute path of `diag-gpu_uhf`):

```bash
cp algorithms/closed_loop/uhf-sqd/uhf_sqd_blocks.roquo-gpu.example.toml uhf_sqd_roquo_gpu.toml
uhf-sqd-create-blocks uhf_sqd_roquo_gpu.toml
```

The file sets:

- the partition `roquo`, one GPU and `solver_mode = "gpu"`;
- 36 OpenMP threads, because ROQUO gives a one-GPU job a quarter node (`[ROQUO-W003]` from
  `sbatch`);
- `srun --mpi=pmix`, and the modules the driver was built with;
- block names of its own (solver block `sbd-uhf-solver-roquo-gpu`).

The Davidson settings are the same loose defaults as on CPUs.

**Run the flow** on the login node, with the solver block of this step. If the random source of
Step 3 is in another Prefect workspace, create it here as in Step 3:

```python
from uhf_sqd.flow_params import FlowParameters
from uhf_sqd.main import uhf_sqd_de

energy = uhf_sqd_de(parameters=FlowParameters(
    fcidump="/path/to/qcsc-prefect/algorithms/closed_loop/uhf-sqd/data/fcidump_NH2_STO3G.txt",
    solver_block_ref="sbd_solver_job/sbd-uhf-solver-roquo-gpu",
))
```

Each walker submits its solve as a Slurm job of one GPU. Each job's output begins with
`[qcsc] integrals layout: SBD_UHF_INTERLEAVED`. The result is the same as on CPUs: about
-54.8726 Eh for NH2.

> **Checked on ROQUO, 2026-11-25** (branch `sbd-gpu-uhf`, SBD `95bf45a`, cuda 13.2, nvhpc 26.5,
> hpcx 2.50, one rank per solve)
>
> - **The NH2 flow** above completed. Its 4 solves ran as 4 one-GPU jobs, about 36 s each
>   including the queue. Its energy, -54.87256711207442, differs by 7e-15 Eh from the CPU run
>   with the same settings.
> - **Solver alone.** `diag-gpu_uhf` and the CPU `diag_uhf` were given the same integrals and
>   determinants, with tight Davidson settings: 36 subspaces of NH2 and stretched H6, including
>   both full spaces. The energies agree within 2.1e-14 Eh and the occupancies within 3.0e-14.
> - **Native tests.** The one-rank cases of the native tests pass on one GPU: energies against
>   FCI, occupancies, and the integral-layout checks.
>
> Not checked: several ranks on one GPU, and larger molecules. Several GPUs per solve are
> Step 8. Each solve waits in the queue on its own; running the flow and its solves inside one
> allocation is not supported yet (issue #85).

### Step 8. Run each solve on several GPUs on ROQUO (not yet run on ROQUO)

A solve can also use several GPUs of one node, one MPI rank per GPU. SBD then splits the
determinants over the ranks, so a subspace that does not fit in the memory of one GPU can still
be solved. The flow is the same as in Step 7; only the driver and the solver blocks change.

This is different from running several solves inside one allocation (issue #85): here each solve
is still one Slurm job, of several GPUs.

> **Not yet run on ROQUO**
>
> The build, the blocks and the job script of this step are tested offline (the job script is
> rendered in the tests), and the CPU driver passes the split-determinant FCI tests on 2 ranks in
> CI. The multi-GPU driver has not been run on ROQUO yet.

**Build `diag-gpu_uhf-mpi` on a compute node**, with the same modules as in Step 7. `MPI=1`
adds NCCL, which the NVIDIA HPC SDK provides, and stages MPI messages through host memory,
because `hpcx/2.50` is not a CUDA-aware MPI:

```bash
srun --account=<account> --partition=roquo --gpus-per-node=1 --time=01:00:00 --pty bash
module load cuda/13.2 nvhpc/26.5 && module load hpcx/2.50
export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5/compilers/bin:$PATH
cd packages/qcsc-prefect-sbd/native && MPI=1 UHF=1 bash build_sbd_gpu.sh   # -> diag-gpu_uhf-mpi
exit
```

The script prints the NCCL directories it found, and stops if the driver has unresolved
libraries. If NCCL is elsewhere, set `NCCL_INCLUDE` and `NCCL_LIB` to `nccl.h` and
`libnccl.so`. See "ROQUO" in
[native/README.md](https://github.com/qiskit-community/qcsc-prefect/blob/main/packages/qcsc-prefect-sbd/native/README.md)
for the defines and for running the native tests on two GPUs.

**Create the blocks** from
[uhf_sqd_blocks.roquo-multigpu.example.toml](https://github.com/qiskit-community/qcsc-prefect/blob/main/algorithms/closed_loop/uhf-sqd/uhf_sqd_blocks.roquo-multigpu.example.toml),
with `project`, `root_dir` and `sbd_executable` set as in Step 7:

```bash
cp algorithms/closed_loop/uhf-sqd/uhf_sqd_blocks.roquo-multigpu.example.toml uhf_sqd_roquo_multigpu.toml
uhf-sqd-create-blocks uhf_sqd_roquo_multigpu.toml
```

| Setting | Value in the example | Why |
|---|---|---|
| `sbd_executable` | `.../native/diag-gpu_uhf-mpi` | the `MPI=1 UHF=1` driver |
| `gpus_per_node`, `mpiprocs` | 2, 2 | one rank per GPU, on one node |
| `ompthreads` | 36 | OpenMP threads per rank; ROQUO gives 36 cores per GPU |
| `task_comm_size`, `adet_comm_size`, `bdet_comm_size` | 1, 1, 2 | SBD's communicator grid; the beta determinants are split over the two ranks. The product must divide the number of ranks |
| `mpi_options` | `--mpi=pmix`, `--cpus-per-task=36` | `srun` does not take `--cpus-per-task` from `sbatch` (Slurm 22.05 and later) |
| block names | `sbd-uhf-solver-roquo-multigpu` and others | the blocks of Step 7 stay as they are |

Each rank sees every GPU of the job, and the driver takes GPU (node-local rank) modulo (number of
GPUs). So do not add `--gpu-bind`. For a whole node, set `gpus_per_node = 4`, `mpiprocs = 4`,
`adet_comm_size = 2` and `bdet_comm_size = 2`, the grid of the UHF-SQD source's ROQUO runs.

**Run the flow** as in Step 7, with the solver block of this step:

```python
energy = uhf_sqd_de(parameters=FlowParameters(
    fcidump="/path/to/qcsc-prefect/algorithms/closed_loop/uhf-sqd/data/fcidump_NH2_STO3G.txt",
    solver_block_ref="sbd_solver_job/sbd-uhf-solver-roquo-multigpu",
))
```

**Check that the ranks used different GPUs.** Each solve's job directory under `root_dir` has the
driver's output in `output.out`. With more than one rank, the driver prints the GPU of each rank:

```bash
grep -h '^\[gpu\]\|integrals layout' /path/to/work/uhf-sqd-multigpu/job_*/output.out | sort | uniq -c
```

Each job should show `[gpu] rank 0/2 uses CUDA device 0`, `[gpu] rank 1/2 uses CUDA device 1`
and `[qcsc] integrals layout: SBD_UHF_INTERLEAVED`. The flow returns about -54.8726 Eh for NH2,
as in Steps 3 and 7.

### Step 9. Sample on an IBM Quantum QPU (not yet run on hardware)

The walkers can sample their LUCJ circuits on an IBM Quantum QPU instead of drawing random bits.
The first route below is QRMI's direct IQP execution (`scheduler = "iqp"`): QRMI runs in the
workflow process and submits one IBM Quantum job per walker and trial, without a session. The
solves use any solver block of the earlier steps.

> **Not yet run on hardware**
>
> The device path is tested offline: the flow runs on NH2 through the real QRMI source, the IQP
> runtime and the compile hooks, with a fake QRMI module and the Target of a fake heavy-hex
> backend (`test_uhf_sqd_device_path.py`). It has not been run on a QPU. The reload of an
> earlier IBM Quantum job by its ID, checked on hardware for the open-loop `sqd` workflow, is not
> part of `uhf-sqd`.

**Where it runs.** Sampling and compilation run in the workflow process, so the workflow host
must be a Linux machine (QRMI publishes Linux wheels only) that reaches the IBM Quantum API and
IBM Cloud IAM, for example the ROQUO login node. The solves run wherever their solver block says.

**1. Install QRMI** in the Python environment of the flow, after the workflow (Step 1):

```bash
uv pip install "qrmi[ibm]"
python -c "import qrmi; print('qrmi', qrmi.__file__)"
```

`uv sync` removes packages that are not in the lock file. If you run it, install QRMI again
afterwards.

**2. Store the API key and choose the instance and the backend** with `qcsc-qrmi login`:

```bash
qcsc-qrmi login
qcsc-qrmi list
```

`login` asks for the IBM Cloud API key, lists the IBM Quantum instances and backends it can use,
and asks for a default instance and backend. It writes the key to `~/.qiskit/qrmi_config.json`
(mode 600, or `$QRMI_CONFIG`) and saves a QRMI Runtime block named `qrmi-runtime` with your
instances and backends. The key is never stored in a block. See
[QRMI Sampler](../howto/howto_qrmi_slurm_sampler.md#set-up-access-with-qcsc-qrmi-login).

**3. Save a QRMI Runtime block for direct IQP execution.** A new `qrmi-runtime` block is saved for
Slurm, and the open-loop `sqd` workflow may use it, so save one of its own for `uhf-sqd`. It keeps
the instance and the backend that `login` saved; `scheduler = "iqp"` uses no Slurm settings:

```python
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

login = QrmiRuntimeBlock.load("qrmi-runtime")
settings = login.settings() | {
    "scheduler": "iqp",
    "root_dir": None,
    "command_block_name": None,
    "execution_profile_block_name": None,
    "hpc_profile_block_name": None,
    # "resource": "ibm_kingston",   # another backend of the instance
}
QrmiRuntimeBlock(**settings).save("uhf-sqd-iqp", overwrite=True)
print(QrmiRuntimeBlock.load("uhf-sqd-iqp").instance, QrmiRuntimeBlock.load("uhf-sqd-iqp").resource)
```

Check that the instance can use the backend before any job:

```bash
qcsc-qrmi check --block uhf-sqd-iqp
```

**4. Save the source Block.** Put the table in its own file, for example `uhf_sqd_qpu.toml`:

```toml
[quantum_sources.uhf-sqd-qpu]
config = { kind = "qrmi", block_name = "uhf-sqd-iqp" }
```

```bash
uhf-sqd-create-blocks uhf_sqd_qpu.toml
```

A file with only `[quantum_sources]` creates or updates the source Blocks and leaves the SBD
blocks of Steps 3, 6, 7 and 8 unchanged. A `[solver]` table, even an empty one, is always
processed: an empty one fails because the required settings are missing. `tags` in `config`, for
example `tags = ["res: qpu"]`, are applied to the sampling task, so that a Prefect concurrency
limit can use them.

**5. Run a small smoke test**: one walker, one trial and few shots, so that one small QPU job is
submitted. On the ROQUO login node, the one-GPU solver block of Step 7 is a natural choice; on a
workstation, the local block of Step 3.

```python
from uhf_sqd.flow_params import FlowParameters
from uhf_sqd.main import uhf_sqd_de

energy = uhf_sqd_de(parameters=FlowParameters(
    fcidump="/path/to/qcsc-prefect/algorithms/closed_loop/uhf-sqd/data/fcidump_NH2_STO3G.txt",
    quantum_source="uhf-sqd-qpu",
    solver_block_ref="sbd_solver_job/sbd-uhf-solver-roquo-gpu",
    sampler={"shots": 1000},
    de_params={"num_walkers": 1, "iterations": 1},
))
print(energy)
```

NH2 has 7 orbitals, so the circuit has 14 qubits, measured into `meas` (14 bits) and, with reset
mitigation (the default), first into `test` (14 bits). Before the job is submitted, the circuit is
compiled on the workflow host for the backend's Target, with a SABRE layout search of
`circ_params.sabre_layout_trials` (default 1024) trials; this can take a few minutes. Lower it
for a quick check.

**6. Check the run.**

- **The flow log**, in this order: `Reference: E_UHF=...`, `Sampling from <backend>
  (shots=1000)`, `Submitted IQP job <job ID> to <backend>`, `IQP job <job ID> completed on
  <backend>`, `[persist] <n>-shot pool (trial 0 walker 0) saved: ...`, then the solve. The job ID
  is also on the IBM Quantum dashboard.
- **The flow run** is Completed in the Prefect UI, and the returned energy is a finite number.
  This step checks the workflow, not the accuracy: a 1000-shot, one-walker run is not expected
  to reach the energy of Step 3.
- **The `uhf-sqd-telemetry` artifact** has one row. Its device fields:
  - `reset_test_passed_shots`: the shots whose `test` register was all zeros;
  - `shot_retention_rate`: those shots over `shots`;
  - `reset_mitigation_fallback`: `true` if no shot passed and all were used. On a QPU this should
    be `false`;
  - `raw_samples_path`: the saved sample pool (`raw_samples_t00_w0_<id>.npz`), with the FCIDUMP
    sha256, the reference key and the bit layout.
- **The solve** completed as in the step of its solver block.

**If something fails**

| Message | Meaning |
|---|---|
| `scheduler 'iqp' needs an API key: run qcsc-qrmi login` | this host has no `qrmi_config.json` with a key, and `QRMI_IBM_QCS_IAM_APIKEY` is not set |
| `ModuleNotFoundError: No module named 'qrmi'` | QRMI is not installed in the flow's environment (step 1), or the host is not Linux |
| `scheduler="iqp" does not use root_dir, ...` | the block still has Slurm settings; save it as in step 3 |
| `IqpJobFailed: IQP job <id> on <backend> ended as ERROR` | IBM Quantum rejected or failed the job; see the job on the dashboard |
| `IqpSubmissionUnknown` | submission raised, so it is unknown whether a job exists. It is not resubmitted; check the dashboard before running again |

If the flow process stops while it waits for a job, the job can keep running on IBM Quantum;
cancel it on the dashboard. A job that has finished is never stopped.

**Notes on the device path**

- **Compilation.** The workflow compiles each LUCJ circuit with its own SABRE layout search, using
  the hook `uhf_sqd.transpilation:transpile`. With `scheduler = "iqp"` it runs in the workflow
  process.
- **Reset mitigation** keeps the shots whose `test` register is all zeros. If none pass, all
  shots are used and a warning is logged.
- **Sample pool.** Before any diagonalization, each walker saves its shots after reset mitigation.
  Loading a pool back into a run is not supported yet (`UHF_SQD_DEFERRED_FEATURES.md`, section 2).
- **Layouts and DE.** The source reused one SABRE layout for a whole flow run; `uhf-sqd` searches
  one per circuit. Walkers' circuits differ only in angles, so the layouts are mostly the same,
  and the extra searches cost compile time. Where layouts differ, walkers run on different
  physical qubits, and energy differences between walkers include qubit quality. Check the
  layouts before comparing walkers on hardware.

**Through Slurm and SPANK.** A QRMI Runtime block with `scheduler = "slurm"` samples inside a
Slurm job that holds the QPU through the QRMI SPANK plugin; the same source Block kind refers to
it. This needs an administrator: the SPANK plugin and `qrmi_config.json` on the compute nodes,
and QRMI, `task_runner`, `qcsc-prefect-qiskit` and `uhf_sqd` (for the compile hook) in the
compute-node Python. See [QRMI Sampler](../howto/howto_qrmi_slurm_sampler.md). This route is
used by the open-loop `sqd` workflow; it has not been run for `uhf-sqd`.

## 3. Troubleshooting

### `the FCIDUMP layout is RESTRICTED (no QCSC_INTEGRALS key) but this binary reads SBD_UHF_INTERLEAVED`

A restricted file reached `diag_uhf`. This happens when a restricted solver block points at the
UHF driver. Check `HPCProfileBlock.load("<name>").executable_map`, and recreate the restricted
blocks with `diag`.

### `Solver block ... has integrals='restricted'`

The flow was given a restricted solver block, for example `sqd_de`'s `davidson-solver`. Use
the block created by `uhf-sqd-create-blocks` (default `sbd_solver_job/sbd-uhf-solver`).

### `The SBD binary did not report its integral layout`

`diag_uhf` was built before the integral-layout check. Rebuild it (Step 2).

### `IUHF=1 is a MOLPRO-style UHF FCIDUMP` or `QCSC_INTEGRALS=SBD_UHF_INTERLEAVED is not a restricted FCIDUMP`

The input FCIDUMP is already unrestricted. Give the workflow the restricted FCIDUMP from which
the UHF reference should be built.

### `A QuantumSourceBlock named ... already exists`

`uhf-sqd-create-blocks` does not replace a source Block that may belong to another workflow. Use
another name, refer to the existing Block by name in the flow parameters, or add
`overwrite = true` to that source's table.

### The energy is above the expected value

Check the Davidson settings first (the note in Step 3), then `sqd_dim`. With
`sqd_dim` smaller than the whole space, raise `n_recovery_steps` and `n_batches`.

### Site-specific Slurm profiles

Clusters with their own submission rules need site-specific settings in `[solver]`. Examples are
in-allocation `srun` launches, reservations and nodes without outbound access. ROQUO as a Slurm
profile is tracked in issue #81.
