# Run the UHF-SQD Closed-loop Workflow (qcsc-prefect)

This tutorial runs `uhf-sqd`, the closed-loop SQD workflow on a broken-symmetry UHF (BS-UHF)
reference. It runs on your workstation first, with a random quantum source, and then submits the
SBD diagonalizations to a Slurm cluster.

`uhf-sqd` is the unrestricted counterpart of the SBD closed loop (`sqd_de`). If you have run
[Run SBD Closed-loop Workflow on Slurm](run_sbd_closed_loop_workflow_slurm.md), most steps will
look familiar. The differences are in the reference, the solver binary and the block names.

## Prerequisites

- A clone of `qcsc-prefect` and a Python 3.12 environment, as in Steps 1 and 2 of the
  [Slurm tutorial](run_sbd_closed_loop_workflow_slurm.md).
- A running Prefect server, or Prefect Cloud, that your client is connected to. For a Slurm login
  node, see [How to Run and Access a Prefect Server on a Slurm Login Node](../howto/howto_setup_prefect_server_slurm.md).
- An MPI C++ compiler and OpenBLAS, to build the SBD driver.
- For Step 6 only: a Slurm cluster where you can submit jobs.

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
`build_sbd_fugaku.sh` with `UHF=1`.

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

### Step 6. Run the solver on Slurm and deploy

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

> **Note**
>
> This Slurm procedure has not been run end to end yet. Nor has a comparison of multi-rank
> energies with the single-process result of Step 3. Compare them yourself before relying on
> the Slurm results.

### Step 7. Sample on a quantum device

To sample on hardware, save a `QuantumSourceBlock` of kind `qrmi` that refers to a QRMI runtime
Block, and pass its name as `quantum_source`. Set up the runtime Block as in
[How to use a quantum source](../howto/howto_use_quantum_source.md).

Put the table in its own file, for example `uhf_sqd_device.toml`, and pass it to the same command:

```toml
[quantum_sources.uhf-sqd-device]
config = { kind = "qrmi", block_name = "qrmi-runtime", tags = ["res: quantum"] }
```

```bash
uhf-sqd-create-blocks uhf_sqd_device.toml
```

A file with only `[quantum_sources]` creates or updates the source Blocks and leaves the SBD
blocks of Steps 3 and 6 unchanged. A `[solver]` table, even an empty one, is always processed:
an empty one fails because the required settings are missing.

Then run the flow with `quantum_source="uhf-sqd-device"`.

- **Sharing a source Block.** A source Block that another workflow already uses can be named
  directly in the flow parameters. You do not need to create it again.
- **Compilation.** The workflow compiles each LUCJ circuit with its own SABRE layout search, using
  the hook `uhf_sqd.transpilation:transpile`. Install `uhf_sqd` wherever the QRMI compiler runs.
- **Reset mitigation.** It keeps the shots whose `test` register is all zeros. If none pass, all
  shots are used and a warning is logged. The telemetry records `reset_test_passed_shots`,
  `reset_mitigation_fallback` and `shot_retention_rate`.
- **Sample pool.** Before any diagonalization, each walker saves its shots after reset mitigation
  as `raw_samples_t<trial>_w<walker>_<id>.npz`. The telemetry field `raw_samples_path` records
  where. Each pool records the FCIDUMP sha256, the reference key and the bit layout, so that it
  can be matched with its reference later.

> **Note**
>
> The random path of `uhf-sqd` is verified bit for bit against the source workflow it was ported
> from. The device path is not yet verified on hardware.
>
> - **Layout search.** The source reused one SABRE layout for a whole flow run; `uhf-sqd`
>   searches one per circuit. Walkers' circuits differ only in angles, so the layouts are mostly
>   the same and the extra searches cost compile time (lower `sabre_layout_trials` if that
>   matters).
> - **Effect on DE.** Where layouts differ, walkers run on different physical qubits. Energy
>   differences between walkers then include qubit quality.
> - **Before comparing walkers on hardware,** check the layouts that the runs used.

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
