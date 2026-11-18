# QCSC Prefect SBD

Shared Python-side SBD Davidson solver integration for qcsc-prefect workflows.

This package provides:

- `SBDSolverJob` as a Prefect block (block type slug `sbd_solver_job`)
- `SBDResult`, the parsed solver output
- `create_sbd_blocks` / `register_sbd_block_types`: create the command,
  execution-profile, HPC-profile and solver blocks for `local`, `slurm`, `miyabi` or
  `fugaku`
- SBD input/output file handling and scheduler execution through qcsc-prefect blocks

It is intended to be reused by multiple algorithms. Installing this package does not
build, download, vendor, or install the external SBD executable (`diag`). The executable
must be compiled separately for each target HPC environment.

Source-tree native build helper assets live under:

```text
packages/qcsc-prefect-sbd/native
```

They are not part of the Python package install. See [native/README.md](./native/README.md)
for the build scripts and how to point `sbd_executable` at the separately built binary.

## Usage Example

```python
from qcsc_prefect_sbd import SBDSolverJob, create_sbd_blocks

create_sbd_blocks(
    hpc_target="slurm",
    queue="normal",
    root_dir="/shared/sbd-jobs",
    sbd_executable="/abs/path/to/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag",
    num_nodes=2,
    mpiprocs=4,
    mpi_options=["--mpi=pmix"],
)

# Inside a flow:
solver = await SBDSolverJob.load("sbd-solver")
result = await solver.run(
    ci_strings=(alpha_strings, beta_strings),
    one_body_tensor=h1,
    two_body_tensor=h2,
    norb=norb,
    nelec=nelec,
)
print(result.energy, result.carryover_bitstrings.shape)
```

`create_sbd_blocks` checks the request before it saves any block: local execution takes
no `modules` or `pre_commands`, `gpus_per_node` must fit the target, `solver_mode="gpu"`
needs `resource_class="gpu"`, and `task_comm_size x adet_comm_size x bdet_comm_size` must
divide the MPI rank count (`num_nodes x mpiprocs`). The launcher defaults to `single`
(local), `srun` (Slurm), `mpiexec.hydra` (Miyabi) or `mpiexec` (Fugaku).

## Closed and open shells

`SBDSolverJob.run` takes the same arguments as `DiceSHCISolverJob.run`. SBD diagonalizes
in the product of the alpha and beta determinant sets with the restricted integrals, so it
solves closed-shell and restricted open-shell (ROHF, `nelec[0] != nelec[1]`) problems.

- When the alpha and beta sets are the same, SBD reads `AlphaDets.bin` for both spins.
- When they differ (always for `nelec[0] != nelec[1]`), the beta set goes to
  `BetaDets.bin` (`--bdetfile`), and `SBDResult.carryover_bitstrings_b` holds the beta
  carryover. It is None when one set is shared.

`run` raises `ValueError` (see `check_sbd_input`) when:

- the alpha set does not start with its Hartree-Fock determinant, `(1 << nelec[0]) - 1`,
  or the beta set with `(1 << nelec[1]) - 1`;
- `spin_sq` is not `None` (SBD has no total spin target).

Each determinant must also have `nelec[0]` (alpha) or `nelec[1]` (beta) set bits and a
set must not repeat a determinant. The FCIDUMP has no core energy, so `energy` is the
electronic energy, as with the DICE solver.

## Unrestricted (UHF) integrals

A driver built with `UHF=1` (`native/diag_uhf`, compiled with SBD's `-D_UHF`) takes separate
alpha and beta integrals. Use it through a block with `integrals="unrestricted"`, for example
`create_sbd_blocks(..., sbd_executable=".../native/diag_uhf", integrals="unrestricted")`.
Blocks are saved with `overwrite=True`, so the unrestricted set has its own default names
(`cmd-sbd-uhf-solver`, `exec-sbd-uhf-<target>`, `hpc-<target>-sbd-uhf`, `sbd-uhf-solver`,
executable key `sbd_uhf_solver`), and `create_sbd_blocks` refuses, before saving anything,
unrestricted blocks named like the restricted defaults. Then pass the beta blocks to `run`:

```python
result = await solver.run(
    (alpha_strings, beta_strings), h1_a, h2_aa, norb, (n_alpha, n_beta),
    one_body_tensor_b=h1_b, two_body_tensor_ab=h2_ab, two_body_tensor_bb=h2_bb,
)
```

All tensors are chemist notation in the UHF alpha (a) and beta (b) MO bases; `h2_ab` is
`(aa|bb)`. The alpha and beta determinant files are always written separately. As with
restricted input, the FCIDUMP has no constant term (`energy` is the electronic energy), and
each determinant set must start with its Hartree-Fock determinant.

`-D_UHF` is a compile-time switch, and the two FCIDUMP layouts are not self-describing: a
`-D_UHF` driver given a restricted FCIDUMP would finish with a wrong energy. Three checks keep
the binary and the input together:

- `run` raises `ValueError` before any job when the input kind (beta blocks given or not) does
  not match `SBDSolverJob.integrals`, or when an unrestricted block lacks the permutation
  symmetry the file format assumes.
- The unrestricted FCIDUMP declares `QCSC_INTEGRALS=SBD_UHF_INTERLEAVED` in its header, and
  `native/main.cc` aborts on a file whose layout does not match its build. A file without the
  key is restricted, so existing FCIDUMPs keep working with restricted drivers.
- The driver writes its layout to `sbd_build_info.txt`, and `run` checks it after the job.
  `SBDResult.integrals` holds it. A reported layout that does not match the input raises
  `RuntimeError`, and unrestricted input requires the file.

### Drivers built before the layout check

A driver built before this check writes no `sbd_build_info.txt`. For restricted input, such a
run is still accepted, so existing restricted setups keep running. It logs a Prefect warning,
and `SBDResult.integrals` is `None`. Treat such a result as **not verified**: if a `-D_UHF`
binary built before the check (for example `diag_uhf` from the UHF-SQD fork) is registered as
the restricted executable, it reads the restricted FCIDUMP as a different Hamiltonian, and the
run completes without an error. The energy then cannot be confirmed to be correct. Numerical
validation and published examples use rebuilt drivers that report their layout.

To move a setup to drivers that report their layout:

1. Rebuild the drivers from `native/main.cc` with the site's build script: `diag` without
   `UHF`, and `diag_uhf` with `UHF=1` if unrestricted runs are needed (see
   [native/README.md](native/README.md)).
2. Check where each HPC profile points, and recreate the blocks with `create_sbd_blocks` if a
   path is not the rebuilt binary. The names below are the `create_sbd_blocks` defaults; use
   the names your workflow created (for example `hpc-<system>-sqd-sbd` for the open-loop SQD
   workflow):

   ```python
   from qcsc_prefect_blocks.common.blocks import HPCProfileBlock

   print(HPCProfileBlock.load("hpc-<target>-sbd").executable_map)  # restricted: .../diag
   print(HPCProfileBlock.load("hpc-<target>-sbd-uhf").executable_map)  # unrestricted: .../diag_uhf
   ```

3. Run once and confirm that the warning is gone and `SBDResult.integrals` is `"RESTRICTED"`
   (or `"SBD_UHF_INTERLEAVED"`). The job directory then holds `sbd_build_info.txt` with the
   same value.

The block refers to a `CommandBlock`, an `ExecutionProfileBlock` and an `HPCProfileBlock`
by name. `create_sbd_blocks` creates all four. In the open-loop SQD workflow,
`sqd-create-blocks --solver sbd` (`algorithms/open_loop/sqd/sqd_sbd/create_blocks.py`)
creates them from `sqd_blocks.toml` as `sqd-sbd-solver`, `cmd-sqd-sbd-solver`,
`exec-sqd-sbd-<system>` and `hpc-<system>-sqd-sbd`; see
[its README](../../algorithms/open_loop/sqd/README.md).
`algorithms/closed_loop/sqd_de/create_blocks.py` creates them for the closed-loop
workflow on Miyabi, Fugaku or Slurm.

Blocks saved before this package existed, when `SBDSolverJob` was defined in
`algorithms/closed_loop/sqd_de`, keep working: the block type name (`SBD Solver Job`) and
slug are unchanged, and `carryover_type`, which they lack, defaults to 1.
