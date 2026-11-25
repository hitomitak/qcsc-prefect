# UHF-SQD: sample-based quantum diagonalization on a broken-symmetry UHF reference

`uhf_sqd` is a closed-loop Prefect workflow. It builds an unrestricted, broken-symmetry when that
is lower, Hartree-Fock (BS-UHF) reference from an FCIDUMP. It then optimizes the parameters of a
spin-unbalanced LUCJ ansatz by differential evolution (DE). Each DE walker:

1. samples bitstrings from a quantum source;
2. recovers configurations per spin;
3. subsamples the alpha and beta determinants separately;
4. diagonalizes them with the unrestricted SBD solver.

Use it for open-shell and spin-polarized systems, where the alpha and beta orbitals differ. For
restricted (RHF) references, use [`sqd_de`](../sqd_de/README.md).

The workflow is a port of the UHF-SQD work in
[kunalroy332/qcsc-prefect](https://github.com/kunalroy332/qcsc-prefect), branch
`bk/feature/bs-uhf-sqd-roquo` at `3968e585f7ca8137c127e5ee9f4fb383322d3d0d` (flow
`riken_sqd_de`, by kunalroy332, Boseong Kim and others). This first integration ports the
source's default computation path. The source's optional features are listed, with how to add
them later, in [UHF_SQD_DEFERRED_FEATURES.md](UHF_SQD_DEFERRED_FEATURES.md).

| Part | Where |
|---|---|
| BS-UHF reference, UHF integrals, UCCSD t2, per-spin occupancies | `qcsc_workflow_utility.uhf` |
| Unrestricted SBD solver (`-D_UHF` driver, `integrals="unrestricted"`) | `qcsc-prefect-sbd` |
| Quantum source (`random` in this release) | `QuantumSourceBlock` in `qcsc-prefect-qiskit` |
| LUCJ, DE, recovery loop, K-batch subsampling, telemetry, sample pool | `uhf_sqd` (this directory) |

## Scope of this release

| | Status in this release |
|---|---|
| SBD solver | CPU `diag_uhf`. On ROQUO also `diag-gpu_uhf` with one GPU per solve ([uhf_sqd_blocks.roquo-gpu.example.toml](uhf_sqd_blocks.roquo-gpu.example.toml), checked on 2026-11-25). Several GPUs per solve and other GPU sites are not supported |
| Quantum source | `random` only. Quantum devices (`qrmi`, `qfw` sources) are not supported |
| HPC target | `local`: verified against the source. `slurm`: run end to end on ROQUO with one GPU per solve (2026-11-25). The CPU Slurm procedure is documented but not yet run end to end |
| MPI ranks | One rank verified, on CPUs and on one GPU. Runs with more than one rank are not verified: their energies have not been compared with the one-rank result |

The device code paths are in the package: the compile hook, reset mitigation and the sample
pool. They have not been run on hardware, and the sections that describe them say so.

## Setup

A step-by-step walkthrough, from a local run with a random source to Slurm, is in
[Run the UHF-SQD Closed-loop Workflow](../../../docs/tutorials/run_uhf_sqd_closed_loop_workflow.md).

1. **Install** this directory and its local dependencies, for example
   `uv pip install -e algorithms/closed_loop/uhf-sqd`.
2. **Build the unrestricted SBD driver** with `UHF=1`:

   ```bash
   UHF=1 bash packages/qcsc-prefect-sbd/native/build_sbd_slurm.sh
   ```

   This gives `native/diag_uhf`. Use the site script for Miyabi or Fugaku; see
   [native/README.md](../../../packages/qcsc-prefect-sbd/native/README.md). A driver built before
   the integral-layout check writes no build info: its first solve fails, because it cannot be
   confirmed as a `-D_UHF` build.
3. **Create the blocks** from a TOML or JSON file:
   [uhf_sqd_blocks.example.toml](uhf_sqd_blocks.example.toml), then
   `uhf-sqd-create-blocks uhf_sqd_blocks.toml`.
   - `[solver]` takes the keyword arguments of `qcsc_prefect_sbd.create_sbd_blocks`. `integrals`
     is always `"unrestricted"`, so the blocks get their own default names: `sbd-uhf-solver`,
     `hpc-<target>-sbd-uhf` and so on. They never overwrite the restricted SBD blocks.
   - `[quantum_sources.<name>]` tables are saved as `QuantumSourceBlock` documents. An existing
     Block of the same name, which another workflow such as `sqd_de` may use, is replaced only
     with `overwrite = true` in its table; otherwise creation stops. The example creates
     `uhf-sqd-random`. A random source needs no backend. Device sources are not supported in
     this release; a `qrmi` source would use this package's compile hook
     `uhf_sqd.transpilation:transpile`, so `uhf_sqd` would have to be installed where the QRMI
     compiler runs.

## Running

```python
from uhf_sqd.flow_params import FlowParameters
from uhf_sqd.main import uhf_sqd_de

energy = uhf_sqd_de(parameters=FlowParameters(
    fcidump="data/fcidump_NH2_STO3G.txt",
    quantum_source="uhf-sqd-random",
    sqd_dim=400,
))
```

Or deploy it with `uhf-sqd-deploy`. The flow returns the best total energy, ECORE included. The
telemetry table artifact `uhf-sqd-telemetry` has one row per walker and trial: energy, subspace
dimension and the per-recovery-step trace. The SQD arrays of every pass are stored as `.npz`:

- in the S3 block `s3-sqd` if it exists;
- otherwise under `$SBD_LOCAL_STORAGE_DIR` (default `$PREFECT_HOME/storage`), in
  `sqd_data/<flow name>/<flow run name>/`.

### Input

`fcidump` is a **restricted, spatial-orbital** FCIDUMP with all orbitals active:

- `MS2` sets `n_alpha - n_beta`;
- ECORE is a constant term, added once to the energies.

The UHF reference is built from it. FCIDUMPs in a UHF layout are refused before any integral is
read: `IUHF` other than 0, or `QCSC_INTEGRALS=SBD_UHF_INTERLEAVED`.

### Parameters

The defaults are the source's.

| Parameter | Default | Meaning |
|---|---|---|
| `sqd_dim` | 1 000 000 | subspace dimension (about `sqrt(sqd_dim)` determinants per spin) |
| `n_recovery_steps` | 1 | self-consistent configuration-recovery passes per walker |
| `n_batches` | 1 | independent subspaces per pass (K-batch SQD, arXiv:2405.05068). The lowest energy is kept; the mean occupancies feed the next pass |
| `quantum_source` | `uhf-sqd-random` | name of a `QuantumSourceBlock`; a Block shared with another workflow can be named here |
| `random_seed` | 24 | base of each walker's seed, for random bits and for recovery and subsampling |
| `sampler.shots` | 10 000 | shots per walker |
| `circ_params` | `n_lucj_layers=2`, `ab_stride=4`, `ucj_optimize=True`, reset mitigation, SABRE settings | LUCJ and compilation |
| `de_params` | 4 walkers, 1 iteration, `randomization_factor=0.2`, `fxc=0.6`, `cr_prob=0.9` | differential evolution |
| `reference.af_groups` | None | atom-localized antiferromagnetic guess: fragments of 0-based orbitals with `up`, `down`, optional `free` and `pol` |
| `reference.cache_dir` | `~/.cache/qcsc_uhf` | content-keyed BS-UHF cache; None disables it |
| `reference.cache_read`, `ccsd_max_cycle`, `ccsd_restarts` | True, 50, 10 | cache reuse and UCCSD runs |
| `solver_block_ref` | `sbd_solver_job/sbd-uhf-solver` | must have `integrals="unrestricted"`; checked at the flow start |

Unknown keys are rejected.

The SBD target `spin_sq` is not used. The reference <S^2> is logged at the start, and it is a
field of the reference (`reference_spin_square`).

### Davidson settings

`create_blocks` uses the source's defaults: block 4, iteration 1, tolerance 1e-2,
`carryover_ratio` 0.1, `carryover_type` 0. **They are loose.** The Davidson solve stops before
convergence: in the source baseline, stretched H6 in its full space ends 9.61 mEh above FCI. The
defaults are kept as in the source. Pass tighter settings in `[solver]`
when converged subspace energies are needed, for example `block = 30`, `iteration = 60`,
`tolerance = 1e-10`.

### Sample pool (device sources)

Device sources are not supported in this release, and this path has not been run on hardware.
With the supported `random` source, no pool is written and no reset mitigation is applied.

With a source that runs circuits (`requires_circuit`), each walker saves its sample pool before
any diagonalization. The pool is the shots kept after reset mitigation, one row per shot, so
multiplicities are kept. It is saved as `raw_samples_t<trial>_w<walker>_<uuid>.npz`, and the
telemetry field `raw_samples_path` records where. A failed save is logged and does not stop the
run.

With reset mitigation, the telemetry records:

- `reset_test_passed_shots`: the shots whose `test` register is all zeros;
- `reset_mitigation_fallback`: whether none passed. All shots are then used, as in the source.

`shot_retention_rate` is the source's: the fraction of shots used, so it is 1.0 in both those
cases.

The format is the source's:

| Key | Content |
|---|---|
| `packed_bits` | `BitArray.array`, uint8 `(num_shots, ceil(2 norb / 8))` |
| `num_bits`, `num_shots` | int64 `[1]` |
| `ref_meta` | 1-element Unicode JSON: `<U2048` as in the source, or wider when the JSON is longer |

`ref_meta` has the source's keys `norb`, `nelec`, `fcidump_sha256`, `reference_tag` and
`method`, plus two more:

- `fcidump_sha256` is the sha256 of the FCIDUMP the reference was built from.
- `reference_cache_key` is `v2|<FCIDUMP sha256>|<guess token>|nelec=(n_alpha, n_beta)`. It
  identifies the BS-UHF reference by the FCIDUMP contents, the broken-symmetry guess and the
  electron numbers, and not by the code version. The bitstrings live in that reference's MO
  basis.
- `bit_layout` records the bit order, `qiskit-bitarray-big-endian;meas[i]=q[i];jw:alpha=q[0:norb],beta=q[norb:2norb]`:
  in `bit_array_to_arrays` form, columns `[:norb]` are beta and `[norb:]` alpha.

The source's loader ignores the extra keys and accepts these files. Loading a pool back into the
workflow is deferred.

## Differences from the source

The computation of the ported path is the source's. The only changes are these:

- **Behaviour change: the first valid energy of a walker is accepted whatever its sign.**
  - The source compared each walker's first energy with an initial 0.0, a placeholder from the
    first SBD workflow that assumed negative total energies. So for model Hamiltonians, or an
    FCIDUMP whose constant term makes the total positive, the flow result was 0.0.
  - The first generation is the starting population, with no prior energy to beat. Later trials
    keep the source's rule: only a lower energy replaces a walker's.
  - **Known issue kept from the source** (also in `sqd_de`): a walker that failed with a NaN
    energy in trial 0 keeps an all-NaN parameter row until it succeeds. DE mutation can draw
    that row, which gives NaN mutants. This is to be revisited together with `sqd_de`, by
    re-initialising failed walkers or excluding NaN rows.

- **UHF only.** The RHF mode of the source flow is not ported; use `sqd_de`.
- **Sampling goes through `QuantumSourceBlock`.** A random source draws the same bits as the
  source's `generate_bit_array_uniform` (same seed formula). Reset mitigation on device results
  is the source's, including its fallback to the unmitigated shots when every shot is rejected.
- **Device compilation: a layout search for each circuit**, accepted by the source's author. The
  QRMI hook uses the source's SABRE settings (seed, passes, optimization level).
  - The source ran the search as a Prefect task cached on the flow run only, so the layout of
    the first circuit was normally reused for every walker and trial.
  - For the same circuit, both give the same compiled circuit.
  - The trade-offs:
    - Walkers' LUCJ circuits differ only in angles, and SABRE is seeded, so most searches find
      the same layout and only cost compile time (`sabre_layout_trials`).
    - Where layouts differ, walkers use different physical qubits, and qubit quality can bias the
      DE selection.
  - The source's author used ffsim's error-aware LUCJ mapper in practice (deferred). Measure these
    effects on hardware before relying on device results. Only the random path is verified
    against the source.
- **Options are explicit flow parameters.** The source read some of them from environment
  variables. No environment variable is read except `SBD_LOCAL_STORAGE_DIR` for array storage.
- **The task runner is an explicit `ThreadPoolTaskRunner()`.** The source defaulted to Ray, and
  its baseline below ran on the same thread pool.
- **The pool `ref_meta` has the additive keys** described above, and its width follows its
  length.
- **Reset-mitigation telemetry** adds `reset_test_passed_shots` and `reset_mitigation_fallback`.

## Verification against the source

The port was checked against a baseline of the unmodified source flow (PR0):

- **Systems.** NH2 doublet (STO-3G, 7 orbitals, 5 + 4 electrons) and stretched H6
  (broken-symmetry, <S^2> 2.74), with a random source.
- **Settings.**
  - D0: the source defaults.
  - D1: `sqd_dim=400`, `n_recovery_steps=2`, `n_batches=2`, two DE iterations.
  - T0: tight Davidson settings.
- **Environment.** The baseline's third-party versions, one BLAS/OpenMP thread, one MPI rank.

In all 12 runs (2 systems × 3 settings × 2 repetitions), these outputs are **identical bit for
bit**:

- flow energies;
- telemetry rows;
- every solver input and output: determinant files, carryover, occupancies, and the FCIDUMP
  integral records;
- stored SQD arrays;
- the reference integrals, t2 blocks and occupancies.

T0 matches FCI within 1e-13 Eh.
