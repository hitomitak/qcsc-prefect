# UHF-SQD: features deferred from the first integration

**Status: not yet ported. None of these features is removed or rejected.** The first integration
of UHF-SQD into `qiskit-community/qcsc-prefect` deliberately has a narrow scope. Every feature
below, except section 1, exists in the source branch and can be added in a later pull request.
Section 1 (saving the sample pool) is **part of the first integration**.
It stays in this file next to the deferred read side, so that both halves of the pool are
described together. This file records where each feature lives, how it is switched on, what it
depends on, and how to connect and validate it when it is ported.

The first integration is the `uhf_sqd` package in this directory; see [README.md](README.md).

## Source and scope

| Item | Value |
|---|---|
| Source repository | https://github.com/kunalroy332/qcsc-prefect |
| Source branch / commit | `bk/feature/bs-uhf-sqd-roquo` @ `3968e585f7ca8137c127e5ee9f4fb383322d3d0d` |
| Source baseline (fork main) | `149c564f44da7a3d00328bec25934d57698049b5` |
| Integration target at survey time | `qiskit-community/qcsc-prefect` @ `dc481a1d9ae5d8e516c747f77366d808bd1b853d` |
| SBD library used for validation | `r-ccs-cms/sbd` @ `95bf45a5be9ffd977fac7b0aaa359833caf3401a` |

All source paths below are at `3968e58` unless stated otherwise. `sqd.py`, `main.py`,
`flow_params.py`, `solver_job.py`, `data_io.py` and `transpile_custom.py` are under
`algorithms/sbd/sbd/`. `chem.py`, `active_space.py` and `orbital_opt.py` are under
`algorithms/qcsc_workflow_utility/src/qcsc_workflow_utility/`.

The first integration ports the default computation path of the source flow `riken_sqd_de`:

- generic FCIDUMP input with explicit broken-symmetry (BS) settings;
- the UHF/BS-UHF reference, spin-unbalanced LUCJ and differential evolution (DE);
- per-spin recovery and subsampling, including the recovery loop `n_recovery_steps` (default 1)
  and the **K-batch recovery** `n_batches` (default 1, arXiv:2405.05068). Both are in the first
  integration, and the PR0 configuration D1 runs them with value 2. Not to be confused with the
  deferred `SamplerParameters.n_shot_batches` (section 10);
- the CPU SBD solver through `qcsc-prefect-sbd`;
- sampling through the shared `QuantumSourceBlock`, with `random` and QRMI sources;
- **saving the sample pool** after device sampling (section 1).

The scientific content of that path is preserved, and the PR0 baseline checks that it is.

The first release supports only the `random` source on the CPU SBD driver. The QRMI path and the
pool save are ported but have not been run on hardware; see "Scope of this release" in
[README.md](README.md).

**Known limitation kept from the source.** The source's default solver settings (`create_blocks.py`:
block 4, iteration 1, tolerance 1e-2) do not converge to the exact energy of the subspace. On the
PR0 H6 full space the result is 9.61 mEh above FCI. The defaults are kept as they are; see
`PR0_BASELINE.md`, "Known limitation of the original solver defaults".

The source flow default (all optional features off) is listed per feature below. **"Off by
default" does not mean "unused".** The authors' ROQUO configuration (`algorithms/sbd/roquo/run_bs.toml`
with `run_task.py`) turns several of these features on.

---

## 1. Sample pool: saving (the write side) — IN THE FIRST INTEGRATION

- **Purpose.** After device sampling, persist the merged sample pool before any diagonalization,
  so that an expensive hardware sample can be re-diagonalized later as the same distribution.
  The read side (section 2) is deferred.
- **In the source (ported as it is).**
  - `sqd.py:589-607` in `walker_sqd`. It runs by default on the `real-device` path and only
    there; random sampling writes no pool. It is best-effort: a failed save is logged and the run
    continues.
  - Written by `save_ndarray(file_prefix=f"raw_samples_t{trial:02d}_w{walker}", packed_bits,
    num_bits, num_shots, ref_meta)`.
  - The reference record is built by `_pool_ref_meta` / `_pool_ref_meta_array` (`sqd.py:296-312`).
  - Storage helpers are in `data_io.py:32-88` (`_flow_scoped_subdir`, `_local_storage_dir`,
    `save_ndarray`).
- **Commits.**
  - `b9023b1` (kunalroy332): persist the merged pool.
  - `d3d97b0` (Boseong Kim): local path rules and `SBD_LOCAL_STORAGE_DIR`.
  - `3968e58` (Boseong Kim): the `ref_meta` record.
- **Storage format (kept as in the source).** A compressed `npz` (`allow_pickle=False`) with these
  keys:
  - `packed_bits`: `BitArray.array`, uint8, shape `(num_shots, ceil(2*norb/8))`. One row per kept
    shot, in sampler order, so the **multiplicities (frequencies) are preserved**.
  - `num_bits` (int64[1]) and `num_shots` (int64[1]).
  - `ref_meta`: a 1-element `<U2048` JSON string with the keys `norb`, `nelec`,
    `fcidump_sha256`, `reference_tag` and `method`.
- **Bit layout** (documented; it is not stored in the file by the source). Checked with qiskit
  2.6.2:
  - `packed_bits` is qiskit's big-endian packing: clbit 0 is the least significant bit of the last
    byte.
  - The circuit measures qubit i into `meas` clbit i (`lucj.py` `create_lucj_circuit`).
  - ffsim's Jordan-Wigner layout puts the alpha orbitals on qubits `0..norb-1` and the beta
    orbitals on `norb..2norb-1`.
  - So in `bit_array_to_arrays` matrix form, columns `[:norb]` are beta and `[norb:]` are alpha,
    each with orbital `norb-1-c` in column c. This matches `sqd.py:720-725`.
- **What the pool contains.** The shots kept after reset mitigation (the distribution that feeds
  recovery), not the raw device shots. The `test` register is not stored. The retention rate is in
  telemetry (`shot_retention_rate`). This is the source behaviour and is kept.
- **Correspondence with the reference.**
  - On the FCIDUMP path the source writes `fcidump_sha256: null` and `reference_tag: null`
    (`sqd.py:284-312`). The record then identifies only `norb`, `nelec` and the method.
  - The bitstrings live in the BS-UHF MO basis. That basis is determined by the FCIDUMP contents,
    the BS guess settings, `nelec` and the code version, so this record is weak.
  - **In `uhf_sqd`, the record is extended additively.** The source loader compares only its
    five keys and ignores extra ones, so it still reads the files (checked with the unchanged
    `_check_saved_pool` of `3968e58`).
    1. The existing `fcidump_sha256` key holds the sha256 of the FCIDUMP the reference was built
       from.
    2. `bit_layout` records the bit order:
       `"qiskit-bitarray-big-endian;meas[i]=q[i];jw:alpha=q[0:norb],beta=q[norb:2norb]"`.
    3. `reference_cache_key` is the BS-UHF cache key `v2|<fcidump sha256>|<guess token>|nelec`.
       It identifies the reference (the FCIDUMP contents, the BS guess settings and `nelec`) and
       not the code version.

    The format (`.npz` keys, dtypes, the `<U2048` JSON) is otherwise the source's. Pool loading
    and checkpointing stay deferred.
- **Where it is stored.** The S3 block `s3-sqd` if present, otherwise
  `$SBD_LOCAL_STORAGE_DIR` or `$PREFECT_HOME/storage`. The subdirectory is
  `sqd_data/<flow name>/<flow run name>/`. The function returns `file://...` or an S3 key, and
  the telemetry field `raw_samples_path` records it.
- **Where it goes in the port.** In the `uhf_sqd` walker, right after `source.sample(...)` and
  reset mitigation, before recovery, and only for sources with `requires_circuit=True`. That is the
  device case; the source condition is `quantum_source == "real-device"`. The `data_io` helpers
  are ported unchanged.
- **Validation for the first integration.**
  - On a mocked device result: saving and reloading (`np.load` plus `BitArray(packed, num_bits)`)
    reproduces the `BitArray` bit for bit.
  - Recovery on the reloaded array gives the same recovered bitstrings.
  - The written file passes the source's `_check_saved_pool` loader unchanged.
  - Telemetry has `raw_samples_path`.
  - A failed save does not stop the run.

## 2. Sample pool: loading (the read side)

- **Purpose.** Re-diagonalize from a saved pool (`quantum_source="saved"`) without any device
  call. Used for sweeps over `sqd_dim` or recovery settings on fixed samples, and for DE or
  orbital-optimization runs that share one pool across walkers.
- **In the source.**
  - `sqd.py:608-635` reads the pool: one pool per walker if given, otherwise every walker uses
    `saved_samples[0]`.
  - `_check_saved_pool` (`sqd.py:314-359`) guards it:
    - `num_bits != 2*norb` always raises;
    - a mismatch in `ref_meta` raises unless `SBD_ALLOW_POOL_MISMATCH=1`;
    - a pool without `ref_meta` is accepted with a warning;
    - when either side of `fcidump_sha256` / `reference_tag` is None, they are not compared.
  - Readers are `data_io.py:90-145` (`_local_path`, `load_ndarray`, `load_ndarray_optional`).
- **Commits.** `b9023b1`, `e7e1a53` (reuse one saved pool across DE walkers), `3968e58` (the
  guard).
- **How it is enabled.**
  - `FlowParameters.quantum_source="saved"` (`flow_params.py:400`) and
    `FlowParameters.sampler.saved_samples=[...]` (`flow_params.py:246-256`).
  - The ROQUO launcher switches a `real-device` task to `saved` automatically once a pool exists
    (`roquo/run_task.py:357-362`).
- **Where it is used.**
  - `examples/sbd_uhf_recover_any_molecule/run_recover.py` and its README;
  - `docs/tutorials/run_bs_uhf_sqd_advanced_roquo.md`;
  - `docs/archive/run_uhf_bsuhf_any_molecule.md`;
  - `algorithms/sbd/README.md`;
  - `roquo/run_task.py` and `roquo/task_state.py`.
- **Tests.** `algorithms/sbd/tests/test_saved_pool_guard.py` and `test_data_io_local.py`.
- **Not in the first integration.**
  - `quantum_source` names a `QuantumSourceBlock`; there is no `saved` kind.
  - `saved_samples` is not a flow parameter, and `SBD_ALLOW_POOL_MISMATCH` is not read.
- **Where to connect it later.** Either of:
  - a source kind in `qcsc-prefect-qiskit` (shared; this belongs with the sampling-mode work in
    #76/#77);
  - a `uhf_sqd`-local branch at the same place the walker obtains its `BitArray`.

  Keep the reference fields (`norb`, `nelec`, `fcidump_sha256`, `reference_tag`, method) in the
  first-release reference model and telemetry so that the guard can be ported unchanged.
- **Caveats.**
  - The key has no record of orbital rotations or `cache_token`; a pool sampled in a rotated
    basis passes the guard.
  - Legacy pools are only checked for their bit count.
- **Validation needed.**
  - Port `test_saved_pool_guard.py`.
  - Check that every rejection happens before any solver job.
  - Compare energies with the run that produced the pool.

## 3. Checkpoint and resume of the recovery loop

- **Purpose.** Persist per-step recovery state so that a long multi-step recovery can resume
  after a wall-time limit or node failure.
- **In the source.**
  - Load and resume: `sqd.py:668-700`.
  - Atomic save at the end of each recovery step: `sqd.py:1168-1199`.
  - File `$FE4S4_CKPT_DIR/ckpt_t{trial:02d}_w{walker}.npz` with `occ_a`, `occ_b`, `next_step`,
    `best_energy`, `carryover`, and `carryover_b` (UHF).
  - The launcher-side identity guard (`source.json`: samples, `fcidump_sha256`, `spin_tag`,
    `sqd_dim`, `n_batches`) is in `roquo/run_task.py:298-349`; `roquo/task_state.py` holds the
    resume rule.
- **Commits.** `4a60460` / `dbf25c9` (kunalroy332), `34d89f2` (missing import fix), `73aa034`
  (Boseong Kim, per-spin carryover).
- **How it is enabled.** The environment variable `FE4S4_CKPT_DIR`. The ROQUO launcher sets it
  (`roquo/run_task.py:301`), and `run_recover.py --ckpt-dir` sets it too.
- **Where it is used.** `examples/sbd_uhf_recover_any_molecule`,
  `docs/tutorials/run_bs_uhf_sqd_roquo.md` and `docs/tutorials/run_bs_uhf_sqd_advanced_roquo.md`.
- **Not in the first integration.** The environment variable is not read and no checkpoint is
  written.
- **Where to connect it later.** The same two points in the ported recovery loop: before the loop
  for resume, and after each step's occupancy update for save. Keep the loop's per-step state
  (`avg_occ`, `best_energy`, `best_carryover(_b)`) as explicit variables in the port, as the source
  has them.
- **Caveats.**
  - In the flow, the key is only `(trial, walker)`; there is no Hamiltonian or sample identity.
    That guard exists only in the ROQUO launcher.
  - The walker RNG state is not saved, so a resumed run is not bit-identical to an uninterrupted
    one.
  - Upstream's open-loop SQD already has a content-keyed checkpoint
    (`algorithms/open_loop/sqd/sqd_common/checkpoint.py` @ `dc481a1`). It should gain a reference
    tag for UHF.
  - Decide whether to keep the source behaviour or the content-keyed design. That is a behaviour
    change that needs approval.
- **Validation needed.**
  - A resume gives the same final energy as an uninterrupted run, or the documented difference if
    the RNG state stays unsaved.
  - A checkpoint from another Hamiltonian or set of samples is rejected.

## 4. Active-space manifest input (BS-UNO-AVAS)

- **Purpose.** Build the BS-UHF reference from a manifest (schema v1) produced by `bs_uno_avas`,
  instead of from an FCIDUMP plus a broken-symmetry guess. The manifest pins the FCIDUMP, the
  orbital labels, the alpha/beta 1-RDM and the spin pattern by sha256.
- **In the source.**
  - `active_space.py`: the whole file. It provides `ActiveSpaceSpec`, `validate_cluster_labels`,
    `density_from_af_groups` and the sha256 checks.
  - In `chem.py`:
    - `_converge_reference_uhf` (`chem.py:470-570`) runs the RDM start and the label start,
      checks the tag and energy, and supports `SBD_BS_STRICT` and `SBD_BS_E_TOL`;
    - `manifest_reference_summary` (`chem.py:592-609`);
    - the content-keyed cache with `cache_token` (`chem.py:616-715`);
    - the entry point `compute_molecular_integrals_from_fcidump(..., active_space_manifest)`
      (`chem.py:1029-1072`);
    - the provenance fields `reference_tag`, `reference_source` and `fcidump_sha256`
      (`chem.py:84-94`).
  - `flow_params.py:288` and `flow_params.py:559-566` (`fcidump` or a manifest), and
    `main.py:214-224`.
- **Commits.** `3968e58` and `5ab2e86` (Boseong Kim).
- **How it is enabled.** `FlowParameters.active_space_manifest`; optionally
  `SBD_SKIP_MANIFEST_HASH`, `SBD_BS_STRICT` and `SBD_BS_E_TOL`.
- **Where it is used.**
  - `default_params/bs_fe2s2_manifest.example.json` and `roquo/run_bs.toml`;
  - `docs/tutorials/run_bs_uhf_sqd_roquo.md` and `run_bs_uhf_sqd_advanced_roquo.md`;
  - `docs/BS_UHF_MANIFEST_CHANGES_REPORT.md`.
- **Tests.**
  - `qcsc_workflow_utility/tests/test_active_space.py`.
  - `algorithms/sbd/tests/test_bs_manifest_reference.py`. Its Fe-S tests need untracked fixtures
    under `plugins/bs_uno_avas/workspace/output/` and are skipped in a clean clone.
- **Dependencies and formats.**
  - PySCF.
  - `files.{fcidump,labels,rdm}`, each with `name` and `sha256`. The labels are a 1-D unicode
    `.npy` with entries `<idx>_<El>_<AO>`; the RDM has shape `(2, norb, norb)`.
  - The FCIDUMP is restricted and spatial; its ECORE includes the frozen-core energy.
- **Limits.** Fe2S2/Fe4S4 only, with Fe 3d and bridging S 3p orbitals (`active_space.py:160-186`).
- **Not in the first integration.** Only FCIDUMP plus explicit BS settings is accepted.
- **Where to connect it later.**
  - The reference module `qcsc_workflow_utility.uhf`. Keep the cache key format (`v2|fcidump sha256|guess
    token|nelec`) so that a manifest `cache_token` fits as the guess token.
  - Keep the provenance fields in the reference model and telemetry. They are None on the FCIDUMP
    path, as in the source.
- **Validation needed.**
  - Generate the fixture in CI with the plugin's `bare_Fe2S2` demo. Measured during the survey:
    about 22 s on CPU, about 820 KB of output, PySCF 2.15.
  - Check the rebuilt reference against the manifest: measured E difference 5.6e-10 Eh with a
    matching tag.
  - Check full-space or subspace energies against an exact reference.

## 5. `plugins/bs_uno_avas` (manifest generator)

- **Purpose.** Turn a protonated PDB excerpt (full-system high-spin UHF, then BS-UHF per spin
  pattern) or a Gaussian BS-DFT `.fchk` into a UNO-AVAS active space plus a schema-v1 manifest.
- **In the source.**
  - `plugins/bs_uno_avas/`: `cli.py`, `pipeline.py`, `manifest.py`, `bs_guess.py`,
    `uno_avas.py`, `fchk*.py`, `verify.py`, and `configs/`, `examples/`, `tests/`;
  - ROQUO scripts in `plugins/bs_uno_avas/roquo/`;
  - `docs/tutorials/run_bs_uno_avas_roquo.md`.
- **Commit.** `5ab2e86` (Boseong Kim).
- **Dependencies.** numpy, scipy, PySCF; optionally `gpu4pyscf`. It does not import qcsc-prefect.
- **Data.** Small `.npy` AO references and `.chk` files under `tools/AOs` (self-generated).
  `examples/5XTD_FES_O301_excerpt.pdb` has no attribution REMARK; add one before release.
- **Not in the first integration.** The plugin is not added.
- **When adding.**
  - Publish one JSON Schema for manifest v1 shared by writer and reader; today the schema is
    defined twice.
  - Keep the plugin as an optional package.

## 6. GPU SBD solver

- **Purpose.** UHF solves on GPUs: a single-GPU `diag-gpu_uhf` and a multi-GPU MPI/NCCL
  `diag-gpu_uhf-mpi-*`.
- **In the source.**
  - `algorithms/sbd/native/build_sbd_gpu.sh`: `UHF=1` adds `-D_UHF`; `MPI=1` adds NCCL,
    `-DSBD_NON_CUDA_AWARE_MPI` and rank distribution. It needs NVHPC.
  - `native/main.cc:15-36`: per-rank GPU selection before `MPI_Init`.
  - `native/smoke_gpu.sh` and `native/smoke_inputs.py`, and `native/README.md`.
  - `solver_job.py` `solver_mode="gpu"` adds `--adetfile` and `--carryoverfile`.
- **Commits.** `c4a4352` (UHF=1), `af98520`, `548f880`, `d3d97b0` (MPI=1, device selection,
  smoke tests).
- **Already upstream.** The `main.cc` GPU device selection (`90b0f90` in `dc481a1`).
- **SBD prerequisites.** PR79 (`24befee`), PR84 (`47e74d0`) and `nccl_utility.h` are all
  contained in the pinned `95bf45a`.
- **How it is enabled.** The solver block `solver_mode="gpu"`, a GPU executable, and the ROQUO
  `run_bs.toml` `[solver]` section.
- **Where it is used.** `docs/howto/howto_setup_qcsc_prefect_roquo.md`, the ROQUO tutorials, and
  the archived ROQUO GPU tutorial.
- **Not in the first integration.** CPU only.
- **Connected so far.** `packages/qcsc-prefect-sbd/native/build_sbd_gpu.sh` builds the 1-GPU
  `diag-gpu_uhf` with `UHF=1`, at the pinned `SBD_REVISION`, and points an Open MPI wrapper at
  `nvc++`. It is not validated yet. `MPI=1` (multi-GPU) is still deferred.
- **Where to connect the rest later.**
  - `MPI=1` in the same script.
  - The integral-layout marker and the build-info check of `qcsc-prefect-sbd` apply unchanged.
- **Caveats.**
  - `smoke_inputs.py` uses equal alpha and beta integrals. It cannot catch alpha/beta block
    mix-ups, and it never runs the RHF binary.
  - The source build scripts do not pin the SBD revision.
- **Validation needed.**
  - Full-space FCI comparison with alpha and beta integrals that differ, plus BS-UHF, on 1 and N
    GPUs.
  - The layout-mismatch tests on GPU builds.

## 7. ROQUO launcher stack

- **Purpose.** Run many manifest tasks inside one Slurm allocation on ROQUO: node packing,
  reservation handling, pre-flight and reachability checks, resume bookkeeping, and reports.
- **In the source.**
  - `algorithms/sbd/roquo/`: `run_bs.sh` (`check|job|sweep|report`), `run_bs.toml`,
    `run_task.py`, `task_state.py`, `bs_common.py`, `preflight.py`, `check_reach.py`,
    `README.md`.
  - `secrets/` (`README.md`, `secrets.env.example`).
  - `docs/howto/howto_setup_qcsc_prefect_roquo.md` and the ROQUO tutorials.
- **Commit.** `3968e58`.
- **How it works.** It runs SBD in-allocation through the executor's `local` target, with
  `SBD_LAUNCHER=srun` and `SBD_MPI_OPTIONS` (`run_task.py:213-244`). It requires outbound access
  to Prefect Cloud and IBM from compute nodes. `MODE=cloud` uses the private Prefect API
  `runner._scheduled_run_poller` (`run_task.py:475`).
- **Not in the first integration.** None of this is ported.
- **Where to connect it later.**
  - #81: ROQUO as `HPCProfileBlock`/`ExecutionProfileBlock` with `hpc_target="slurm"` and
    `slurm_extra_directives`.
  - #74: session, reattachment and concurrency.
- **Caveats.**
  - The upstream `local` target rejects `modules`, `pre_commands` and `gpus_per_node`
    (`qcsc_prefect_adapters/local/runtime.py:30-52` @ `dc481a1`).
  - The howto clones `qiskit-community/qcsc-prefect` and then checks out a branch that exists only
    on the fork (`howto:74-76`).
  - Per-task block names derive from `run_name` in a shared Cloud workspace.
- **Validation needed.** The #81 checks: submission, monitoring, exit status, logs and
  cancellation on ROQUO.

## 8. Sweeps

- **Purpose.** Run one task over several `sqd_dim` values, reusing the same pool.
- **In the source.**
  - `roquo/run_bs.sh sweep` with a list-valued `[flow].sqd_dim` in `run_bs.toml`.
  - The experiment scripts in `algorithms/sbd/sweep/*`. These contain internal paths and accounts;
    do not port them as they are.
- **Depends on.** Pool loading (section 2), and usually checkpoints (section 3).
- **Not in the first integration.** No sweeps.
- **When adding.** Express a sweep as several flow runs, through deployments or executor bulk
  features, on top of the pool read side. Do not add a separate launcher.

## 9. Dynamical decoupling and twirling

- **Purpose.** IBM SamplerV2 error mitigation: DD (`XX`/`XpXm`/`XY4`), measurement twirling and
  gate twirling.
- **In the source.**
  - `SamplerParameters` (`flow_params.py:194-270`; `runtime_options` at 257-269).
  - `prefect_qiskit_patch.apply_twirling_patch` (`prefect_qiskit_patch.py:33`), applied at import
    (`sqd.py:43`).
- **Commits.** `6ff4edf`, `59e53a7`, `b41fae1` (kunalroy332), `d3d97b0` (moved to flow
  parameters).
- **Defaults.** All False. Gate twirling is documented as rejected by IBM Runtime on the
  fractional `rzz` gates of LUCJ (error 1519).
- **Where it is used.** `roquo/run_bs.toml` `[flow.sampler]`, `sbd_blocks.example.toml`, the
  ROQUO tutorials and `walkthrough.ipynb`.
- **Not in the first integration.** Upstream `QrmiRuntime.sampler` uses `options` only to resolve
  shots (`qcsc_prefect/integrations/qiskit/qrmi/runtime.py:149-153` @ `dc481a1`), so there is no
  path for these options.
- **Where to connect it later.** A shared change to the QRMI sampling options (qcsc-prefect-qiskit).
  Then `SamplerParameters` fields can be re-added; `extra="forbid"` keeps that backward
  compatible.
- **Validation needed.** The options reach the device (inspect the submitted job), and the results
  with options off are unchanged.

## 10. Multi-submission shot batching (`n_shot_batches > 1`)

- **Purpose.** Split the shots into K device submissions and merge them, applying reset
  mitigation per batch.
- **In the source.** The sampler loop at `sqd.py:510-590`, controlled by
  `SamplerParameters.n_shot_batches` (default 1).
- **Commits.** `b8b6f01`, `aac6a3f` (empty-batch guard).
- **Not in the first integration.** One submission per walker (`n_shot_batches=1`, the source
  default). This is different from the recovery `n_batches`, which is ported.
- **Where to connect it later.** A loop over `source.sample(...)` in the `uhf_sqd` walker,
  concatenating `BitArray`s exactly as `sqd.py` does.
- **Validation needed.** The concatenation and per-batch mitigation match the source on mocked
  results.

## 11. Error-aware LUCJ layout

- **Purpose.** Use `ffsim.qiskit.generate_lucj_pass_manager`: a LUCJ-aware, noise-aware layout
  with error thresholds and VF2PostLayout.
- **In the source.**
  - `CircuitParameters.use_error_aware_layout`, `two_qubit_error_threshold`,
    `readout_error_threshold` and `layout_connectivity` (`flow_params.py:86-138`).
  - `transpile_custom.transpile_lucj_error_aware` (`transpile_custom.py:184`).
  - The call at `sqd.py:446-476`.
- **Commits.** `e2a2285`, `dc0633d` (kunalroy332), `68ca6b8` (launcher defaults).
- **Default.** False.
- **Where it is used.** `roquo/run_bs.toml` and `run_bs_uhf_sqd_advanced_roquo.md`.
- **Not in the first integration.** Only the source's default SABRE compile policy is ported, with
  a layout search for each circuit (the source reused one layout per flow run). The source's
  author used this error-aware mapper in practice, so it is the layout to compare against when
  device runs are evaluated.
- **Where to connect it later.** The `uhf_sqd` QRMI transpiler hook. It needs backend error data
  from the Target at compile time.
- **Validation needed.** The transpiled circuits match the source on a fake backend.

## 12. Classical CISD seeding (`seed_cisd`, `seed_budget_frac`)

- **Purpose.** QSCI+SD: force the HF singles and/or doubles into every subspace, capped by a
  budget fraction. This changes the method (the subspace is no longer purely sampled).
- **In the source.**
  - `flow_params.py:336-376`.
  - `_cisd_strings` (`sqd.py:1402`) and `_merge_with_seed` (`sqd.py:1562`).
  - Used inside `subsample_open_shell` / `_subsample_one_spin` (`sqd.py:1705-1842`).
- **Commits.** `31ef152`, `7284105`.
- **Defaults.** `seed_cisd=0`, `seed_budget_frac=1.0`. `FE4S4_SEED_BUDGET_FRAC` overrides the
  fraction (`sqd.py:405-407`).
- **Where it is used.** `roquo/run_bs.toml` and `run_bs_uhf_sqd_advanced_roquo.md`.
- **Not in the first integration.** Not exposed as a flow parameter.
- **How the port keeps it.** It ports the subsampling functions verbatim, with these keyword
  arguments at their defaults, so that the default path stays bit-identical and the feature can
  be exposed later without rewriting the function.
- **Validation needed.** Seeded subspaces match the source for levels 1-3 and budget fractions
  below 1.

## 13. HCI importance booster (`hci_boost`)

- **Purpose.** After each solve, add the single and double children of the high-weight carryover
  determinants, re-solve once and keep the lower energy (SHCI/CIPSI style).
- **In the source.**
  - `flow_params.py:378-404` (`hci_boost`, `hci_boost_max`, `hci_boost_ncore`).
  - `sqd.py:937-1004`, `_hci_boost_strings` (`sqd.py:1367`), `_sd_children` (`sqd.py:1323`),
    `_excitation_amplitude` (`sqd.py:1346`).
  - The environment variable `HCI_BOOST_OVERSHOOT` (`sqd.py:955, 986`).
- **Commits.** `71d8820`, `8ff835e`, `68ca6b8`. Commit `71d8820` has a cluster hostname as its
  author e-mail; ask the author for the attribution address before porting it.
- **Default.** 0 (off). It needs solver carryover (`carryover_type >= 1`).
- **Not in the first integration.** Not exposed.
- **Validation needed.** Boosted energies match the source, and the extra solve count and
  subspace growth are as documented.

## 14. Recombination boost (`FE4S4_RECOMB_BOOST`)

- **Purpose.** Force recombined triple and quadruple candidates (pairs of disjoint sampled
  single/double deltas) into the subspace budget.
- **In the source.**
  - Environment variables at `sqd.py:403-406`: `FE4S4_RECOMB_BOOST` and
    `FE4S4_RECOMB_MAX_PARENTS`.
  - `_recombine_same_spin_candidates` (`sqd.py:1475`) and `_augment_seed_with_recombination`
    (`sqd.py:1533`).
- **Commit.** `71d8820` (see the attribution note in section 13).
- **Default.** Off. It is active even with `seed_cisd=0` once the variable is set.
- **Not in the first integration.** The environment variable is not read.
- **When adding.** Make it an explicit parameter, not an environment variable.

## 15. Orbital optimization and spin-resolved RDMs

- **Purpose.** Orbital optimization (OO) between DE trials from SBD RDMs, self-consistent re-solves,
  CC refresh of the LUCJ seed after OO, and an optional initial OO (`OO_INIT`).
- **In the source.**
  - `orbital_opt.py`: `optimize_orbitals`, `rotate_electronic_properties`,
    `resolve_orbitals_self_consistent`.
  - `main.py:194-543` (gated by `do_orbital_opt = solver.do_rdm != 0`, `main.py:198`) and
    `_apply_cc_refresh_and_reseed` (`main.py:32`).
  - `chem.py` `refresh_cc_seed` (`chem.py:1157` onward) and `_apply_initial_oo`
    (`chem.py:885`; `OO_INIT` at `chem.py:877-880`).
  - `solver_job.py` `_read_rdm_file` (`solver_job.py:366`) and the RDM fields.
  - The `--rdm` dump in the fork's `native/main.cc` (`ba57395`). Upstream `main.cc` does not dump
    RDMs.
- **Commits.**
  - Yuto Terashima: `6b39aa6`, `aa327b5`, `a046efd`, `9eab969`, `4052a7e`, `ba536b5`, `1971a38`,
    `f9eb443`, `5c62407`, `7baed90`, `1047b36`, `757a1c3`.
  - kunalroy332: `fd7f409`, `ba57395`, `c6abf4a`, `45806bd`, `d3d86eb`, `2ae46e6`, `04610d3`,
    `f0791d4`.
  - `61f0fd3` (the `do_rdm` option).
- **How it is enabled.**
  - The solver block `do_rdm != 0` and `FlowParameters.oo_*` (`flow_params.py:440-555`).
  - The environment variables `OO_INIT`, `OO_GRAD_TOL`, `OO_TRUST`, `OO_MAXITER`, `OO_CHECK`,
    `OO_SKIP_NONBEST`, `OO_TO_LUCJ`, `OO_SAT_STAGNATION`, `OO_DUMP_RDM` and `OO_DUMP_DIR`.
- **Where it is used.** `roquo/run_bs.toml` (`do_rdm`) and `run_bs_uhf_sqd_advanced_roquo.md`.
- **Dependencies.** jax (optional `use_jax`).
- **Not in the first integration.** `do_rdm` must stay 0, and the `oo_*` fields are not exposed.
- **Caveats.**
  - The default self-consistent backend `solve_fermion` uses alpha-only integrals for UHF
    (`orbital_opt.py:810-813`).
  - The SBD opposite-spin 2-RDM sign fix (`6e4a3db`) is in the pinned `95bf45a`, but binaries
    built from older revisions do not have it.
- **Where to connect it later.**
  - RDM output in `qcsc-prefect-sbd`: `main.cc` and `SBDResult` already have `rdm1`/`rdm2`
    fields upstream.
  - The DE loop in `uhf_sqd/main.py`, at the same between-trial point as the source.
- **Validation needed.**
  - RDM energy reconstruction equals the solver energy.
  - Alpha/beta and opposite-spin blocks are checked against PySCF.
  - The UHF self-consistent backend is checked before use.

## 16. Subspace <S^2> helper

- **What it is.** `spin_square_from_subspace` (`solver_job.py:68-95`), added in `89ea2a5`. It is
  called only by tests.
- **Caveat.** It uses alpha-only integrals and assumes an identity alpha/beta overlap, so it is
  not valid for UHF orbitals.
- **Not in the first integration.** Not ported. The first release records the reference
  <S^2> (`mf.spin_square()`) as a diagnostic instead.
- **When adding.** Implement it with the alpha/beta MO overlap and validate it against PySCF.

## 17. Examples and data that depend on deferred features

- `examples/sbd_uhf_recover_any_molecule/` (`run_recover.py`, README, `walkthrough.ipynb`) needs
  sections 2 and 3, and uses the AF-group settings (ported as explicit parameters).
- `examples/fe4s4_hci_from_bsuhf_reference/` uses the heat-bath CI app `sbd::gdb` (`544fa29` and
  others). It is not SQD, it relies on non-public pool data, and its Fugaku job scripts include a
  project group.
- `examples/roquo_gpu_prefect_hello_demo/` belongs with section 7.
- The large data files `algorithms/sbd/data/fcidump_Fe4S4_MO.txt` (replaced in `90b67f4`) and
  `fe2s2_40q.fcidump` are excluded by #79. Note that the upstream file with the same Fe4S4 name
  has different content.

## 18. Other differences that are not ported

- **The RHF mode of the source flow** (`solver.method="rhf"`). `uhf-sqd` is UHF only; RHF users
  keep `sqd_de`. The source's RHF changes are not ported to `sqd_de` (for example `0aa5d4a`,
  which changes the RHF occupancy source).
- **The end-of-walker summary plot** (`_comprehensive_summary`, `sqd.py:100-240`, `057ec59`). It
  needs matplotlib, which is optional; the source skips the plot when matplotlib is missing.
- **Ray as the default task runner** (`main.py:70-89`). `uhf-sqd` keeps the runner selection, but
  Ray is optional. The PR0 baseline used `SBD_TASK_RUNNER=concurrent`.

---

## Known issue kept from the source

- **NaN parameter rows in DE mutation.** A walker that failed with a NaN energy in trial 0 keeps
  an all-NaN population row. `mutation_and_crossover` can draw it as r1..r4, or keep it through
  crossover, which gives NaN mutants. The same code is in `sqd_de` (both come from `251f68e`).
  Fixing it, by re-initialising failed walkers or excluding NaN rows, changes the DE draws. It is
  left for a change shared with `sqd_de`, once a re-initialisation technique is chosen.

## Connection points the first integration keeps

These are not abstractions or shared APIs; they are existing structures that the port keeps
unchanged so that the deferred features can be added later.

1. **The walker's single `BitArray`.** The point where the walker holds one merged `BitArray` from
   the source, after mitigation and before recovery. Pool saving (first integration) runs here;
   pool loading and shot batching attach here later.
2. **The recovery loop's explicit per-step state** (`avg_occ`, `best_energy`,
   `best_carryover(_b)`, `walker_rng`) and its two hook points. Checkpoints attach here.
3. **Verbatim subsampling.** The subsampling functions keep their keyword arguments (`seed_cisd`,
   `seed_budget_frac`, `recomb_boost`, `recomb_max_parents`) at the source defaults.
   Seeding and recombination attach here.
4. **Reference provenance fields** (`reference_tag`, `reference_source`, `fcidump_sha256`, `norb`,
   `nelec`, method) in the reference model and telemetry, None where the source has None. The
   pool guard and the manifest path need them.
5. **The cache key format** `v2|<fcidump sha256>|<guess token>|nelec`. A manifest `cache_token`
   fits as the guess token.
6. **`SBDSolverJob.solver_mode`** keeps `"gpu"`, and the integral-layout marker is independent of
   the build backend. GPU binaries attach through blocks.
7. **`FlowParameters` with `extra="forbid"`.** Re-adding fields (sampler options, `oo_*`,
   `saved_samples`) is backward compatible.

**Resolved gap.** Without the pool write side, real-device samples from the first release could not
have been re-used as the same distribution. Section 1 is therefore part of the first integration.
`uhf_sqd` writes the additive `ref_meta` keys described there.

**Environment variables.** The source enables several deferred features through environment
variables (`FE4S4_CKPT_DIR`, `FE4S4_RECOMB_*`, `OO_*`, `SBD_ALLOW_POOL_MISMATCH`). `uhf-sqd`
does not read them. Proposal: log a warning when one of them is set, so that users of the
existing launcher scripts do not assume a feature is active.
