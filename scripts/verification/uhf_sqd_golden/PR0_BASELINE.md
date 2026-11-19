# PR0: baseline of the unmodified source flow (UHF-SQD, `3968e58`)

**Status: final (2026-11-17).** This is the record of the source behaviour. It is kept as it is
and is not rewritten for later ports or runner changes. Later work compares against it.

This file records how the reference values for the UHF-SQD port were produced from the
**unmodified** source code. The port (PR2/PR3) must reproduce them. If it does not, the
difference is investigated and reported. The port's code is never adjusted to match these numbers
without approval. No scientific method, default or suspected bug was changed to produce them.

Planned location in the repository: `scripts/verification/uhf_sqd_golden/` for this file,
`harness/`, `inputs/` and `env/`, and `algorithms/closed_loop/uhf-sqd/tests/golden/` for
`golden/`.

## Code under test

| Item | Value |
|---|---|
| Source | `kunalroy332/qcsc-prefect` @ `3968e585f7ca8137c127e5ee9f4fb383322d3d0d` (`bk/feature/bs-uhf-sqd-roquo`), extracted with `git archive` and used read-only |
| Flow | `sbd.main.riken_sqd_de` |
| Blocks | Created with the source's own `algorithms/sbd/create_blocks.py`, run with `runpy` |
| Native solver | `algorithms/sbd/native/main.cc` @ `3968e58` (sha256 `0c5fe6e1ba474489f214ffad483850c732b41f469924ab4663879116134a44bb`) compiled against `r-ccs-cms/sbd` @ `95bf45a5be9ffd977fac7b0aaa359833caf3401a` |
| Binaries | `-D_UHF` gives `diag_uhf`, used by the flow. `diag_rhf` is registered but not used: the method is `uhf` |
| Compile command | `mpicxx -std=c++17 -fopenmp -O2 [-D_UHF] -c main.cc -Isbd/include`, linked with `-lopenblas -Wl,-rpath,<toolchain>/lib` |

Two notes on the build:

- **SBD revision.** The source build scripts clone SBD without pinning a revision. On 2026-11-13
  and on 2026-11-17, `origin/HEAD` of `r-ccs-cms/sbd` was `95bf45a`, which is also the upstream
  qcsc-prefect pin.
- **Binary hashes depend on the toolchain path.** Binary sha256 values change with the toolchain
  install prefix, which is embedded as an rpath. The values in `env/build-and-binaries.txt` are
  from the first build: diag_uhf `35226dea…`, diag_rhf `6765763d…`. The rebuild (see
  "Reproducibility") gave diag_uhf `97d92701…` and diag_rhf `70043fe2…`, and identical outputs.
  Identify binaries by source, revision, flags and outputs, not by their own hash.

## Environment

- macOS arm64. The first run was on macOS 27.0.1, the rebuild on Darwin 27.2.0.
- Python 3.12.15 in a dedicated uv virtual environment with the source packages installed:
  - `packages/qcsc-prefect{,-core,-adapters,-blocks,-executor,-qiskit,-dice}`;
  - `algorithms/qcsc_workflow_utility`;
  - `algorithms/sbd`.
- Third-party packages are pinned by `env/pip-freeze.txt`, with the local packages excluded
  (183 pins). Key versions:

  | Package | Version |
  |---|---|
  | prefect | 3.9.2 |
  | prefect-qiskit | 0.2.1 |
  | prefect-ray | 0.6.0 |
  | ray | 2.60.0 |
  | ffsim | 0.0.86 |
  | pyscf | 2.15.0 |
  | qiskit | 2.6.2 |
  | qiskit-addon-sqd | 0.11.0 |
  | qiskit-ibm-runtime | 0.47.0 |
  | numpy | 2.5.4 |
  | scipy | 1.18.1 |

- Native toolchain from conda-forge (micromamba 2.10.0):

  | Package | Version |
  |---|---|
  | clang, clangxx | 21.1.8 |
  | cxx-compiler | 2.0.0 |
  | ld64 | 956.6 |
  | libcxx | 23.1.5 |
  | llvm-openmp | 23.1.5 |
  | openblas, libopenblas | 0.3.34 |
  | openmpi | 5.0.11 |

  The full list is in `env/conda-toolchain-list.txt`.
- The source itself does not pin ffsim, PySCF, qiskit-addon-sqd or numpy. The pins above are the
  versions resolved on 2026-11-13.

### Execution infrastructure substitutions

The sandbox forbids `bind()`, so neither a Prefect server, an ephemeral server, Ray nor `mpiexec`
can start. Only the infrastructure was changed; no source file was touched.

1. **Prefect API in-process** (`harness/inprocess_prefect.py`).
   - `PREFECT_API_URL` is a sentinel URL.
   - Async clients use `httpx.ASGITransport` and sync clients use Starlette's `TestClient`
     transport, both into one in-process app, `prefect.server.api.server.create_app(ephemeral=True)`,
     with its SQLite database under `PREFECT_HOME`. The database is migrated at import.
   - Client events go to `NullEventsClient` through `EventsWorker.set_client_override`.
   - Flow, task, block, variable, artifact and cache semantics are Prefect's own.
2. **`SBD_TASK_RUNNER=concurrent`**, a mode the source offers (`main.py:70-73`), instead of the
   default Ray runner. In Prefect 3.9.2, `ConcurrentTaskRunner` is an alias of
   `ThreadPoolTaskRunner` (`prefect.task_runners.ConcurrentTaskRunner is ThreadPoolTaskRunner`), so
   the baseline ran on `ThreadPoolTaskRunner()` with its default `max_workers`. Ray was not run;
   see "Scope".
3. **The solver** is executed by the executor's `local` target with launcher `single`, as an MPI
   singleton: one rank.
4. **Per-run isolation.** `PREFECT_HOME`, `SBD_UHF_CACHE_DIR` and `SBD_LOCAL_STORAGE_DIR` point to
   a fresh directory per run, so the BS-UHF cache starts empty and the first SCF is computed.
   The cache stays enabled, as the source default.
5. **Threads.** `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.
6. **matplotlib** is not installed. The source then skips its summary plot (`sqd.py:234`, log line
   `plot skipped`). The plot does not affect any value.

## Inputs (`inputs/`, generated by `harness/generate_inputs.py`)

Restricted-MO FCIDUMPs with all orbitals active and ECORE = E_nuc (no frozen core). They were
written with `pyscf.tools.fcidump.from_integrals` from converged (RO)HF orbitals with
`conv_tol=1e-12`.

| System | Geometry (Angstrom) | Basis | (n_alpha, n_beta) | NORB | ECORE | sha256 |
|---|---|---|---|---|---|---|
| `nh2_sto3g` | N 0 0 0; H 0 0.80 0.59; H 0 -0.80 0.59 (ROHF orbitals) | STO-3G | (5, 4) | 7 | 7.783694531954562 | `cba014e4b5ea9c1a218651988f60f1bb43406d1f147140269a4caaa6130cb87f` |
| `h6_chain2A_sto3g` | linear H6, 2.0 spacing (RHF orbitals) | STO-3G | (3, 3) | 6 | 2.301920867502001 | `352de236a0f64e171df67e9b3415ed4f231f0cfc50c1514bf3a17ebbb51a29e2` |

The source's own UHF path (no AF groups; stability following plus energy-ordered shifts) turned
the restricted H6 input into a broken-symmetry reference with <S^2> = 2.7401 and
E_UHF = -2.8194898991218.

## Configurations

Everything not listed keeps the `3968e58` default.

| Config | Purpose | Flow parameters | Solver block |
|---|---|---|---|
| **D0** | Behaviour preservation: original defaults | `fcidump`, `quantum_source="random"` | `create_blocks.py` defaults: block 4, iteration 1, tolerance 1e-2, carryover_ratio 0.1, carryover_type 0, do_rdm 0, `--method uhf --solver-mode cpu --hpc-target local` |
| **D1** | Behaviour preservation: cover the in-scope code paths | D0 plus `sqd_dim=400`, `n_recovery_steps=2`, `n_batches=2` (K-batch recovery), `de_params.iterations=2` (DE mutation and crossover) | same as D0 |
| **T0** | Numerical correctness: full-space FCI comparison | same as D0 (default `sqd_dim=1_000_000`, which gives the whole space here) | D0 plus `--block 30 --iteration 60 --tolerance 1e-10` |

D0 and D1 are reference values for preserving behaviour. They are not correctness claims: the
original solver defaults are loose (see "Known limitation"). Correctness is judged only by T0.

**K-batch.** The recovery K-batch (`FlowParameters.n_batches`, default 1) and
`n_recovery_steps` (default 1) are **in the first integration**, and D1 exercises both with
value 2. The sampler's `SamplerParameters.n_shot_batches` (several device submissions) is
deferred and is not exercised here.

## Random seeds (source defaults, not changed)

| RNG | Seed or formula | Used where |
|---|---|---|
| `random_seed` | 24 | flow parameter |
| Random-source bits | `random_seed + (t+w)(t+w+1)/2 + w`, i.e. 24, 26, 29, 33 for trial 0 | `generate_bit_array_uniform`, `sqd.py:636-648` |
| `walker_rng` | the same formula | recovery and subsampling, `sqd.py:718-722` |
| `lucj.MODULE_RNG` | 4520 | perturbation of walkers 1-3 |
| `main.MODULE_RNG` | 4574 | DE mutation; used in D1 only |
| `sqd.MODULE_RNG` | 1333 | not drawn from in this scope |

## Procedure

```bash
ROOT=<empty dir>; H=<this directory>/harness
git -C <fork clone> archive 3968e585f7ca8137c127e5ee9f4fb383322d3d0d | tar -x -C $ROOT/src
# 1. Native toolchain (pins above) and binaries
micromamba create -p $ROOT/toolchain -c conda-forge clang=21.1.8 clangxx=21.1.8 cxx-compiler=2.0.0 \
  ld64=956.6 libcxx=23.1.5 libopenblas=0.3.34 llvm-openmp=23.1.5 openblas=0.3.34 openmpi=5.0.11 make
git -C <sbd clone> archive 95bf45a5be9ffd977fac7b0aaa359833caf3401a include | tar -x -C $ROOT/build/sbd
#    compile $ROOT/src/algorithms/sbd/native/main.cc twice (with and without -D_UHF), see above
# 2. Python environment pinned to the recorded freeze
grep -vE " @ file://|^-e " env/pip-freeze.txt > $ROOT/constraints.txt
uv venv -p 3.12.15 $ROOT/venv
VIRTUAL_ENV=$ROOT/venv uv pip install -c $ROOT/constraints.txt \
  $ROOT/src/packages/qcsc-prefect-core $ROOT/src/packages/qcsc-prefect-adapters \
  $ROOT/src/packages/qcsc-prefect-blocks $ROOT/src/packages/qcsc-prefect-executor \
  $ROOT/src/packages/qcsc-prefect-qiskit $ROOT/src/packages/qcsc-prefect-dice \
  $ROOT/src/packages/qcsc-prefect $ROOT/src/algorithms/qcsc_workflow_utility $ROOT/src/algorithms/sbd
# 3. Inputs, runs, analysis
cp inputs/* $ROOT/inputs/                     # or: $ROOT/venv/bin/python -I $H/generate_inputs.py $ROOT/inputs
bash $H/run_matrix.sh $ROOT <path>/diag_uhf <path>/diag_rhf     # 2 systems x 3 configs x 2 reps
$ROOT/venv/bin/python -I $H/analyze_baseline.py $ROOT           # rep equality, T0 vs PySCF FCI
bash $H/golden_check.sh $ROOT $H <out>/golden                   # build golden, compare, tamper test
```

Each run writes the following under `runs/<system>/<config>/rep{1,2}/`. The copies kept under
`results/` hold everything except `solver_root/`.

- `summary.json`: flow energy, reference quantities, telemetry rows, solver block values,
  sha256 of every solver job's inputs and outputs, and each job's `davidson_energy`;
- `reference.npz`: the integrals h1a, h1b, h2aa, h2ab, h2bb, the t2 blocks and the initial
  occupancies;
- `sqd_data_all.npz`: every walker's `sqd_data` arrays;
- `flow_parameters.json`;
- `solver_root/job_*`: the SBD inputs and outputs (run directories only);
- the log, gzipped as `run.log.gz`.

## Results (`results/baseline_report.json`)

**Repetition.** Rep 1 and rep 2 agree bit for bit in every configuration: flow energy, reference
arrays, telemetry (without paths and timestamps), all walker arrays, and the hashes of every
solver input and output file.

| System / config | Flow energy (Eh) | Net subspace dim / full | Solver jobs (cache hits) | vs E_FCI |
|---|---|---|---|---|
| NH2 D0 | -54.872567112074414 | 735 / 735 | 4 | +4.51e-6 (information only) |
| NH2 D1 | -54.87235626463056 | 400 / 735 | 32 (0) | +2.15e-4 (information only) |
| NH2 T0 | -54.87257162496175 | 735 / 735 | 4 | **-6.4e-14, pass** |
| H6 D0 | -2.837587005469417 | 400 / 400 | 4 | +9.61e-3 (information only) |
| H6 D1 | -2.837587005469417 | 400 / 400 | 16 (+16 cache hits) | +9.61e-3 (information only) |
| H6 T0 | -2.8471921339556046 | 400 / 400 | 4 | **-1.8e-15, pass** |

- **FCI reference.** `pyscf.fci.direct_spin1` on the restricted FCIDUMP integrals plus ECORE.
  This is independent of the UHF transform and of SBD: E_FCI(NH2) = -54.872571624962 and
  E_FCI(H6) = -2.847192133956.
- **T0 pass criterion.** Every walker's subspace is the whole space, and |E_flow - E_FCI| < 1e-8 Eh.
  All four walkers individually also match FCI to within 6.4e-14.
- **Reference quantities** (identical in all configurations):

  | System | E_HF from integrals + ECORE | <S^2> |
  |---|---|---|
  | NH2 | -54.830218365327 (= the cached E_UHF) | 0.7564 |
  | H6 | -2.8194898991218 | 2.7401 |

### Reproducibility after rebuilding the environment

On 2026-11-17 the whole environment was rebuilt from this record: a fresh source extract, the
toolchain recreated with the pins above, the binaries recompiled, and the venv installed with the
recorded constraints. The third-party freeze was identical, 183 of 183 pins. All 12 runs were then
compared with the copies made before the rebuild (`results/`). They are **identical bit for bit**:
energies, reference arrays, telemetry, walker arrays, and the hashes of the solver inputs and
outputs. The only difference was the absolute FCIDUMP path in `flow_parameters.json`, which the
golden set stores as a file name.

### Known limitation of the original solver defaults (kept, not changed)

With the source's `create_blocks.py` solver defaults (block 4, iteration 1, tolerance 1e-2), the
Davidson solve stops well before convergence, and the error grows with subspace size and
correlation. These are the shipped defaults and are kept as they are. Use explicit settings
(as in T0) when converged energies are needed. Measured on the flow's own H6 D0 solver inputs
(full space, 20 x 20), electronic energies:

| Solver settings | E_el (Eh) | Above converged |
|---|---|---|
| block 4, iteration 1, tol 1e-2 (default) | -5.139507873 | +9.61e-3 |
| block 4, iteration 2 | -5.142999563 | +6.11e-3 |
| block 4, iteration 5 | -5.146412515 | +2.70e-3 |
| block 8, iteration 1 | -5.144988209 | +4.12e-3 |
| block 8, iteration 5 | -5.148190249 | +9.23e-4 |
| block 20, iteration 1, 2 or 5 (stops at tol 1e-2) | -5.148967933 | +1.45e-4 |
| block 30, iteration 60, tol 1e-10 (T0) | -5.149113001 | 0 (= FCI) |

An earlier survey number of about 9.5e-5 Eh for H6 with the same default settings is consistent
with this table. That number was measured against the exact energy of a 10 x 10 random subspace
(100 determinants), not against FCI in the 20 x 20 space. Rerunning that case on these solver
inputs gives +9.53e-5 Eh: -5.12195288 with the defaults versus -5.12204821 converged.
Different, but equivalent, UHF orbitals change the loose result by less than 1e-8 Eh.

### Other observations (recorded, nothing changed)

1. **Prefect task cache.** In H6 D1, every batch and step has the same full-space subspace. The
   source's `@task(name="solve_eigenstate")` (default cache policy) therefore returned cached
   results for 16 of the 32 solves (`Cached(type=COMPLETED)` in the log). The values are
   identical to a recomputation. Upstream `qcsc-prefect-sbd` uses the same task name and the same
   default policy.
2. **Coverage gap in D1.** For H6 (C(6,3) = 20 strings per spin), `sqd_dim=400` allows 19 + HF
   strings, which is the whole space, so subsampling truncation is exercised only by NH2 D1.
3. **Incidental warnings.** The source prints pyscf/Newton warnings, for example "Newton solver
   expects mo_coeff with mo_occ" from its stability following. They are recorded in the logs and
   left as they are.

## Golden set (`golden/`, 332 KB) and comparison method

`golden/` is built from rep 1 by `harness/make_golden.py`. For each system and configuration:

- `<system>__<config>.json` holds:
  - the flow energy;
  - the flow parameters, with the FCIDUMP as a file name;
  - the solver block values;
  - the telemetry rows, without `sqd_data` paths and `last_updated`;
  - for each solver job: the sha256 of every input and output, `davidson_energy`, and
    `fcidump_records_sha256` (the sorted integral records with the header excluded);
  - the reference scalars and array hashes;
  - the content hashes of every `sqd_data` file;
  - the sha256 and shape of the raw and recovered bitstring arrays.
- `<system>__<config>.npz` holds the per-(trial, walker) arrays in full: `ucj_parameter`,
  `alphadets`, `betadets`, `avg_occupancy(_b)` and `carryover(_b)`.
- `<system>__reference.npz` holds the reference arrays in full: h1a, h1b, h2aa, h2ab, h2bb, the t2
  blocks and the initial occupancies.

`harness/compare_golden.py GOLDEN RUN SYSTEM CONFIG [--fcidump-records]` exits 0 only if
everything is equal:

- JSON values with `==`; JSON floats are Python repr, which round-trips exactly;
- arrays by dtype, shape and bytes;
- hashes by string.

Run-specific names (`sqd_data` file names, job directory names, paths, timestamps) are ignored.
`--fcidump-records` compares the solver FCIDUMP by `fcidump_records_sha256` instead of the
whole-file sha256. That is the mode for the port, whose FCIDUMP header carries an extra layout
marker.

`harness/golden_check.sh` rebuilds the golden set and compares rep 2 against it in both modes:
all 6 configurations are IDENTICAL. It then runs a tamper test that must fail:

- energy +1e-14;
- two `alphadets` entries swapped;
- one FCIDUMP record +1e-12.

The comparison reports 15 differences and exits 1. Regenerating the golden set gives
byte-identical files.

**How the port uses it.** The port's harness produces the same run layout with the ported flow
and the same inputs, settings and seeds. Then `compare_golden.py --fcidump-records` must report
IDENTICAL for D0, D1 and T0 of both systems. A difference is reported, not fixed by changing the
port.

## Scope of this baseline

**Valid for:**

- FCIDUMP input, UHF/BS-UHF without AF groups, and random sampling;
- CPU SBD as one MPI rank;
- `ThreadPoolTaskRunner()` (reached as `ConcurrentTaskRunner`), with one BLAS/OpenMP thread;
- the package and toolchain versions above;
- the configurations D0, D1 and T0 of the two systems.

**Not verified here** (not a completion condition of PR0):

- the Ray task runner (the source default); no Ray run has been compared;
- `ProcessPoolTaskRunner`, one of the runners the port may use (the port's own runner choice is
  verified when the workflow is ported);
- the real-device path (QRMI / IBM), including the SABRE layout cache and reset mitigation;
- MPI with more than one rank, and GPU binaries;
- other ffsim, PySCF, numpy or SBD versions;
- the AF-group (`FE4S4_AF_GROUPS`) guess;
- `n_shot_batches > 1`;
- the pool save path, which runs only for `real-device`.
