# UHF-SQD verification: the source baseline and the port comparison

This directory has two parts:

- **A baseline** of the UHF-SQD flow it was ported from, together with the harness that
  produced it.
- **Tools that compare** `algorithms/closed_loop/uhf-sqd` and `qcsc_workflow_utility.uhf`
  against that baseline.

The two sit together so that they can be reviewed together. They are not run in CI: they need
the SBD drivers and an environment with the recorded versions.

| Path | Content |
|---|---|
| [PR0_BASELINE.md](PR0_BASELINE.md) | The baseline record: code under test, environment, inputs, settings, seeds, procedure, results and scope. It is kept as recorded. Its golden set lives in `golden/` here, not in the `tests/golden` it planned, and its `env/` is `versions/` here |
| `golden/` (332 KB) | The reduced reference set, from repetition 1 of each run |
| `harness/` | The baseline harness: `run_baseline.py`, `run_matrix.sh`, `make_golden.py`, `compare_golden.py`, `golden_check.sh`, `analyze_baseline.py`, `generate_inputs.py`, and `inprocess_prefect.py` (an in-process Prefect API for hosts that forbid `bind()`) |
| `inputs/` | The NH2 and H6 FCIDUMPs (the same files as `algorithms/closed_loop/uhf-sqd/data/`) and their generation parameters |
| `versions/` | The recorded versions: `pip-freeze.txt` (the baseline venv, 183 third-party pins), `port-venv-freeze.txt` (the comparison venv), `conda-toolchain-list.txt` and `build-and-binaries.txt` |
| `port/` | The comparison tools, listed below |
| `selftest.sh` | Checks that the matrix and comparison tools report failures as a non-zero exit status |

## Pinned revisions

| Item | Revision |
|---|---|
| Source flow | `kunalroy332/qcsc-prefect` @ `3968e585f7ca8137c127e5ee9f4fb383322d3d0d` (`bk/feature/bs-uhf-sqd-roquo`), flow `riken_sqd_de` |
| SBD library | `r-ccs-cms/sbd` @ `95bf45a5be9ffd977fac7b0aaa359833caf3401a` |
| Python and key packages | Python 3.12.15, numpy 2.5.4, scipy 1.18.1, pyscf 2.15.0, ffsim 0.0.86, qiskit 2.6.2, qiskit-addon-sqd 0.11.0, prefect 3.9.2 (all in `versions/pip-freeze.txt`) |
| Native toolchain | conda-forge clang 21.1.8, OpenMPI 5.0.11, OpenBLAS 0.3.34 (all in `versions/conda-toolchain-list.txt`) |

**Bit-for-bit equality holds in the recorded environment only**: macOS arm64, the versions
above, one MPI rank, one BLAS/OpenMP thread. Other versions, platforms or rank counts can change
floating-point results, and with them the sampled subspaces. A difference there is not by itself
a regression.

## 1. Reproduce the baseline

The procedure is in [PR0_BASELINE.md](PR0_BASELINE.md), "Procedure". In short:

1. Extract the source at `3968e58`.
2. Create the toolchain with the pins in `versions/conda-toolchain-list.txt`.
3. Compile the source's `algorithms/sbd/native/main.cc` against SBD `95bf45a`, with and without
   `-D_UHF`.
4. Create a venv with `versions/pip-freeze.txt` as constraints, and install the source
   packages.
5. Copy `inputs/` to `ROOT/inputs`.
6. Run:

```bash
bash harness/run_matrix.sh ROOT DIAG_UHF DIAG_RHF          # 2 systems x 3 settings x 2 runs
bash harness/golden_check.sh ROOT harness /tmp/golden-new  # rebuild, compare, tamper test
diff -r /tmp/golden-new golden                             # the rebuilt set equals this one
```

Both scripts exit non-zero on any failure:

- `run_matrix.sh` when a run fails;
- `golden_check.sh` when repetition 2 differs from the rebuilt set, or when the tamper test is
  not detected.

## 2. Compare the port

Build the drivers of this repository and create the comparison venv:

```bash
# SBD drivers from this repository, against the same SBD revision
UHF=1 bash packages/qcsc-prefect-sbd/native/build_sbd_slurm.sh   # -> native/diag_uhf

# A venv with the baseline's third-party versions and this repository's packages
grep -vE " @ file://|^-e " scripts/verification/uhf_sqd_golden/versions/pip-freeze.txt > constraints.txt
uv venv -p 3.12.15 venv-port
VIRTUAL_ENV=venv-port uv pip install -c constraints.txt \
  -e packages/qcsc-prefect-core -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks -e packages/qcsc-prefect-executor \
  -e packages/qcsc-prefect-qiskit -e packages/qcsc-prefect-sbd \
  -e algorithms/qcsc_workflow_utility -e algorithms/closed_loop/uhf-sqd pytest
```

Then run the matrix:

```bash
bash scripts/verification/uhf_sqd_golden/port/run_port_matrix.sh \
  OUT_ROOT venv-port scripts/verification/uhf_sqd_golden packages/qcsc-prefect-sbd/native/diag_uhf
```

The matrix runs the ported flow `uhf_sqd_de` with the baseline's inputs, settings and seeds:

- a random `QuantumSourceBlock`;
- solver blocks from `uhf_sqd.create_blocks`;
- an empty reference cache per run.

Each run is written in the baseline's layout. `port/compare_port.py` then checks it in two
parts:

1. **Computed outputs**, with the baseline's own reduction and exact comparison:
   - flow energy;
   - reference scalars and arrays;
   - telemetry rows;
   - every solver job's inputs and outputs, with the FCIDUMP compared by its integral records,
     because this repository's writer adds a layout marker to the header;
   - stored SQD arrays.
2. **Settings.** The source's flow parameters and solver block differ in shape from the port's.
   Each source field is matched, through an explicit table, to the port setting it stands for,
   or must be at the value with which its feature is off.

The script exits 0 only if all 12 runs succeed and pass both parts.

### Other comparison tools

| Tool | Checks |
|---|---|
| `port/pr2_vs_pr0_reference.py` | `qcsc_workflow_utility.uhf` against the baseline reference: integrals, t2 blocks, occupancies and the BS-UHF cache file. It covers a fresh SCF, cache reuse, and a cache file written by the source |
| `port/ast_compare_reference.py`, `port/ast_compare_workflow.py` | Which ported functions are unchanged at the AST level (docstrings, comments and formatting ignored), and the diff of the others |
| `port/pool_loader_compat.py` | A sample pool written by the port passes the source's unchanged loader check. Run its two steps in the two venvs |
| `port/transpile_equivalence.py` | The source's SABRE compilation against the port's QRMI hook, on a fake 127-qubit backend |
| `port/inprocess_plugin.py` | A pytest plugin that runs `prefect_test_harness` tests on the in-process Prefect API |

## 3. Self-test

```bash
bash scripts/verification/uhf_sqd_golden/selftest.sh [venv-port OUT_ROOT/runs/nh2_sto3g/D1/rep1]
```

Without arguments it needs no solver binary. It checks that `port/run_port_matrix.sh` and
`harness/run_matrix.sh` exit non-zero when every run fails. With a comparison venv and one
finished port run, it also checks that `port/compare_port.py`:

- passes the run as it is;
- fails a copy with one telemetry energy and one stored array element changed.

## Results recorded so far

- **Baseline.** Repetitions 1 and 2 are identical in every setting. Rebuilding the environment
  from the record gave identical runs. T0 matches FCI within 1e-13 Eh.
- **Port.**
  - The reference is identical to the baseline for NH2 and H6, on all three reference paths.
  - The flow is identical in 12 of 12 runs.
  - The settings mapped 42/42 per run.
  - The source loader accepts the port's pools.
- **Not covered:** real devices, more than one MPI rank, GPU builds, the antiferromagnetic-group
  guess, and other versions or platforms.
