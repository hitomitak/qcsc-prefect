# Cluster runbook: saved eight-shot H2 samples to Fulqrum

The user reported PASS for this saved-result stage on 2026-10-26. For the next
stage, use the [single-flow end-to-end guide](README_end_to_end.md). It reuses
the existing environments/blocks and provides a verifier without manual input
manifests or `sacct`. Do not repeat block registration for that new flow.

These instructions are for a future user-operated validation. Writing this
runbook did not execute anything on the cluster. Step 7 submits classical Slurm
jobs; earlier steps prepare a separate checkout and environments, inspect saved
files, build the native solver and register new Prefect blocks. No step calls
`QrmiRuntime.sampler`, reserves a QPU or modifies PR #42.

Use Bash and keep the same shell for the steps below. Run setup/build commands
on a host where those operations are permitted. The workflow host and compute
nodes must see the new checkout, virtual environments, input and output at the
same absolute paths under `/shared`. The native build needs a C++17/OpenMP
toolchain, Git/submodules, Python 3.12 and access to dependency repositories.
The workflow environment needs `uv`, Slurm client tools and the same Prefect
Cloud workspace/API configuration used for the successful sampling run.

## 1. Get the integration branch from Git

Create a new shared directory so existing checkouts, branches and QRMI virtual
environments remain available:

```bash
set -euo pipefail
export VALIDATION_ROOT="$(mktemp -d /shared/qrmi-pr42-recovery.XXXXXX)"
export CHECKOUT="$VALIDATION_ROOT/qcsc-prefect"
git clone --single-branch --branch qrmi-pr42-recovery-v0.3.0 \
  https://github.com/hitomitak/qcsc-prefect.git "$CHECKOUT"
cd "$CHECKOUT"
git log -1 --format='%H %s'
git status --short
git rev-parse HEAD > "$VALIDATION_ROOT/checkout-commit.txt"

export UV_PROJECT_ENVIRONMENT="$VALIDATION_ROOT/control-venv"
export FULQRUM_VENV="$VALIDATION_ROOT/fulqrum-venv"
export FULQRUM_SRC="$VALIDATION_ROOT/fulqrum-src"
export RECOVERY_OUTPUT="$VALIDATION_ROOT/results"
export FULQRUM_JOB_ROOT="$VALIDATION_ROOT/solver-jobs"
mkdir -p "$RECOVERY_OUTPUT/reference" "$RECOVERY_OUTPUT/fulqrum"
```

The URL is the user's `hitomitak` fork. If the cluster requires SSH Git access,
use `git@github.com:hitomitak/qcsc-prefect.git` for the clone instead. The printed
commit should match the pushed integration commit communicated with this runbook.
Do not switch or pull an unrelated existing checkout.

## 2. Select the actual saved eight-shot run

Locate the old sampling summaries. The initial 100-shot `meas`-only run is a
different input from the reset-mitigated run that retained eight shots:

```bash
find /shared -type f -name sampling_summary.json -print 2>/dev/null || true
```

Set the **actual** `run_<id>` path from the successful reset-mitigation run:

```bash
export SAVED_RUN="/shared/REPLACE_WITH_ACTUAL_PATH/run_id"
test -f "$SAVED_RUN/sampling_summary.json"
test -f "$SAVED_RUN/bitstrings.txt"
```

Do not rebuild `bitstrings.txt` from printed counts: this loses shot ordering.
The full `quantum/<job_id>` directory must remain present, including raw
`result.json`, request/compiler metadata, both QPY files and Slurm metadata.
The input can stay at its original path; no copy or edit is necessary.

## 3. Install the 0.3.0 control environment and validate input

```bash
cd "$CHECKOUT"
uv sync --all-packages --python 3.12
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  -r examples/qrmi_sqd_validation/requirements.txt \
  -e algorithms/qcsc_workflow_utility \
  -e packages/qcsc-prefect-fulqrum \
  -e 'examples/qrmi_sqd_validation[recovery]'
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
uv pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  > "$VALIDATION_ROOT/control-freeze.txt"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
uv run --no-sync python - <<'PY'
import json
import os
from importlib.metadata import version
from pathlib import Path
from examples.qrmi_sqd_validation.pinned_recovery import verify_sources
from examples.qrmi_sqd_validation.recovery import load_saved_h2_run

root = Path(os.environ["SAVED_RUN"]).resolve()
saved = load_saved_h2_run(root)
assert saved.requested_shots == 100
assert saved.reset_mitigation is True
assert saved.bits.num_bits == 4 and saved.bits.num_shots == 8
expected = {"1101": 1, "0010": 1, "1100": 2, "0001": 1,
            "0110": 1, "1110": 1, "1001": 1}
assert saved.bits.get_counts() == expected
verify_sources()
for name in ("qcsc-prefect-qiskit", "qcsc-prefect-blocks",
             "qcsc-prefect-core", "qcsc-prefect-executor"):
    assert version(name) == "0.3.0", (name, version(name))
assert version("qcsc-prefect-fulqrum") == "0.3.0.dev0"
manifest = {"sampling_run": str(root), "input_sha256": saved.hashes,
            "checkout_commit": (Path(os.environ["VALIDATION_ROOT"]) /
                                "checkout-commit.txt").read_text().strip()}
(Path(os.environ["VALIDATION_ROOT"]) / "sampling-input.json").write_text(
    json.dumps(manifest, indent=2))
print("PASS: original H2 / 100 requested / 8 retained / 4 bits / reset selection")
print(json.dumps(saved.bits.get_counts(), indent=2))
PY

uv run --no-sync pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_saved_recovery.py -q
```

This focused test suite uses synthetic fixtures and blocks QPU/network calls;
the preceding input check is what validates the user's actual saved files.
The native test normally skips in the control environment because the native
backend is installed separately below. Always use `--no-sync` after this setup
to retain the explicit application and scientific dependencies.

## 4. Run reference configuration recovery on the saved files

This performs the H2 reference calculation on the workflow host, without Slurm
or QPU submission. Use an allowed CPU host if local calculations are restricted
on the login node:

```bash
uv run --no-sync python -m examples.qrmi_sqd_validation.recovery \
  --sampling-run "$SAVED_RUN" \
  --output-root "$RECOVERY_OUTPUT/reference" \
  --backend reference \
  --subspace-dim 4 --num-batches 1 --max-iterations 1 --seed 24 \
  2>&1 | tee "$VALIDATION_ROOT/reference.log"
```

Expected: `status: completed`, `backend: reference`, `requested_shots: 100`,
`retained_shots: 8`, and finite electronic/total energies. Do not interpret this
as a native Fulqrum result. If input validation fails, stop and inspect the saved
run; do not invoke the sampling entrypoint to replace it.

## 5. Build the pinned native backend in a separate compute environment

Use the system GNU compiler below, or load the site's supported OpenMP compiler
module first and set `CC`/`CXX` to its executable names. Any required runtime
module/library configuration must also be available to the Slurm solver jobs.
The default instructions assume a system compiler/runtime visible on all nodes.

```bash
cd "$CHECKOUT"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
command -v "$CC"
command -v "$CXX"
"$CXX" --version
export FULQRUM_REF=52d7c66ab0986580b00ad9382fe4d67bbe4d883e
git clone https://github.com/qiskit-community/fulqrum.git "$FULQRUM_SRC"
git -C "$FULQRUM_SRC" checkout --detach "$FULQRUM_REF"
git -C "$FULQRUM_SRC" submodule update --init --recursive
test "$(git -C "$FULQRUM_SRC" rev-parse HEAD)" = "$FULQRUM_REF"
git -C "$FULQRUM_SRC" submodule status --recursive \
  > "$VALIDATION_ROOT/fulqrum-submodules.txt"

export PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export FQ_BUILD_PARALLEL=2
unset FQ_ARCH
export FULQRUM_WITH_PRIMME=0
bash packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh \
  "$FULQRUM_VENV" "$FULQRUM_SRC" \
  2>&1 | tee "$VALIDATION_ROOT/fulqrum-build.log"
"$FULQRUM_VENV/bin/python" -m pip install --no-deps \
  -e "$CHECKOUT/packages/qcsc-prefect-fulqrum"

(cd / && "$FULQRUM_VENV/bin/python" -c \
  'import fulqrum; from fulqrum.convert.integrals import integrals_to_fq_fermionic_op; print("fulqrum", fulqrum.__version__, fulqrum.__file__)')
"$FULQRUM_VENV/bin/qcsc-fulqrum-solve" --help
"$FULQRUM_VENV/bin/python" -m pip freeze \
  > "$VALIDATION_ROOT/compute-freeze.txt"
```

Passing an explicit fresh source directory avoids the original build script's
default removal/replacement of `external/fulqrum`. No existing QRMI environment
is modified. `FQ_ARCH` is unset for a portable build; the build and compute hosts
must still have compatible OS/CPU architecture and shared-library runtimes.
This check imports the installed backend; numerical execution occurs in step 7.
Stop on build/import failure. The required fixed native source is used instead
of installing an arbitrary PyPI `fulqrum` release. For this four-dimensional H2
check the adapter uses dense diagonalization, so PRIMME is unnecessary.

The minimal compute environment intentionally installs the adapter with
`--no-deps`: its CLI uses the native backend, NumPy and SciPy, and does not need
the workflow-side Prefect/core dependencies. The full dependency check belongs
to the control environment in step 3.

## 6. Register dedicated CPU solver blocks

Keep the same `PREFECT_API_URL`/workspace authentication as the successful
sampling run. The control environment's `prefect config view` should identify
that workspace. Do not copy authentication values into the run directory.

```bash
"$UV_PROJECT_ENVIRONMENT/bin/prefect" config view
sinfo -o '%P %a %l %D'
```

Choose an available **CPU** partition and its account if required. Replace the
partition placeholder; `test_heron` is a QPU resource, not a CPU partition:

```bash
export CPU_PARTITION="REPLACE_WITH_CPU_PARTITION"
export CPU_ACCOUNT=""
export BLOCK_PREFIX="qrmi-h2-recovery-$(uv run --no-sync python -c 'from uuid import uuid4; print(uuid4().hex[:12])')"
uv run --no-sync python - <<'PY'
import json
import os
from pathlib import Path
from qcsc_prefect_fulqrum import create_fulqrum_blocks, FulqrumSolverJob
from qcsc_prefect_blocks.common.blocks import HPCProfileBlock

prefix = os.environ["BLOCK_PREFIX"]
queue = os.environ["CPU_PARTITION"]
assert queue and not queue.startswith("REPLACE_")
names = create_fulqrum_blocks(
    hpc_target="slurm",
    queue=queue, project=os.environ["CPU_ACCOUNT"],
    root_dir=os.environ["FULQRUM_JOB_ROOT"],
    fulqrum_executable=str(Path(os.environ["FULQRUM_VENV"]) / "bin/qcsc-fulqrum-solve"),
    command_block_name=prefix + "-command",
    execution_profile_block_name=prefix + "-execution",
    hpc_profile_block_name=prefix + "-hpc",
    solver_block_name=prefix + "-solver",
    launcher="single", num_nodes=1, mpiprocs=1, ompthreads=1,
    walltime="00:05:00", timeout_seconds=300,
    environments={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    # Set modules=["site-compiler-module"] here if your compiler needs it at runtime.
)
solver = FulqrumSolverJob.load(names["solver_block_name"])
hpc = HPCProfileBlock.load(solver.hpc_profile_block_name)
assert hpc.hpc_target == "slurm" and not hpc.slurm_qpu
assert hpc.queue_cpu == queue
path = Path(os.environ["VALIDATION_ROOT"]) / "solver-blocks.json"
path.write_text(json.dumps(names, indent=2))
print(json.dumps(names, indent=2))
print("PASS: separate classical Slurm profile; no QPU allocation")
PY
export SOLVER_BLOCK="$BLOCK_PREFIX-solver"
```

This writes four new Prefect block documents but submits no jobs. A fresh random
prefix avoids the helper's `overwrite=True` affecting existing documents. The
old `poc-qrmi-*` blocks are not reused or changed. If the site needs a memory,
module, QoS or other scheduler setting, apply it to this dedicated profile
before proceeding; do not copy a QPU allocation directive.

## 7. Run configuration recovery with the actual Slurm Fulqrum backend

This is the first command in this runbook that submits a scheduler job. It runs
only the classical eigensolver; the quantum input remains the saved eight shots.

```bash
cd "$CHECKOUT"
uv run --no-sync python -m examples.qrmi_sqd_validation.recovery \
  --sampling-run "$SAVED_RUN" \
  --output-root "$RECOVERY_OUTPUT/fulqrum" \
  --backend fulqrum-block --solver-block "$SOLVER_BLOCK" \
  --subspace-dim 4 --num-batches 1 --max-iterations 1 --seed 24 \
  2>&1 | tee "$VALIDATION_ROOT/fulqrum-recovery.log"
```

The expected full H2 subspace needs one solver call with these settings. The
adapter accepts output only after `COMPLETED` with exit status zero. It does not
fall back to PySCF when native Fulqrum fails. Prefect's completed flow state alone
is insufficient: check the files and energy comparison below.

## 8. Verify actual output, energy convention and preserved input

The checker expects one new successful reference and Fulqrum recovery directory
in this fresh validation root. It reads authoritative saved JSON rather than
parsing mixed Prefect/CLI logs:

```bash
uv run --no-sync python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from pyscf import fci, lib

base = Path(os.environ["VALIDATION_ROOT"])
results = Path(os.environ["RECOVERY_OUTPUT"])
manifest = json.loads((base / "sampling-input.json").read_text())
sampling = Path(manifest["sampling_run"])
for name, digest in manifest["input_sha256"].items():
    assert hashlib.sha256((sampling / name).read_bytes()).hexdigest() == digest, name

def read_success(backend_directory, backend_name):
    candidates = list((results / backend_directory).glob("recovery_*/recovery_summary.json"))
    assert len(candidates) == 1, candidates
    summary = json.loads(candidates[0].read_text())
    assert summary["status"] == "completed" and summary["backend"] == backend_name
    assert summary["qiskit_integration_version"] == "0.3.0"
    assert summary["requested_shots"] == 100 and summary["retained_shots"] == 8
    assert summary["solver_calls"] == 1 and summary["records"][0]["dimension"] == 4
    assert np.isclose(summary["total_energy"], summary["electronic_energy"] +
                      summary["nuclear_repulsion_energy"], atol=1e-12, rtol=0)
    assert not (candidates[0].parent / "failure.json").exists()
    return summary

reference = read_success("reference", "reference")
actual = read_success("fulqrum", "fulqrum-block")
assert actual["solver_block"] == os.environ["SOLVER_BLOCK"]
assert np.isclose(actual["total_energy"], reference["total_energy"], atol=1e-8, rtol=0)
with np.load(Path(actual["work_dir"]) / "integrals.npz", allow_pickle=False) as data:
    with lib.with_omp_threads(1):
        electronic_fci = fci.direct_spin1.kernel(
            data["one_body_tensor"], data["two_body_tensor"], 2, (1, 1))[0]
    expected_total = electronic_fci + float(data["nuclear_repulsion_energy"])
assert np.isclose(actual["total_energy"], expected_total, atol=1e-8, rtol=0)

job_ids = []
for directory in actual["solver_work_dirs"]:
    work = Path(directory)
    job = json.loads((work / "slurm_job.json").read_text())
    result = json.loads((work / "result.json").read_text())
    assert job["state"] == "COMPLETED" and job["exit_status"] == 0
    assert (work / "output.npz").is_file() and not (work / "error.json").exists()
    assert result["metadata"]["fulqrum_version"]
    assert result["metadata"]["subspace_dim"] == 4
    assert result["metadata"]["solver_path"] == "dense"
    assert np.isclose(result["energy"], actual["electronic_energy"], atol=1e-12, rtol=0)
    script = (work / "fulqrum_solver.slurm").read_text()
    assert "--qpu" not in script
    job_ids.append(str(job["job_id"]))
    print("Slurm job:", job["job_id"], "directory:", work)
    print("Fulqrum metadata:", json.dumps(result["metadata"], indent=2))
assert len(job_ids) == 1
(base / "solver-job-ids.txt").write_text(",".join(job_ids))
(base / "validated-recovery-summary.json").write_text(json.dumps(actual, indent=2))
print("PASS: saved 8 shots -> pinned recovery -> native Fulqrum via classical Slurm")
print("Total energy:", actual["total_energy"], "FCI:", expected_total)
print("Quantum input hashes unchanged:", len(manifest["input_sha256"]))
PY

sacct -j "$(cat "$VALIDATION_ROOT/solver-job-ids.txt")" \
  --format=JobID,JobName,State,ExitCode,NodeList
```

Check the compute host and `COMPLETED / 0:0` in `sacct`, together with
`output.out`/`output.err` in the printed solver job directory. The H2 total energy
should agree with both reference paths to `1e-8` Hartree. The nuclear constant
must appear exactly once. This validates the saved-result/solver integration;
four-dimensional H2 spans the full fixed-electron-number space, so the energy
agreement does not establish quantum sampling accuracy, eight-shot statistical
quality, sparse/PRIMME execution or the full production Fulqrum workflow.

Keep the entire validation root, actual sampling run, commit record, environment
freezes, native-source/submodule record, logs and JSON/NPZ outputs as evidence.
The commands above require no changes to PR #42 and create no pull request.

## If something fails

- Build/import failure: inspect `fulqrum-build.log` for OpenMP/compiler headers,
  missing submodules, CPU/OS mismatch or runtime libraries. Do not continue with
  an older arbitrary native version.
- Input validation failure: inspect the original summary, ordered bitstrings and
  raw `meas`/`test` result. A missing input must be recovered from the saved run;
  do not fill it from counts or submit replacement sampling.
- Classical job failure: retain the newest `recovery_*/failure.json`, its pending
  `solve_0000.json`, and every listed solver directory. Inspect `slurm_job.json`,
  `output.out`, `output.err` and `error.json` before retrying.
- A 300-second wait timeout does not cancel the Slurm job. Check `squeue`/`sacct`
  and the solver directories before any rerun; cancel explicitly with `scancel`
  only if that is intended.
- Retrying step 7 creates a new classical job. Step 8 intentionally rejects
  multiple success summaries in the same backend directory; select the exact
  run you intend to validate or use a fresh output directory for the retry.
