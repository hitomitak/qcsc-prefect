# Local PR #42 sampling integration with QRMI

`integration.py` adapts the sampling stage of commit
`2d435b19af4648cbbc5fbc6a27ff258c93817d1a` to QRMI/Slurm and returns the `BitArray`
expected by subsequent algorithm stages. The original PR, source fixture and
library implementation are unchanged by the adapter.

```text
CircuitParameters + ElectronicProperties + options
  -> pinned LUCJ construction
  -> ffSIM preparation -> QPY
  -> library Slurm submission
  -> compute-node QRMI Target -> pinned dedicated compiler
  -> task_runner -> PrimitiveResult
  -> meas extraction / matching test-shot postselection
  -> BitArray
```

## Python entrypoint

```python
from pathlib import Path
from qcsc_prefect.integrations.qiskit.qrmi import QrmiSlurmRuntime
from examples.qrmi_sqd_validation.integration import sample_bitstrings
from examples.qrmi_sqd_validation.pr42 import load_pr42

params = load_pr42().CircuitParameters(
    use_reset_mitigation=False,
    sabre_layout_trials=4,
    sabre_swap_trials=2,
    sabre_max_iterations=2,
)
runtime = QrmiSlurmRuntime(
    root_dir=Path("/shared/qrmi-pr42-bitarray/quantum"),
    command_block_name="poc-qrmi-command",
    execution_profile_block_name="poc-qrmi-execution",
    hpc_profile_block_name="poc-qrmi-hpc",
    resource="test_heron",
)
bits = await sample_bitstrings(
    circuit_params=params,
    elec_props=properties,  # ElectronicProperties from molecular preprocessing
    runtime=runtime,
    options={"params": {"shots": 100}},
)
assert bits.num_bits == 4
assert bits.num_shots == 100
```

Pass a runtime without compilation hooks or configuration. The adapter installs
its pinned hooks on a copy, leaving the caller's runtime unchanged and preserving
its resource, blocks, work directory and monitoring settings. Existing hook/config
values cause an error rather than being overwritten silently.

## Contract with the fixed PR

| Original sampling stage | Local adapter |
| --- | --- |
| `QuantumRuntime.load(runner_name)` | Inject an installed library runtime |
| `Variable.get(option_name)` | Inject `options`; no Prefect Variable reads or writes |
| Workflow-host Target and compilation | Compute-node Target and pinned compilation hook |
| LUCJ construction and dedicated compiler | Reuse the SHA-256-verified source definitions |
| Return `data.meas` | Preserve BitArray type, bit endianness and shot ordering |
| Select `test.bitcount() == 0` | Retain the corresponding meas shots in their original order |

The omitted-shots default is the PR's **100,000**, not the generic library's 100.
The adapter makes this explicit in the options passed to the library. The H2 CLI
explicitly defaults to 100 shots.
Only `options["params"]["shots"]` is supported; booleans, strings, floats,
nonpositive shots and extra Runtime options are rejected.

If mitigation rejects every shot, the adapter raises a descriptive `ValueError`.
The original also fails to construct an empty BitArray, but reports the underlying
container error. There is no random fallback. The sampling task and H2 flow disable
automatic retries, and this adapter adds no result cache.

## Offline verification

From the repository root, using the example dependencies:

```bash
PYTHONPATH=packages/qcsc-prefect-blocks/src \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  packages/qcsc-prefect-executor/tests/test_qrmi_sqd_slurm_poc.py -q --tb=short
```

Tests replace Slurm and QRMI service calls but run real QPY round trips, the pinned
compiler, QASM export/import, a fake CLI process and result decoding.
They cover H2 with 100 shots, CZ/CZ+RZZ Targets, and meas-only/meas-plus-test.
Asymmetric bitstrings and nonzero test shots check endianness, shot correspondence
and postselection ordering.

The unchanged pinned `sample_bitstrings` and `_resolve_shots` functions are also
loaded in the tests as a reference. Only their direct Runtime/target/compiler
boundary is stubbed for comparison. This does not run the original entire flow
unchanged. Additional cases cover default shots, invalid options, submission
failure, zero retained shots and saved H2 artifacts.

## Cluster entrypoint

Use the deployment, shared venv, Cloud workspace and three quantum blocks from
[the H2 cluster guide](README_h2_slurm.md). Copy the updated example directory and
tests. Verify both hosts import the same library and pinned hook.
The following command **submits a real Slurm and quantum workload**:

```bash
cd /shared/qcsc-prefect
source /shared/venv/qrmi/bin/activate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m examples.qrmi_sqd_validation.integration \
    --shots 100 --work-dir /shared/qrmi-pr42-bitarray
```

Omit `--reset-mitigation` for meas-only validation. Each `run_<ID>` retains:

- `sampling_summary.json`: provenance, requested/returned shots, width, counts and paths.
- `bitstrings.txt`: one retained shot per line, preserving ordering.
- `quantum/<ID>/`: QPY, request, config, report, raw result and Slurm records.

`h2_bitarray_flow()` returns a `BitArray`; the CLI prints its width, shots and
counts. Restore saved samples for the next algorithm stage using the saved width:

```python
bits = BitArray.from_samples(
    (run_dir / "bitstrings.txt").read_text().splitlines(),
    num_bits=summary["num_bits"],
)
```

The user reported cluster completion as Prefect run `uber-sloth`, returning
four bits, 100 shots and count total 100. This run preceded preparing the focused
branch on latest main; it has not been repeated on this exact branch.
Configuration recovery, Fulqrum execution, energy validation and hardware
acceptance of the optional test register remain outside the verified scope.
