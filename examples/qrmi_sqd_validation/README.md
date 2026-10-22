# Pinned PR #42: LUCJ QPY and transpilation validation

This example checks the H2/STO-3G LUCJ circuit and specialized transpiler from
[PR #42](https://github.com/qiskit-community/qcsc-prefect/pull/42), pinned to commit
`2d435b19af4648cbbc5fbc6a27ff258c93817d1a`.
The original PR is unchanged. The local snapshot is immutable and verified by SHA-256.

See [cluster deployment](README_h2_slurm.md) for QRMI/Slurm sampling and
[BitArray integration](README_integration.md) for the algorithm-facing return value.
The reusable runtime is documented in the
[QRMI/Slurm guide](../../docs/howto/howto_qrmi_slurm_sampler.md).

## QPY boundary and known limitation

| Path | Outcome |
| --- | --- |
| Raw LUCJ -> QPY round trip | Registers and ideal measurement probabilities preserved |
| Raw LUCJ -> pinned compiler -> QPY round trip | Passes |
| Raw LUCJ -> QPY round trip -> pinned compiler | Two strict expected failures |
| ffSIM PRE_INIT and gate expansion -> QPY -> pinned compiler -> QPY | Passes |

QPY preserves gate definitions but does not restore custom ffSIM Python classes.
After a raw round trip, the pinned compiler's `MergeOrbitalRotations` accesses
`norb` on an ordinary Qiskit `Gate` and raises `AttributeError`.
The measured probabilities are not corrupted by QPY.

`prepare_qpy_transport()` runs ffSIM PRE_INIT while the custom attributes still
exist, then expands remaining ffSIM gates. This preparation does not need a
hardware Target. `prepare_sampler_circuit()` connects it to the library's
workflow-side hook, while `transpile_for_target()` runs the unchanged pinned
compiler on the compute node and checks its ISA against the actual Target.

## Offline tests

Use a separate validation environment rather than upgrading a working QRMI
installation. The dependency versions are listed in `requirements.txt`.
They describe the tested Python 3.12 environment, not a cross-platform lockfile.

```bash
python3.12 -m venv /tmp/qcsc-sqd-validation
/tmp/qcsc-sqd-validation/bin/python -m pip install \
  -r examples/qrmi_sqd_validation/requirements.txt

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/qcsc-sqd-validation/bin/python -m pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_sqd_qpy_transpile.py \
  -q -rx --tb=short
```

Expected result: **6 passed, 2 xfailed**. Missing optional dependencies cause
skips; a skipped module does not establish validation. Tests block network
connections and invoke Prefect task `.fn` implementations to avoid a server.

To reproduce the known raw-QPY failure as an ordinary error:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_sqd_qpy_transpile.py \
  -k raw_qpy_before_pr_transpiler --runxfail -x -q --tb=short
```

This is expected to exit with code 1. The normal tests use strict xfail so an
unexpected pass signals that the workaround should be reviewed.

## Coverage

- H2 at 0.74 angstrom, STO-3G, two spatial orbitals, electrons `(1, 1)`,
  four logical qubits, one LUCJ layer. Molecular integrals and CCSD amplitudes
  are computed using the same local chemistry utility as the pinned PR.
- Four-bit `meas`, with and without the four-bit initial `test` register.
- Six-qubit linear fake Targets with CZ only or CZ plus RZZ.
- Target instruction compatibility, bound parameters, classical registers,
  layout/ancilla allocation and bit ordering.
- Exact ideal measurement probabilities with QPY tolerance `1e-12` and
  transpilation tolerance `1e-10`.
- The dedicated pass pipeline and optimization level 3 are retained.
  SABRE budgets are reduced to layout trials 4, swap trials 2 and iterations 2.

For ideal-state comparison, the initial `test` measurement observes an all-zero
state and is removed before statevector simulation. This does not validate
hardware mid-circuit measurement or noisy reset mitigation.

## Source provenance

`packages/qcsc-prefect-qiskit/tests/fixtures/pr42/` contains the unchanged source
`algorithms/sqd/sqd_fulqrum/main.py` and `source.json` with its commit and SHA-256.
The loader extracts `CircuitParameters`, `create_ansatz_circuits` and
`transpile_circuit` without modifying their bodies. Only Prefect task decoration
and logging are substituted for offline execution.
The reference compiler snapshot does not install or execute Fulqrum.
