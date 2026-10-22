# QCSC Prefect Qiskit

Native Qiskit Runtime integration utilities for QCSC Prefect workflows.

QRMI Batch sampling through Slurm is available as
`qcsc_prefect.integrations.qiskit.qrmi.QrmiSlurmRuntime`.
Install the `slurm` extra on the workflow host to include the HPC executor.
The compute-node entrypoint is `python -m qcsc_prefect.integrations.qiskit.qrmi.worker`
or `qcsc-qrmi-sampler`.

See [QRMI/Slurm Sampler](../../docs/howto/howto_qrmi_slurm_sampler.md) for configuration,
algorithm hooks, the supported PR #42 call subset, migration, and validation limits.
