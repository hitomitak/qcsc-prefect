# QCSC Prefect Adapters

Local process and scheduler adapters for QCSC Prefect workflows. Local commands
are executed without a shell or generated job script; scheduler targets use
their Jinja2 script templates.

## Identity recovery capability

Scheduler runtimes may implement the optional
`IdentityRecoveryRuntime.find_candidates_by_identity()` protocol from
`qcsc_prefect_adapters.base`. The backend owns its scheduler commands, output
parsing, allocation/step filtering, metadata normalization, and submission-time
window checks. It returns normalized candidates only; the shared executor keeps
the safety decisions for zero, one, or multiple candidates, attachment, grace,
and operator hold.

| Target | Identity recovery |
|---|---|
| Slurm | Implemented with active `squeue` and historical `sacct` lookup |
| Fugaku/PJM | Not implemented; capability dispatch raises an explicit error |
| Miyabi/PBS | Not implemented; capability dispatch raises an explicit error |
| Local | Not applicable; capability dispatch raises an explicit error |

To add resumable identity recovery for another scheduler, implement
`find_candidates_by_identity(identity)` on that backend runtime. Keep all
scheduler-specific query syntax and parsing in the adapter, return every
normalized candidate without choosing one, and add adapter tests for active,
historical, duplicate, step/array, metadata, and time-window behavior. The
shared executor can then dispatch to that runtime without reimplementing its
attach and hold state machine.
