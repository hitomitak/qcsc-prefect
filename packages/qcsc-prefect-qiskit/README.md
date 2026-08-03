# QCSC Prefect Qiskit

Native Qiskit Runtime integration utilities for QCSC Prefect workflows.

## Durable submit-or-attach

The split Sampler and Estimator submit tasks support an opt-in durable mode.
Pass all three of `submission_key`, `spec_hash`, and `journal_path`; calls that
omit them keep the existing submit behavior.

```python
job_reference = await submit_sampler_job_task(
    pubs,
    runtime_block_name="runtime-config-block",
    shots=shots,
    input_digest=input_digest,
    submission_key=submission_key,
    spec_hash=resolved_spec_hash,
    journal_path=shared_journal_path,
)
```

The caller owns the submission-key policy. The library does not derive a
campaign or item key. `spec_hash` must cover every input that can change the
result, including the primitive type, pubs/input digest, backend, shots or
precision, and primitive options. Reusing one key with a different hash raises
`QiskitSpecHashMismatchError` before a Runtime submit or search.

Durable mode writes a SQLite `PREPARED` claim before `SamplerV2.run()` or
`EstimatorV2.run()`. Exactly one concurrent caller owns that claim. A short
SHA-256-derived tag is inserted into `options.environment.job_tags` before the
primitive is created, so a process that stops after Runtime accepts the job but
before the job ID is recorded can recover with `QiskitRuntimeService.jobs()`.
The key and spec hash are not included in the tag, and additional tags must not
contain credentials or other secrets.

Recovery always uses a journaled job ID first. With no ID, it searches the
stable tag and creation window. One validated result is attached; multiple
results enter `AWAITING_OPERATOR` immediately; zero results remain `PREPARED`
during the configured grace and then enter `AWAITING_OPERATOR`. It never picks
the newest result and never automatically resubmits an ambiguous `PREPARED`
record. The durable `QiskitJobReference` is stored in the shared journal, so
normal restart recovery does not depend on Prefect task-result retention.

The safety contract corresponds to the Slurm submit-or-attach path as follows:

| Contract | Slurm | Qiskit Runtime |
| --- | --- | --- |
| Pre-submit claim | Bulk registry `PREPARED` | Qiskit journal `PREPARED` |
| Search identity | Stable job name/comment digest | Stable pre-submit job tag |
| Durable external ID | Slurm job ID | Runtime job ID and `QiskitJobReference` |
| Ambiguous recovery | Zero/multiple candidates enter or retain hold | Zero after grace/multiple candidates enter hold |
| Automatic retry | Attach only when identity is unique | Attach only when identity is unique |

The SQLite journal schema is provider-neutral in purpose even though its table
and Python API are Qiskit-specific. `submission_key` is the primary key;
`spec_hash` and `stable_tag` are immutable identity fields; `status`,
`prepared_at`, and `updated_at` track lifecycle; `job_id` and
`job_reference_json` hold confirmed Runtime identity; and `last_error` plus
`held_at` preserve recovery evidence for operator inspection. The lifecycle is
`PREPARED` to either `SUBMITTED` or `AWAITING_OPERATOR`.

IBM Runtime currently permits at most eight job tags of at most 86 characters.
The library reserves one slot for its stable tag, preserves existing
`environment.job_tags`, removes duplicates, and rejects an over-limit submit
before any external side effect. Prefect submit-result caching is disabled when
durable parameters are present; the journal is the source of truth. Fetch-result
caching by job ID remains available.

`QiskitSubmissionJournal` exposes records for audit and operator inspection.
Store its SQLite file on the approved shared filesystem and do not commit the
journal, job references, results, or credentials to Git.
