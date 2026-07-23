# ROQUO resumable submit: Slurm identity contract

PR3 adds an opt-in identity contract to the Slurm adapter. It is deliberately
limited to Slurm; Fugaku and Miyabi script rendering is unchanged.

## Generated values

`build_slurm_job_identity(job_key, spec_hash)` returns:

- `job_name`: `qcsc-<normalized-job-key>-<24-hex-digest>`, truncated before
  the suffix when needed. Normalization replaces non-ASCII or non
  `[A-Za-z0-9_.-]` characters with `-` and the generated value is always
  scheduler-safe.
- `comment`: `qcsc-prefect-slurm-identity-v1:sha256:<64-hex-digest>`.

The SHA-256 digest covers an identity schema version, the full logical
`job_key`, and the complete immutable `spec_hash`, separated unambiguously.
Neither source value appears in the generated comment. Changing either input
therefore changes both identity values.

The job-name suffix is the first 24 hexadecimal digits (96 bits) of the full
digest. It remains intact after truncation, so two long job keys with the same
visible prefix retain distinct names except at the normal cryptographic
collision probability. Recovery must validate the complete comment, not only
the shortened job-name suffix.

## Limits and target confirmation

The library's `DEFAULT_SLURM_JOB_NAME_MAX_LENGTH` is **64** as a conservative
default. It is not a portable statement about Slurm's `MaxJobName` setting.
Before a ROQUO deployment chooses a different value, an operator must record
the target cluster's permitted job-name and comment behavior in Test 1 of
`REAL_MACHINE_RUNBOOK.md`, including preservation in both `squeue` and
`sacct`.

`SlurmJobRequest.job_name` and `.comment` remain optional. With neither field,
the rendered script is unchanged for existing callers. When supplied, both
values are restricted to safe single-token directive characters; this avoids
newline/directive injection because Slurm parses `#SBATCH` directives before
the shell begins.

PR4 will generate these values before the durable `PREPARED` claim, persist
them in the registry, and use the full comment during scheduler search and
attach. PR3 alone does not change submission recovery behavior.
