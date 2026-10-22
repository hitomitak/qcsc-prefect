# QCSC Prefect Adapters

Local process and scheduler adapters for QCSC Prefect workflows. Local commands
are executed without a shell or generated job script; scheduler targets use
their Jinja2 script templates.

Slurm single-job execution polls `scontrol` while the job is retained by the
controller, so it works without Slurm accounting storage. `sacct` is used only
when `scontrol` returns no matching job record. This does not provide historical
job recovery after the controller has purged a completed job; bulk monitoring
still uses `sacct`.
