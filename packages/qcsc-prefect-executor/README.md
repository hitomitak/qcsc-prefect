# QCSC Prefect Executor

Prefect executor integrations for local processes and QCSC HPC scheduler workflows.

## Local execution

Set `HPCProfileBlock.hpc_target` to `"local"` and map the command's executable
key to a path or command available on the Prefect worker. Local execution does
not generate a job script and does not call a scheduler. It invokes the command
directly with `cwd=work_dir`.

With `launcher="single"`, the command is:

```text
executable default_args... user_args...
```

With another launcher, it is:

```text
launcher mpi_options... executable default_args... user_args...
```

`environments` are merged into the worker process environment. Values are
passed literally without shell expansion. `modules` and `pre_commands` are
explicitly unsupported for local execution and cause `ValueError` before the
process starts.
