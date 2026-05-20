# Level 0 Quickstart

This tutorial shows the smallest local workflow shape used by `qcsc-prefect`:
one task prepares mock quantum data, one task describes a local execution
profile, and one task stands in for an HPC solver job.

You do not need IBM Quantum credentials or real HPC credentials.

## What you will learn

- How a Prefect flow connects small workflow tasks
- Where a `qcsc-prefect` execution profile fits into the workflow
- How to run a local mock workflow before moving to an HPC tutorial

## Prerequisites

- Python 3.10 or newer
- A shell with `python` available
- No IBM Quantum account
- No Miyabi, Fugaku, or Slurm account

## Installation

Create and activate a virtual environment, then install `qcsc-prefect`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install qcsc-prefect
```

The Level 0 quickstart does not need the `qiskit` extra.

## Minimal local workflow

Create a file named `level0_quickstart.py`:

```bash
cat > level0_quickstart.py <<'PY'
from collections import Counter
from random import Random

from prefect import flow, task

from qcsc_prefect_core.models.execution_profile import ExecutionProfile


@task
def choose_execution_profile() -> ExecutionProfile:
    return ExecutionProfile(
        command_key="mock-bitcount",
        num_nodes=1,
        launcher="single",
        arguments=["--mock"],
    )


@task
def sample_bitstrings(shots: int = 8, seed: int = 123) -> list[str]:
    rng = Random(seed)
    return ["".join(rng.choice("01") for _ in range(4)) for _ in range(shots)]


@task
def run_mock_hpc_job(
    bitstrings: list[str],
    profile: ExecutionProfile,
) -> dict[str, int]:
    print(f"Mock submit: {profile.command_key} on {profile.num_nodes} node(s)")
    return dict(Counter(bitstrings))


@flow(name="level-0-qcsc-prefect-quickstart", log_prints=True)
def quickstart() -> dict[str, int]:
    profile = choose_execution_profile()
    bitstrings = sample_bitstrings()
    counts = run_mock_hpc_job(bitstrings, profile)
    print("Counts:", counts)
    return counts


if __name__ == "__main__":
    quickstart()
PY
```

Run it with an isolated Prefect home so existing Prefect profiles do not affect
the quickstart:

```bash
export PREFECT_HOME="$(pwd)/.prefect-level0"
unset PREFECT_API_URL
unset PREFECT_PROFILE
python level0_quickstart.py
```

Prefect may start a temporary local server for this run and stop it
automatically when the flow finishes. No external service or credentials are
required.

## Where to see results

For this Level 0 run, check the terminal output. You should see Prefect flow
and task logs, a mock submit message, and a small counts dictionary.

If you later connect to Prefect Cloud or a persistent local Prefect server,
the same flow shape can also be inspected in the Prefect UI.

## What to read next

- [Tutorial Roadmap](index.md)
- [Miyabi Workflow](create_qcsc_workflow_for_miyabi.md) for the Level 1/2
  BitCount path on Miyabi
- [Fugaku Workflow](create_qcsc_workflow_for_fugaku.md) for the Level 1/2
  BitCount path on Fugaku
- [Closed Loop Workflow](run_sbd_closed_loop_workflow.md) for the Level 3 SBD
  workflow

Level 1/2 tutorials add real HPC execution and optional native Qiskit Runtime
execution. Level 3 tutorials are advanced SBD closed-loop workflows.
