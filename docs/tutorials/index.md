# Tutorial Roadmap

This page describes the staged tutorial path for `qcsc-prefect`. The goal is to
separate beginner-friendly workflows from advanced, production-like examples so
new users can learn one concept at a time.

## Staged tutorial levels

| Level | Focus | Credentials required | Status |
| --- | --- | --- | --- |
| Level 0 | Local or mock execution | No IBM Quantum credentials and no real HPC credentials | Planned / missing |
| Level 1 | Random quantum source + HPC execution | No IBM Quantum credentials; real HPC scheduler required for the main path | Partially covered |
| Level 2 | Native Qiskit + HPC execution | IBM Quantum credentials and HPC scheduler may be required | Incomplete / planned |
| Level 3 | Real application / SBD closed-loop workflow | Production-like quantum and HPC setup | Covered by advanced tutorials |

## Level 0: Local or mock execution

Level 0 should be the first stop for users who want to understand the basic
Prefect workflow shape without setting up IBM Quantum access or a real HPC
account.

- IBM Quantum credentials: not required
- Real HPC credentials: not required
- Goal: understand the basic flow/task/block shape used by `qcsc-prefect`
- Status: planned / missing

No Level 0 tutorial exists yet.

## Level 1: Random quantum source + HPC execution

Level 1 uses deterministic random bitstrings as the quantum source, then submits
and monitors an HPC job through `qcsc-prefect`.

- IBM Quantum credentials: not required for the random-source path
- HPC scheduler: real HPC scheduler for the main path
- Goal: confirm that `qcsc-prefect` can submit and monitor an HPC job
- Existing basis: `examples/prefect_bitcount_demo` supports
  `--quantum-source random`

Existing tutorials:

- [Miyabi Workflow](create_qcsc_workflow_for_miyabi.md)
  - Current: BitCount on Miyabi
  - Recommended level: Level 1 / Level 2
  - Note: the first path uses random quantum data with Miyabi HPC execution;
    IBM Quantum is an optional follow-up.
- [Fugaku Workflow](create_qcsc_workflow_for_fugaku.md)
  - Current: BitCount on Fugaku
  - Recommended level: Level 1 / Level 2
  - Note: the first path uses random quantum data with Fugaku HPC execution;
    SSL, IBM Quantum, and native Qiskit Runtime setup are optional real-device
    follow-up concerns.

Related local validation material:

- [Local Slurm Workflow](create_qcsc_workflow_for_local_slurm.md)
  - Current: Local Slurm + random/optional IBM
  - Useful when you want to test Slurm submission locally without real HPC
    credentials.
  - This is not the primary beginner path because it includes Docker Slurm,
    Prefect Cloud, manual block creation, and current quantum runtime
    installation assumptions.
  - If a hands-on Local Slurm tutorial is needed, it can be refined later as a
    Level 1 local scheduler variant.

## Level 2: Native Qiskit + HPC execution

Level 2 should connect native Qiskit execution and HPC execution in one
workflow.

- Uses Native Qiskit directly
- Uses `qcsc-prefect` helpers for Prefect integration, such as artifacts,
  retries, caching, or orchestration
- Goal: connect quantum execution and HPC execution in one workflow
- Status: incomplete / planned

Related material exists, but it is not yet an integrated tutorial:

- [Native Qiskit on Prefect](../howto/howto_use_native_qiskit_prefect.md)

## Level 3: Real application / SBD closed-loop workflow

Level 3 is the advanced path. These tutorials demonstrate the full QCSC pattern
with the SBD workflow, solver blocks, and production-like HPC settings.

- Uses the SBD workflow
- Uses solver blocks and production-like HPC settings
- Goal: demonstrate the full QCSC workflow pattern

Existing tutorials:

- [Closed Loop Workflow](run_sbd_closed_loop_workflow.md)
  - Current: SBD closed-loop on Miyabi
  - Recommended level: Level 3
  - Note: advanced tutorial. Includes solver build, deployment, editable
    install, and IBM Quantum assumptions.
- [Closed Loop Workflow (Fugaku)](run_sbd_closed_loop_workflow_fugaku.md)
  - Current: SBD closed-loop on Fugaku
  - Recommended level: Level 3
  - Note: advanced tutorial. Includes Fugaku, SBD, IBM Quantum, and deployment.

## Recommended path for first-time users

1. Start with the planned Level 0 tutorial when it becomes available.
2. Use the Level 1 random-source path to learn HPC submission and monitoring.
3. Move to Level 2 when you need Native Qiskit and HPC in the same workflow.
4. Use Level 3 only after the BitCount and HPC concepts are familiar.

Until Level 0 exists, use the Level 1 random-source path only when you already
have an HPC environment available. Use the Local Slurm workflow as a reference
for local scheduler validation, not as the default first tutorial.

## Known cleanup items

- Beginner tutorials still need follow-up work to reduce remaining platform
  setup burden.
- The Local Slurm workflow can stay as reference material until a hands-on
  Local Slurm tutorial is explicitly needed.
- Do not treat the planned Level 0 and Level 2 tutorials as existing
  documentation until they are added in separate PRs.
