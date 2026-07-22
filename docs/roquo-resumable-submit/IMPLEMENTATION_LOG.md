# ROQUO resumable submit: implementation log

This log is updated in the same pull request as each implementation increment. “Real
machine verified” is used only after a human records an actual scheduler/runtime result in
`REAL_MACHINE_RUNBOOK.md`.

## Progress

| PR | G items | State | Changed files | What and why | Compatibility | Automated checks | Real machine | Open items |
|---|---|---|---|---|---|---|---|---|
| PR0 | Design gate for G1/G4/G5/G6 | approved | `docs/roquo-resumable-submit/{ATOMICITY_DECISION,IMPLEMENTATION_LOG,G_STATUS,REAL_MACHINE_RUNBOOK}.md` | Recorded storage/locking choices and the approved shared-SQLite-only direction before changing scheduler behavior. | Documentation only; no public API, schema, submit, monitor, or cancel behavior changed. | `uv run --with mkdocs-material --with 'mkdocstrings[python]' mkdocs build --strict` passed. No code tests required. | Atomicity preflight waived as a gate. Tests 1–5 remain future human gates. | No sidecar/external mutex. Revisit only if the trusted filesystem assumption fails. |

## PR0 review notes

- Verified target: `qcsc-prefect` commit `bb97c7f`, branch `slurm`, clean target
  working tree at review start.
- Current SQLite registry has a five-second busy timeout but no external access mutex and no
  compare-and-set submit claim.
- Current submit order remains unchanged in PR0: candidate check → prepare → scheduler
  submit → `mark_submitted`.
- SQLite official guidance makes a shared-NFS, multi-process deployment a design decision,
  not a safe default. The repository owner explicitly accepted that platform assumption
  and selected option A without a sidecar claim or external mutex.
- No ROQUO job, Prefect Cloud action, IBM Runtime request, commit, push, or pull request was
  performed as part of PR0.

## Future PR entry template

Copy this section for each PR and complete every field.

```text
PR:
G items:
State: implemented (CI) | real-machine verified | hold
Files:
What changed and why:
Backward compatibility:
Tests and exact commands:
Real-machine requirement/result:
Unresolved items:
gb_demo gap-checklist delta note:
```

## gb_demo gap-checklist delta note

PR0 does not change the G1–G9 implementation findings in
`gb_demo_2026/python/docs/QCSC_PREFECT_GAP_CHECKLIST.md`. After the owner approves the
atomicity mechanism, a future gb_demo documentation change should record the chosen
production registry path, filesystem assumptions, and claim primitive. The qcsc-prefect
PR must not modify the gb_demo document directly.
