# HPC Workflow Execution Design

This document specifies an HPC workflow execution architecture
that combines admin-defined execution profiles with user-adjustable tuning parameters.

The design allows users to flexibly adjust resource usage (nodes, memory, threads) 
while keeping HPC-specific complexity safely encapsulated in centrally managed profiles.
---

## 1. Design Goals

- Enable one workflow to run on multiple HPC systems (Fugaku, Miyabi, Slurm)
  without modification
- Reduce user burden by eliminating the need to write HPC-specific job blocks

---

## 2. Architectural Overview

```
[ Workflow ]
   │  (algorithm logic + run-time parameters)
   ▼
[ Command Block ]
   │  What to run (logical executable name)
   ▼
[ Execution Profile Block ]
   │  Recommended baseline configuration
   ▼
[ Tuning Parameters ]   ← user-adjustable (optional)
   │
   ▼
[ HPC Profile Block ]
   │  Scheduler / MPI / filesystem resolution
   ▼
[ Executor ]
```

---

## 3. Block Types

### 3.1 Command Block

**Purpose**
- Define what command is executed

**Characteristics**
- Logical executable name only
- No absolute paths
- No scheduler or resource logic

**Examples**
- `cmd-diag`
- `cmd-sbd`

---

### 3.2 Execution Profile Block (Admin task)

**Purpose**
- Provide a  execution baseline for a specific command

**Characteristics**
- Command-specific
- Safe, pre-validated defaults
- Intended to cover common use cases

**Typical contents**
- resource class (`cpu` / `gpu`)
- default node count
- default walltime
- default launcher
- default MPI hints
- default modules / environment variables

**Examples**
- `exec-diag-n2`
- `exec-diag-n16`
- `exec-diag-gpu`

> Execution Profiles express recommended intent, not fixed hardware allocations.

---

### 3.3 HPC Profile Block (Admin task, environment-specific)

**Purpose**
- Resolve execution intent into concrete HPC-specific settings

**Responsibilities**
- Scheduler type (PJM / PBS / Slurm)
- Batch template and submission logic
- Resource-class → queue / resource-group mapping
- MPI option derivation and validation
- Executable path resolution
- Enforcement of system limits
- Reference to UserContext Block

**Examples**
- `hpc-fugaku`
- `hpc-miyabi`

---

### 3.4 UserContext Block

**Purpose**
- Resolve execution identity information

**Responsibilities**
- Map `hpc_identity` → group / account / project
- Absorb differences between HPC environments
- Eliminate manual specification of group/account by users

---

## 4. Run-Time Parameters Specification

### 4.1 Required Parameters

```
hpc_target: string
  - e.g. "fugaku", "miyabi"

hpc_identity: string
  - user identifier (e.g. "z30541")

exec_profile: string
  - baseline execution profile (e.g. "exec-diag-n16")
```

---

### 5.2 Algorithm Parameters

Algorithm-specific inputs (files, numerical parameters, etc.)
are passed unchanged and are outside the scope of this specification.

---

### 5.3 Optional Tuning Parameters (User-adjustable)

Users may optionally provide tuning parameters to adjust resource usage
relative to the selected execution profile.

```yaml
tuning:
  nodes: int
  walltime: string
  ranks_per_node: int
  threads_per_rank: int
  mem_gib: int
```

#### Design Intent
- Tuning parameters allow lightweight customization
- Users can adapt execution to problem size without creating new profiles

---

## 6. Example: `exec-diag-n16` with User Tuning

### User Input

```yaml
exec_profile: exec-diag-n16
tuning:
  nodes: 32
  threads_per_rank: 4
```

### Resolution

- Execution Profile provides baseline intent (`diag`, CPU, large-scale)
- User tuning overrides node count and threading

Workflow code remains unchanged.

---

## 8. Responsibility Split

### Administrators
- Define and maintain:
  - HPC Profiles
  - Execution Profiles
  - UserContext mappings
- Encode best practices and policies
- Control safety and validation rules

### Users
- Write workflow logic
- Select execution profiles
- Adjust tuning parameters as needed
- Focus on algorithm development

---

## 9. Summary

This architecture replaces user-written HPC job definitions with
centrally managed execution profiles, while still allowing users
to tune resource usage in a controlled and transparent way.
