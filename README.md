# HPC-Agnostic Workflow Execution Design

This document describes an **HPC-agnostic workflow execution architecture** that
shifts HPC-specific complexity from individual users to centrally managed blocks.
As a result, users can run workflows on heterogeneous HPC systems
(e.g. Fugaku, Miyabi, Slurm) **by selecting predefined options at run time**, without
writing or maintaining HPC-specific job definitions.

---

## Motivation

### Before: User-driven HPC Configuration

Traditionally, users were required to:
- Create HPC-specific job blocks (PJM / PBS / Slurm)
- Determine node counts, MPI options, and resource groups
- Know executable paths and filesystem layouts
- Debug scheduler- and MPI-related failures

This required **HPC operational expertise at the user level**, leading to:
- High onboarding cost
- Repeated configuration mistakes
- Strong dependence on individual experience

---

### After: Admin-defined Profiles, User Selection

In this design:
- **Administrators define execution knowledge once**
- **Users select from safe, predefined execution profiles**
- Workflow code remains **unchanged and HPC-independent**

This enables users to focus on **algorithm logic and inputs**, not HPC operations.

---

## Core Design Principle

> **Separate algorithm logic from execution knowledge.**

- Workflow code describes *what to compute*
- Blocks encode *how and where to run it*
- Users choose *which predefined option to use*

---

## Architecture Overview

[ Workflow ]
│ (algorithm logic + run-time selection)
▼
[ Command Block ]
│ What to run (logical command name)
▼
[ Execution Profile Block ]
│ How to run (nodes / GPU / MPI / walltime)
▼
[ HPC Profile Block ]
│ Where & how to submit (PJM / PBS / Slurm specifics)
▼
[ Executor ]
│ submit / wait / status


---

## Block Types and Responsibilities

### Command Block (User-facing, HPC-agnostic)

**Purpose**
- Define *what* command is executed

**Characteristics**
- Logical executable name only (no absolute paths)
- No scheduler or resource details
- Reusable across all HPC systems

**Examples**
- `cmd-diag`
- `cmd-sbd`

---

### Execution Profile Block (Admin-defined, Command-specific)

**Purpose**
- Define *how* a specific command should be executed

**Characteristics**
- Command-specific presets
- Prevalidated and safe
- Selected by users at run time

**Typical contents**
- resource class (CPU / GPU)
- node count, walltime
- launcher (single / mpiexec / srun)
- MPI hints
- environment modules

**Examples**
- `exec-diag-n2`
- `exec-diag-n16`
- `exec-diag-gpu`

> These profiles represent **intent**, not raw scheduler directives.

---

### HPC Profile Block (Admin-defined, Environment-specific)

**Purpose**
- Encapsulate all HPC-specific knowledge

**Characteristics**
- Scheduler type (PJM / PBS / Slurm)
- Batch templates and submission logic
- Resource-class → queue / resource-group mapping
- MPI option derivation rules
- Executable path resolution
- Reference to user context resolution

**Examples**
- `hpc-fugaku`
- `hpc-miyabi`

---

### UserContext Block (Admin-defined)

**Purpose**
- Resolve execution identity information

**Responsibilities**
- Map `hpc_identity` → group / account / project
- Absorb differences between HPC systems
- Remove the need for users to specify group/account manually

---

## User Experience at Run Time

Users provide only:
- `hpc_target` (e.g. fugaku / miyabi)
- `hpc_identity` (user ID)
- `exec_profile` (e.g. diag-n16 / diag-gpu)
- Algorithm-specific inputs

Users **do NOT** provide:
- scheduler directives
- resource groups / queues
- MPI options
- executable paths

> Users select from **curated options**, rather than constructing jobs themselves.

---

## Example: `diag-n16` on Fugaku vs Miyabi

### User Intent (Same in Both Cases)

- Command: `cmd-diag`
- Execution Profile: `exec-diag-n16`
- Workflow code: **identical**

---

### Fugaku Resolution

Resolved by `hpc-fugaku`:
- Scheduler: PJM
- Node count: 16
- Resource group selected for CPU jobs
- MPI options derived for Fugaku topology
- Executable path resolved to Fugaku filesystem

---

### Miyabi Resolution

Resolved by `hpc-miyabi`:
- Scheduler: PBS / Slurm
- Node count adjusted for memory-rich nodes (e.g. 2 nodes)
- Queue / project selected automatically
- MPI options derived for Miyabi
- Executable path resolved to Miyabi filesystem

---

### Key Observation

**What changes**
- Node count
- Scheduler directives
- MPI configuration
- Executable path

**What does not**
- Workflow code
- Command selection
- Execution intent

---

## Responsibility Split

### Administrators
- Define HPC Profiles
- Define Execution Profiles
- Maintain UserContext mappings
- Encode best practices and policies

### Users
- Write workflow logic
- Select execution profiles
- Provide algorithm inputs

This clean separation:
- Reduces user burden
- Prevents configuration errors
- Centralizes HPC expertise

---

## Key Benefits

- Fully HPC-agnostic workflows
- Safe and reproducible execution
- Centralized operational knowledge
- Faster onboarding for new users
- Easier long-term maintenance

---

## Summary

> This design moves HPC complexity out of user workflows and into centrally managed execution profiles.  
> Users no longer *construct* HPC jobs — they *select* how to run them.

**Workflows describe algorithms.  
Blocks encode execution knowledge.**

