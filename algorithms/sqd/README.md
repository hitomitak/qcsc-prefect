# SQD Experiment with DICE Solver

This package provides the SQD experiment demonstrated in the paper [Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Superomputer](https://arxiv.org/abs/2405.05068). 
The experiment requires a [custom DICE solver](https://github.com/caleb-johnson/Dice) to diagonalize a molecular Hamiltonian using MPI.

---

## Getting Started

### 0. Pre-requisite

Build the Dice solver executable according to the guide in [prefect-dice](../../framework/prefect-dice/).

### 1. Set Up Python Environment

Create and activate a virtual environment:

```bash
uv venv -p 3.12 && source .venv/bin/activate
```

Install the project in editable mode:

```bash
uv pip install -e .
```

Verify installation:

```bash
uv pip list | grep qii-miyabi-kawasaki
```

Example output:

```
prefect-dice              0.1.0       /home/prefectuser/qii-miyabi-kawasaki/framework/prefect-dice
prefect-miyabi            0.1.0       /home/prefectuser/qii-miyabi-kawasaki/framework/prefect-miyabi
sqd-dice                  1.0.0       /home/prefectuser/qii-miyabi-kawasaki/algorithms/sqd
```

### 2. Set Prefect Server Endpoint

#### A. Using Prefect Cloud

We strongly recommend this option, as self-hosted Prefect servers do not support multi-tenancy.  
This means secrets (e.g., IBM Quantum API keys) may leak to other users with access to the same endpoint.

To set up Prefect Cloud:

```bash
prefect cloud login
```

Enter your Prefect Cloud API key to log in. 
Once logged in, verify the connection:

```bash
prefect config view
```

Example output:

```
🚀 you are connected to:
https://app.prefect.cloud/account/.../workspace/...
PREFECT_PROFILE='local'
PREFECT_API_KEY='********' (from profile)
PREFECT_API_URL='https://api.prefect.cloud/api/accounts/.../workspaces/...' (from profile)
```

#### B. Using Self-Hosted Prefect Server

If your project is confidential and cannot rely on third-party cloud storage, use a private server:

```bash
prefect config set PREFECT_API_URL=https://.../api
prefect config set PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=false
```

For example, to use the Prefect server hosted by mdx:

```bash
prefect config set PREFECT_API_URL=https://qii-kawasaki-miyabi-serv.cspp.cc.u-tokyo.ac.jp/prefect/api
```

Verify the connection:

```bash
prefect config view
```

Example output:

```
🚀 you are connected to:
https://qii-kawasaki-miyabi-serv.cspp.cc.u-tokyo.ac.jp/prefect
PREFECT_PROFILE='ephemeral'
PREFECT_API_URL='https://qii-kawasaki-miyabi-serv.cspp.cc.u-tokyo.ac.jp/prefect/api' (from profile)
PREFECT_SERVER_ALLOW_EPHEMERAL_MODE='False' (from profile)
```

### 3. Register Block Schemas

Register the data schemas for dependency blocks:

```bash
prefect block register -m prefect_qiskit
prefect block register -m prefect_qiskit.vendors
prefect block register -m prefect_dice
```

### 4. Configure `QuantumRuntime` Block

> [!NOTE]
> You can skip this step if you don't have access to any quantum resource. The SQD workflow will then switch to uniform random sampling.

To get the URL of the webpage to configure the block:
```bash
prefect block create quantum-runtime
```
Refer to the [Prefect Qiskit tutorial](https://qiskit-community.github.io/prefect-qiskit/tutorials/01_getting_started/) for guidance.

> [!IMPORTANT]
> Use `"sqd-runner-{$USER}"` as the block name. Replace `{$USER}` with your login username (e.g., `sqd-runner-prefectuser`) to isolate your settings.

You may also define primitive execution options using the Prefect Variable `sampler_options`.

### 5. Configure `DiceSHCISolverJob` Block

To get the URL of the webpage to configure the block:
```bash
prefect block create dice-shci-solver-job
```
Configure the block with the path to the `Dice` executable, working directory, and job accounting info.

Set the environment variable:

```yaml
{"LD_LIBRARY_PATH": "path/to/bin:$LD_LIBRARY_PATH"}
```

Ensure the required MPI module is loaded:

```yaml
["mpi/openmpi-x86_64"]
```

> [!IMPORTANT]
> Use `"sqd-solver-{$USER}"` as the block name.

If using the `local` executor, see [Special Tips for Local Shell](#special-tips-for-local-shell).

### 6. Deploy

Deploy the workflow and trigger it from the Prefect console:

```bash
sqd-deploy
```

Example output:

```
Your flow 'sqd-2405-05068' is being served and polling for scheduled runs!

To trigger a run for this flow, use the following command:

        $ prefect deployment run 'sqd-2405-05068/sqd_2405_05068'

You can also run your flow via the Prefect UI: ...
```

Ensure the deployed URL matches your Prefect server endpoint.

---

## Special Tips for Local Shell

When running experiments in a local shell (e.g., on a VM or laptop), keep the following in mind:

### ⚠️ Avoiding Race Conditions in Local Execution

If `sqd_num_batches > 1`, multiple `DiceSHCISolverJob` instances may invoke `mpirun` simultaneously, causing race conditions and errors like:

```
terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 22) >= this->size() (which is 22)
```

To avoid this, limit concurrency:

```bash
prefect concurrency-limit create "res: local" 1
```

This forces sequential execution—suitable for validation.  
For performance, use the `pbs` executor on Miyabi.

### 🛠️ Preventing Segmentation Faults from UCX

By default, `mpirun` may use UCX, which can cause segfaults or bus errors in shared-memory environments.

To disable UCX and use TCP:

```yaml
{
  "LD_LIBRARY_PATH": "path/to/bin:$LD_LIBRARY_PATH",
  "OMPI_MCA_pml": "ob1",
  "OMPI_MCA_btl": "tcp,self",
  "OMPI_MCA_btl_tcp_if_include": "lo"
}
```

**Explanation:**
- `OMPI_MCA_pml="ob1"`: Uses a stable messaging layer.
- `OMPI_MCA_btl="tcp,self"`: Enables TCP and self-communication.
- `OMPI_MCA_btl_tcp_if_include="lo"`: Restricts to loopback interface.

These settings ensure reliable local parallel execution.

---

## Contribution Guidelines

This package is a reference implementation of the SQD experiment described in the publication.  
It is considered feature-complete and is now in maintenance mode.

Further contributions are limited to bug fixes and improvements to Prefect usage patterns.  
Workflow developers may use this as a testbed for workflow technology research.
