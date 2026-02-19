# Run SBD Closed-loop Workflow (After BitCounts Tutorial)

This tutorial walks us through reproducing an Sample-based Quantum Diagonalization (SQD) experiment from the paper [Closed-loop calculations of electronic structure on a quantum processor and a classical supercomputer at full scale](https://arxiv.org/abs/2511.00224).
We will run a hybrid quantum-classical workflow using the [SBD](https://github.com/r-ccs-cms/sbd) solver to diagonalize a sparse chemistry Hamiltonian on the Miyabi-C environment, orchestrated via Prefect.

<img src="./images/img-closed-loop.png" alt="sbd" width="90%"/><br>

The goal is to compute the ground state energy of N2-Mo State.
## Prerequisites

Before starting, make sure:

- You have completed [Create Your QCSC Workflow with Prefect](./create_qcsc_workflow.md).

> [!IMPORTANT]
> - Replace `g00` and `z12345` with your actual group and account name.
> - There is currently an issue where the `~` (home) directory on mdx runs out of available space. As a workaround, we can use the `/large` directory instead. In this tutorial, we use the `/large` directory.

## 0. What changes from BitCounts?

### What you did in BitCounts (quick recap)
- **(HPC side)** Compile C++ and produce an executable (`get_counts`) — *build on the execution environment*.  
- **(Prefect side)** Register a Block type from Python code (e.g., `prefect block register -f ...`).  
- **(UI side)** Create a Block **instance** and save environment-specific parameters (queue, nodes, executable path, etc.).  
- **(Flow side)** A Flow loads Blocks and Variables and runs tasks (quantum → HPC → post-process).

### What SBD closed-loop adds
- HPC execution expands from a single `get_counts` binary to a full **SBD solver (`diag`) + closed-loop logic**.
- Additional Blocks are required (solver job block, MinIO/S3, etc.).
- **Deployment (Deploy)** becomes important so that participants can run from the Prefect UI reliably.

---
## 1. Big picture: Flow / Task / Block / Variable / Deployment

### 1.1 Minimal “where it runs” model
- **MDX workflow client (mdx-cli)**: where the Prefect process runs the Flow (the “runner/worker” side)
- **Miyabi-C**: where HPC binaries (e.g., `diag`) run via PBS/MPI
- **IBM Quantum**: where quantum sampling runs via Qiskit Runtime (if used by the workflow)

### 1.2 Mapping to Prefect core concepts

#### Flow (end-to-end experiment procedure)
The entire closed-loop SQD experiment is implemented as a single Flow, including iterations, branching, and convergence checks.

#### Tasks (individual steps)
- Execute quantum sampling (IBM Quantum / Qiskit Runtime)
- Subsampling / configuration recovery (Python on MDX)
- Davidson diagonalization (PBS job on Miyabi-C)
- Collect results and store artifacts (Prefect)

#### Blocks (reusable “configuration + credentials”)
- **Quantum**: IBM Quantum credentials / runtime configuration
- **HPC**: SBD solver job settings (queue, nodes, executable path, modules, etc.)
- **Storage**: MinIO credentials / S3 bucket configuration

#### Variables (runtime parameters)
- Quantum sampler options (shots, DD settings, etc.) and other run-time knobs are stored as Prefect Variables.

#### Deployment (how the Flow becomes runnable from the UI)
Deployment is the “launch entry” that tells Prefect:
- which Flow to run,
- under which deployment name,
- and how it should be executed by a serving process.

---
## 2. Tutorial steps
![SBD Setup Flow](./images/img-sbd-setup-flow.png)
### Step 1. SSH to the MDX workflow client (where the Flow is executed)

Connect to the MDX workflow client using SSH. This is where we will install the workflow.

<img src="./images/icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh -A z12345@qii-kawasaki-miyabi-cli.cspp.cc.u-tokyo.ac.jp
```

### Step 2. Install the workflow code (bring the Flow definition into your environment)
To run the workflow, you need to setup the following blocks.
<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
git clone git@github.com:ibm-quantum-collaboration/qii-miyabi-kawasaki.git
```

Activate the environment:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
source  ~/venv/prefect/bin/activate
```

Install the SBD submodule:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
uv pip install -e ./qii-miyabi-kawasaki/algorithms/sbd
```

Check that the packages are installed correctly:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
uv pip list | grep qii-
```

You should see output like:

```text
prefect-miyabi                     0.1.0       /large/z12345/qii-miyabi-kawasaki/framework/prefect-miyabi
prefect-sbd                        0.1.0       /large/z12345/qii-miyabi-kawasaki/framework/prefect-sbd
qcsc-workflow-utility              0.1.0       /large/z12345/qii-miyabi-kawasaki/algorithms/qcsc_workflow_utility
sbd                                0.1.0       /large/z12345/qii-miyabi-kawasaki/algorithms/sbd
```

---

### Step 3. Build the SBD solver on Miyabi-C (prepare the HPC executable)
#### 3.1 Copy the build directory into `/work`
Copy the DICE build script to your shared volume on Miyabi-C:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
cp -r ./qii-miyabi-kawasaki/framework/prefect-sbd/sbd /work/gz00/z12345
```

#### 3.2 SSH to Miyabi-C and build inside an interactive job
> [!NOTE]
> The operating system and computer architecture of the MDX workflow server differs from that of the Miyabi-C compute nodes.
> To run programs written in compiled languages such as C++, it's important to build them directly on the environment where they will be executed.

Open a new terminal and connect to the Miyabi-C login node:

<img src="./images/icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh -A z12345@miyabi-c.jcahpc.jp
```

You will be prompted to enter a verification code. Open your authenticator app and input the OTP.

Request an interactive job to avoid overloading the login node:

<img src="./images/icon-miyabi.png" alt="miyabi" width="50"/><br>
```bash
qsub -I -W group_list=gz00 -q interact-c -l walltime=01:00:00
```

Once the job starts, the current shell will switch to the compute node.

Navigate to the directory where we copied the build script:

<img src="./images/icon-miyabi.png" alt="miyabi" width="50"/><br>
```bash
cd /work/gz00/z12345/sbd
sh ./build_sbd.sh
```

This process may take several minutes. After completion, a directory named `diag` in the `sbd` directory:

<img src="./images/icon-miyabi.png" alt="miyabi" width="50"/><br>
```bash
ls -l | grep diag
```

Example output:

```text
-rwxr-x--- 1 z12345 gz00 1542016 Nov 30 15:18 diag
```

Great! You have completed building SBD on Miyabi-C!

#### 3.3 Record the absolute path of `diag` (you will paste it into the solver Block)
Get the absolute path to the SBD executable:

<img src="./images/icon-miyabi.png" alt="miyabi" width="50"/><br>
```bash
realpath ./diag
```

Example output:

```text
/work/gz00/z12345/sbd/diag
```

We will need this path in the next step.

Once we completed this step, we can escape from the session (e.g. press `<ctrl> + d`) and log out from the Miyabi login node.  
Go back to the SSH shell of the MDX workflow server to proceed with the following steps.

---

### Step 4. Register integrations and create Blocks (Block Type vs Block Instance)

This mirrors BitCounts:
- BitCounts used `prefect block register -f ...` (register a Block Type from a file).
- Here we use `prefect block register -m prefect_sbd` (register from a Python module).

#### 4.1 Create a job working directory on `/work`

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
mkdir -p /work/gz09/z12345/observability_demo_jobs
cd /work/gz09/z12345/observability_demo_jobs
```

#### 4.2 Register the Block Type (schema)

Update your prefect token:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
prefect-auth login
```

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
prefect block register -m prefect_sbd
```

#### 4.3 Create the solver Block Instance
<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
prefect block create sbd-solver-job
```

This command will display a URL to the Prefect console.
Open it in your browser and fill in the following fields. You may leave any fields blank unless otherwise specified.
Fill in the UI :

| Field | Value / Example |
|---|---|
| Block Name | `davidson-solver`|
| Root Directory | `/work/gz09/z12345/observability_demo_jobs` <br>  The absolute path to the work directory we created above. |
| Executable | `/work/gz00/z12345/sbd/diag` <br>  The absolute path to the `diag` executable. |
| Executor | `pbs`|
| Launcher | `mpiexec.hydra` |
|️ Queue Name | `regular-c`|
| Project | `gz00` (Your Miyabi project name.)|
| Num Nodes | `1` |
| Num MPI Processes | `48` |
| MPI Options | `["-np","48"]` |
| Max Walltime | `02:00:00`|
| Task Comm Size | `1` |
| Aded Comm Size | `1` |
| Bdet Comm Size | `1` |
| Block | `10` |
| Iteration | `2` |
| Tolerance | `0.0001` |
| Carryover Ratio | `0.5`|
| Solver Mode | `cpu`|


> [!NOTE]
> Adding more processes yields shorter wall-clock time, but it demands more memory space to copy state vectors.
> With too many processes, the job will be killed by out-of-memory (OOM-killed).

![Setup SBD Solver Job](./images/img-sbd-block.png)

#### 4.4 Create MinIO / S3 related Blocks (for telemetry and outputs)
Create the required MinIO credentials and bucket Blocks as specified by your environment.

We store auxiliary data such as sampled and carryover bitstrings on MinIO, an S3 compatible object storage service. For this purpose, we host a dedicated S3 service on the MDX platform.

To use this storage, set up Prefect S3 integration. Start by setting up your MinIO credentials:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
prefect block create minio-credentials
```

This command will display a URL to the Prefect console.
Open it in your browser and fill in the following fields. You may leave any fields blank unless otherwise specified.

| Field | Value / Example |
|---|---|
| Block Name | `prefect-minio-cred` |
| Minio Root User | `z12345` <br> Your Miyabi ID|
| Minio Root Password | `********` <br> Please ask the admin about the password.|
| Use SSL | `true` |
| Endpoint URL | `https://qii-kawasaki-miyabi-serv.cspp.cc.u-tokyo.ac.jp` |


![Setup MinIO cred](./images/img-minio-cred-block.png)

Next, configure the S3 bucket where binary objects will be stored:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
prefect block create s3-bucket
```

This command will display a URL to the Prefect console.
Open it in your browser and fill in the following fields. You may leave any fields blank unless otherwise specified.

| Field | Value / Example |
|---|---|
| Block Name | `s3-sqd`|
| Bucket Name | `bucket-z12345` <br> Please change z12345 to your ID|
| Credentials | `prefect-minio-cred` <br> the MinIO credential block you created in the previous step |

![Setup S3 bucket](./images/img-s3-bucket-block.png)


### Step 5. Set Variables (runtime options)

This is the same pattern as BitCounts `prefect variable set ...` but with SBD/SQD options.
Provide sampler execution options: `sqd_options`.

Example:
<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
prefect variable set sqd_options '{"params": {"shots": 500000, "options": {"dynamical_decoupling": {"enable": true, "sequence_type": "XY4", "skip_reset_qubits": true}}}}' --overwrite
```

### Step 6. Deploy SBD Work Flow

**Deploy = register a Flow as a runnable entry point (Deployment) so it can be started from the Prefect UI/CLI by name.**

Having Python code on disk is not enough for stable UI-driven execution. Deployment records “what to run” and “how to run it”.

Start a screen session:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
screen -S sbd-workflow
```

To avoid using a small filesystem on MDX, set the following environment variable and Prefect configuration:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
source ~/venv/prefect/bin/activate
mkdir -p /large/z12345/tmp/ray
export RAY_TMPDIR="/large/z12345/tmp/ray"
```

> [!NOTE]
> By setting this environment variable, we intend to change the temporary directory used by Ray from `/tmp` to `/large/z12345/tmp/` in order to aboid overuse `/tmp`. However, it has been reported that this configuration is not taking effect, and Ray continues to use `/tmp` instead. 
> 
> At the moment, `/tmp` still has sufficient storage, so this is not causing any immediate issues. However, we need to identify a suitable workaround to prepare for a situation where `/tmp` may run out of space.

Deploy the workflow:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
sbd-deploy
```

You should see:

```text
Your flow 'riken-sqd-de' is being served and polling for scheduled runs!

To trigger a run for this flow, use the following command:

        $ prefect deployment run 'riken-sqd-de/riken_sqd_de'

You can also run your flow via the Prefect UI: ...
```

Detach the screen session (`<ctrl> + a`, then `d`).

### Step 7. Provide Workflow Parameters

In the Prefect console, click **Run** → **Custom run** and set as following. Any fields not explicitly specified may remain at their default values.

| Field | Value / Example |
|---|---|
| FCIDump File | `/large/z1234/qii-miyabi-kawasaki/algorithms/sbd/data/fcidump_N2_MO.txt` | 
| SQD Subspace Dimension (Optional) | `10000000` |
| Sabre Layout Trials (Optional) | `40000` |
| Differential Evolution Iterations (Optional)| `2` |

> [!NOTE]
> For this tutorial, the number of iterations is set to 2, but feel free to increase it as needed.

![Setup SBD Workflow Parameters](./images/img-sbd-workflow-paramaters-small.png)

### Step 8. Execute the Workflow

Click **Start Now** → **Submit**.

![SBD Flow Run](./images/img-sbd-flow-run.png)

After the run completes, check the `sqd-telemetry` artifact. It should contain 8 (= 4 walker $\times$ 2 iter) rows of intermediate energies. The final energy should converge around -326.64 Hartree.

![SQD Telemetry](./images/img-sqd-telemetry.png)

### Step 9. Cleanup
Follow [How to shutdown the workflow](../howto/howto_shutdown_workflow.md).

---

## 3. What happens when you “Submit” from the Prefect UI?

### 3.1 What the UI actually creates
When you choose **Run → Custom run → Submit**, the Prefect server creates a **Flow Run request** for a specific Deployment, with the parameters you provided.

### 3.2 What actually executes the Flow (the key mental model)
The process started by `sbd-deploy` is the **serving process**. It:
1. polls the Prefect server for new Flow Runs,
2. when it finds one, it executes the Flow **on mdx-cli**.

> If the `screen` session dies (or `sbd-deploy` stops), the UI can still create Flow Runs, but there is no active “runner” to pick them up — so the run will not progress.

### 3.3 Confirm the information of the deployment 
Instead of memorizing filenames, teach participants to trace the entry point:

1) List deployments  
```bash
prefect deployment ls
```

2) Inspect the deployment (shows the entry point / flow reference)  
```bash
prefect deployment inspect 'riken-sqd-de/riken_sqd_de'
```

3) Locate the Flow definition in the repo  
https://github.com/ibm-quantum-collaboration/qii-miyabi-kawasaki/blob/main/algorithms/sbd/sbd/main.py#L73


---

## Appendix: Using GPU Solvers and Custom Solver Blocks

This section explains how to:
1. Create a GPU-enabled SBD solver Block
2. Select which Solver to use at run time
3. Add custom Solver configurations without modifying the workflow code

This is an advanced usage of the SBD closed-loop workflow.

### B. Creating a GPU Solver Block
You already created a CPU solver Block:
```
davidson-solver
```
Now, we will create a GPU-enabled version.

#### B.1 Create a new Block Instance 

Run:

```
prefect block create sbd-solver-job
```

Open the displayed URL and fill in the fields.
| Field | Value / Example |
|---|---|
| Block Name | `davidson-solver-gpu`|
| Root Directory | `/work/gz09/z12345/observability_demo_jobs` <br>  The absolute path to the work directory we created above. |
| Executable | `/work/gz00/z12345/sbd/diag` <br>  The absolute path to the `GPU version's diag` executable. |
| Executor | `pbs`|
| Launcher | `mpirun` |
|️ Queue Name | `regular-g`|
| Project | `gz00` (Your Miyabi project name.)|
| Num Nodes | `1` |
| Num MPI Processes | `1` |
| MPI Options | `["-n","1"]` |
| Max Walltime | `02:00:00`|
| Task Comm Size | `1` |
| Aded Comm Size | `1` |
| Bdet Comm Size | `1` |
| Block | `10` |
| Iteration | `2` |
| Tolerance | `0.0001` |
| Carryover Ratio | `0.5`|
| Solver Mode | `gpu`|


## C. How the Workflow Selects a Solver

The workflow does not hard-code Solver names.
Instead, it accepts a Block reference at run time.

The format is:

```
<block_type_slug>/<block_name>
```

For example:

```
sbd_solver_job/davidson-solver
sbd_solver_job/davidson-solver-gpu
```

## D. Selecting the Solver at Run Time

When launching a run from the Prefect UI:
1. Click Run → Custom run
2. Set the parameter:

| Field	| Example |  
|---|---|
| Solver Block Ref | sbd_solver_job/davidson-solver-gpu |

This tells the workflow to use the GPU solver Block.

----
*END OF TUTORIAL*