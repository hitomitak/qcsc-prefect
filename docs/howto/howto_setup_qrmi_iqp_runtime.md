# How to Set Up a QRMI Runtime Block for Direct IQP Execution

This guide creates the `QrmiRuntimeBlock` that the algorithms load for
`quantum_source = "real-device"` with `scheduler = "iqp"`. With this scheduler,
QRMI runs in the workflow process and submits to IBM Quantum (IQP) directly,
without Slurm. See the [QRMI Sampler guide](./howto_qrmi_slurm_sampler.md) for
the runtime itself and for `scheduler = "slurm"`.

## Prerequisites

- A host for the workflow process that can reach the IBM Quantum and IBM Cloud IAM
  endpoints. QRMI publishes wheels for Linux (x86_64, aarch64, ppc64le); on other
  platforms `pip` builds it from source, which needs a Rust toolchain.
- An IBM Cloud API key and the CRN of your IBM Quantum service instance, from the
  [IBM Quantum Platform](https://quantum.cloud.ibm.com/).
- A Prefect server or workspace selected in the current profile.

Install the Qiskit integration and QRMI in the workflow environment:

```bash
cd /path/to/qcsc-prefect
uv pip install -e packages/qcsc-prefect-qiskit "qrmi[ibm]"
```

## The `resource_config` secret

`resource_config` holds QRMI's IBM Quantum Compute Service settings. The keys are
the QRMI environment variable names **without** the `<resource>_` prefix:

| Key | Value |
| --- | --- |
| `QRMI_IBM_QCS_ENDPOINT` | `https://quantum.cloud.ibm.com/api/v1` |
| `QRMI_IBM_QCS_IAM_ENDPOINT` | `https://iam.cloud.ibm.com` |
| `QRMI_IBM_QCS_IAM_APIKEY` | your IBM Cloud API key |
| `QRMI_IBM_QCS_SERVICE_CRN` | your service instance CRN |

All values are strings. The runtime passes them to QRMI directly; it does not read
or write process environment variables, so flows in one worker can use different
credentials.

Do **not** set `QRMI_IBM_QCS_SESSION_ID` or `QRMI_JOB_ACQUISITION_TOKEN`.
`scheduler = "iqp"` submits jobs without a session, and the runtime rejects these
keys. Each `sampler()` call is one IQP job; jobs are not grouped in a session or
batch.

## Option 1: create the block with an algorithm's `create_blocks.py`

The algorithm block generators create the block from a `[runtime]` table and read
the API key from the environment, so the key is never written to a file.

The generators import the algorithm and its solver packages, which the command in
[Prerequisites](#prerequisites) does not install; with only that command,
`create_blocks.py` fails with `ModuleNotFoundError`. First complete the installation
in the algorithm's README (`algorithms/open_loop/sqd/README.md`,
`algorithms/open_loop/skqd/README.md` or `algorithms/closed_loop/sqd_de/README.md`),
in the same environment.

The generators also create the algorithm's solver and HPC blocks, so the config file
needs the algorithm's other settings too (for example `project`, `queue`, `work_dir`
and the solver executable). Start from the algorithm's example config and follow its
README for those settings:

```bash
cp algorithms/open_loop/sqd/sqd_blocks.example.toml algorithms/open_loop/sqd/sqd_blocks.toml
vim algorithms/open_loop/sqd/sqd_blocks.toml
```

The Miyabi examples already end with the `[runtime]` tables; in the Fugaku and Slurm
examples they are commented out. Set them as follows:

```toml
# at the end of algorithms/open_loop/sqd/sqd_blocks.toml
runtime_block_name = "qrmi-runtime"

[runtime]
scheduler = "iqp"
resource = "ibm_kingston"

[runtime.qrmi]
QRMI_IBM_QCS_ENDPOINT = "https://quantum.cloud.ibm.com/api/v1"
QRMI_IBM_QCS_IAM_ENDPOINT = "https://iam.cloud.ibm.com"
QRMI_IBM_QCS_SERVICE_CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/your-account:your-instance::"
```

```bash
export QRMI_IBM_QCS_IAM_APIKEY="<your API key>"
python algorithms/open_loop/sqd/create_blocks.py \
  --config algorithms/open_loop/sqd/sqd_blocks.toml --hpc-target miyabi
```

The same table works for `algorithms/open_loop/skqd` and
`algorithms/closed_loop/sqd_de` (with their own example configs and `--hpc-target`). If
you only want to create the runtime block and already have the other blocks, Option 2
is simpler. The closed loop always uses the block name
`qrmi-runtime`. Writing `QRMI_IBM_QCS_IAM_APIKEY` in the file, or setting
`transpiler`, `transpile_config` or `resource_config` in `[runtime]`, is an error:
the flows set the transpiler themselves.

## Option 2: create the block in Python

```python
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

QrmiRuntimeBlock(
    scheduler="iqp",
    resource="ibm_kingston",
    resource_config={
        "QRMI_IBM_QCS_ENDPOINT": "https://quantum.cloud.ibm.com/api/v1",
        "QRMI_IBM_QCS_IAM_ENDPOINT": "https://iam.cloud.ibm.com",
        "QRMI_IBM_QCS_IAM_APIKEY": "<your API key>",
        "QRMI_IBM_QCS_SERVICE_CRN": "<your service instance CRN>",
    },
).save("qrmi-runtime", overwrite=True)
```

The block stores `resource_config` as a secret: the Prefect UI and `repr` mask it.

## Check the block

```bash
prefect block ls | grep -i qrmi
```

```python
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

runtime = QrmiRuntimeBlock.load_runtime("qrmi-runtime")
print(runtime.scheduler, runtime.resource)  # iqp ibm_kingston; the secret is hidden
```

`load_runtime` validates the settings. To sample once end to end, run
[`examples/qrmi_iqp_bell`](https://github.com/qiskit-community/qcsc-prefect/tree/main/examples/qrmi_iqp_bell)
with this block.

## If a submission result is unknown

If submission raises, for example because of a network error, it is unknown whether
IQP created a job. The runtime raises `IqpSubmissionUnknown` and does **not**
resubmit. The error and the log contain the resource and the submission time.
Check the IBM Quantum dashboard for a job on that resource at that time before you
run the flow again; otherwise the same work may run twice.
