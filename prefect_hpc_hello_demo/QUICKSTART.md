# Prefect Miyabi Demo (HPC-agnostic) — MPI Hello + qsub

This demo runs a tiny MPI program (`hello_mpi`) via Prefect Blocks and submits it with `qsub`.

## 0) Compile the MPI hello program
```bash
cd src
make
cp hello_mpi ..
cd ..
ls -l hello_mpi
```

## 1) Register Prefect block types
```bash
prefect block register -m prefect_hpc_demo.blocks
```

## 2) Create demo Blocks (admin)
```bash
python admin_create_blocks.py
```

If Miyabi requires modules to get MPI runtime, edit:
- `ExecutionProfileBlock.modules`
- `MiyabiHPCProfileBlock.module_init`

## 3) Run the flow (generates script + qsub)
```bash
python -c "from prefect_hpc_demo.flow_demo import miyabi_mpi_hello_flow; print(miyabi_mpi_hello_flow(submit=True))"
```

## 4) Tune parallelism without changing workflow code
```python
from prefect_hpc_demo.flow_demo import miyabi_mpi_hello_flow
from prefect_hpc_demo.models import Tuning

miyabi_mpi_hello_flow(
    exec_profile_block_name="exec-hello-n2",
    tuning=Tuning(nodes=4, ranks_per_node=8),
    submit=True,
)
```
