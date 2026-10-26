# Workflow for observability demo on Miyabi
#
# Author: Naoki Kanazawa (knzwnao@jp.ibm.com)

from prefect import task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import RUN_ID
from prefect.logging import get_run_logger
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout, Target

from .transpile_core import (
    _find_optimal_layout,
    _transpile_circuit,
)


@task
def transpile_circuit(
    circuit: QuantumCircuit,
    target: Target,
    layout: Layout,
    optimization_level: int,
) -> QuantumCircuit:
    return _transpile_circuit(
        circuit=circuit,
        target=target,
        layout=layout,
        optimization_level=optimization_level,
    )


# Cache only within a flow run without explicit filesystem locking.
cache_policy = RUN_ID


@task(
    cache_policy=cache_policy,
)
def find_optimal_layout(
    test_circuit: QuantumCircuit,
    target: Target,
    optimization_level: int,
    max_iterations: int,
    swap_trials: int,
    layout_trials: int,
) -> Layout:
    logger = get_run_logger()

    layout, layout_info = _find_optimal_layout(
        test_circuit=test_circuit,
        target=target,
        optimization_level=optimization_level,
        max_iterations=max_iterations,
        swap_trials=swap_trials,
        layout_trials=layout_trials,
        logger=logger,
    )

    create_table_artifact(
        table=layout_info,
        key="isa-qubit-properties",
    )

    return layout
