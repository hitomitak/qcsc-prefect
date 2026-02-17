# Workflow for observability demo on Miyabi
#
# Author: Naoki Kanazawa (knzwnao@jp.ibm.com)

from ffsim.qiskit import PRE_INIT
from prefect import task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import RUN_ID
from prefect.locking.filesystem import FileSystemLockManager
from prefect.logging import get_run_logger
from prefect.settings import PREFECT_HOME
from prefect.transactions import IsolationLevel
from qiskit import QuantumCircuit
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import Layout, Target, generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGatesDecomposition,
    RemoveIdentityEquivalent,
    SabreLayout,
    SetLayout,
)
from qiskit.transpiler.passmanager import PassManager
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

TRANSPILE_SEED = 6538


@task
def transpile_circuit(
    circuit: QuantumCircuit,
    target: Target,
    layout: Layout,
    optimization_level: int,
) -> QuantumCircuit:
    cusotm_pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        seed_transpiler=TRANSPILE_SEED,
        target=target,
    )
    cusotm_pm.pre_init = PRE_INIT
    cusotm_pm.layout = PassManager(
        [
            BarrierBeforeFinalMeasurements(
                label="qiskit.transpiler.internal.routing.protection.barrier",
            ),
            SetLayout(
                layout=layout,
            ),
            FullAncillaAllocation(coupling_map=target.build_coupling_map()),
            EnlargeWithAncilla(),
            ApplyLayout(),
        ]
    )
    if "rzz" in target.operation_names:
        cusotm_pm.post_optimization = PassManager(
            [
                FoldRzzAngle(),
                Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
                RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
            ]
        )
    return cusotm_pm.run(circuit)


# Run single task at a time to prevent race condition.
cache_policy = RUN_ID.configure(
    isolation_level=IsolationLevel.SERIALIZABLE,
    lock_manager=FileSystemLockManager(
        lock_files_directory=PREFECT_HOME.value() / "locks"
    )
)


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
    coupling_map = target.build_coupling_map()

    test_pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        seed_transpiler=TRANSPILE_SEED,
        target=target,
    )
    test_pm.pre_init = PRE_INIT
    if "rzz" in target.operation_names:
        test_pm.post_optimization = PassManager(
            [
                FoldRzzAngle(),
                Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
                RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
            ]
        )
    test_pm.layout = PassManager(
        [
            BarrierBeforeFinalMeasurements(
                label="qiskit.transpiler.internal.routing.protection.barrier",
            ),
            SabreLayout(
                coupling_map=coupling_map,
                seed=TRANSPILE_SEED,
                max_iterations=max_iterations,
                layout_trials=layout_trials,
                swap_trials=swap_trials,
            ),
            ConditionalController(
                tasks=[
                    FullAncillaAllocation(coupling_map=coupling_map),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                condition=lambda propset: propset["final_layout"] is None,
            ),
        ]
    )
    isa_trial = test_pm.run(test_circuit)
    depth = isa_trial.depth(lambda inst: inst.operation.name not in ("rz", "barrier", "measure"))
    logger.info(f"Circuit depth = {depth}\nInstruction counts = {dict(isa_trial.count_ops())}")
    final_sabre_layout = isa_trial.layout.initial_virtual_layout(filter_ancillas=True)

    layout_info = []
    for vi, pi in final_sabre_layout.get_virtual_bits().items():
        qubit = {
            "v_index": test_circuit.qubits.index(vi),
            "p_index": pi,
            "t1": None,
            "t2": None,
        }
        try:
            qubit_prop = target.qubit_properties[pi]
            qubit["t1"] = qubit_prop.t1 * 1e6
            qubit["t2"] = qubit_prop.t2 * 1e6
        except (IndexError, TypeError):
            pass
        layout_info.append(qubit)

    create_table_artifact(
        table=layout_info,
        key="isa-qubit-properties",
    )
    
    return final_sabre_layout
