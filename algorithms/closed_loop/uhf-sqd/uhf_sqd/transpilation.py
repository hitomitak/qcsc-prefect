"""QRMI compile hooks of the UHF-SQD workflow (device path).

``QrmiRuntime`` runs ``prepare`` on the workflow host and calls ``transpile(circuit, target,
config)`` with the actual Target: in the workflow process for ``scheduler="iqp"``, and on the
compute node for ``scheduler="slurm"``, where this package must be installed.

The compilation is the source's default SABRE policy (kunalroy332/qcsc-prefect at 3968e58,
``algorithms/sbd/sbd/transpile_custom.py``: ``find_optimal_layout`` and ``transpile_circuit``,
with the same seed, passes and post-optimization). Two things differ from the source, in the
same way as the closed-loop ``sqd_de`` hook:

- the ffsim ``PRE_INIT`` stage runs in ``prepare``, because QPY transport to a compute node turns
  the ffsim gate objects into plain gates;
- the SABRE layout is searched for each circuit. The source searched it once per flow run (a
  Prefect task with ``cache_policy=RUN_ID``) and reused it for every walker and trial.

The source's author accepted the per-circuit search. The trade-offs, from that discussion:

- The LUCJ circuits of different walkers differ only in rotation angles, not in two-qubit
  connectivity, and SABRE is deterministic with ``TRANSPILE_SEED``. Most searches therefore
  return the same layout, and each costs ``sabre_layout_trials`` of compile time for no gain.
  The exceptions are circuits whose angles let optimization cancel gates and change the
  interaction pattern.
- When the layouts do differ, walkers run on different physical qubits. Walker-to-walker energy
  differences then include qubit quality as well as the parameters, which can bias the DE
  selection.

The source's author used the ffsim error-aware LUCJ mapper in practice. That layout is deferred,
and its effect, and that of per-circuit layouts, should be measured on hardware.
"""

from __future__ import annotations

from ffsim.qiskit import PRE_INIT
from qcsc_prefect.integrations.qiskit.qrmi import PreparedCircuit
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

TRANSPILER = "uhf_sqd.transpilation:transpile"
TRANSPILE_SEED = 6538
OPTIONS = ("optimization_level", "sabre_max_iterations", "sabre_swap_trials")
OPTIONS += ("sabre_layout_trials",)


def transpile_config(circuit_params) -> dict:
    """JSON-serializable ``transpile`` configuration from ``CircuitParameters``."""
    config = {name: getattr(circuit_params, name) for name in OPTIONS}
    _validate(config)
    return config


def _validate(config: dict) -> None:
    if set(config) != set(OPTIONS):
        raise ValueError(f"UHF-SQD transpile config must have exactly the keys {list(OPTIONS)}")
    level = config["optimization_level"]
    if type(level) is not int or not 0 <= level <= 3:
        raise ValueError("optimization_level must be an integer from 0 to 3")
    for name in OPTIONS[1:]:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")


def prepare(circuit) -> PreparedCircuit:
    """QrmiRuntime ``prepare`` hook: run the ffsim pre-init stage on the workflow host."""
    return PreparedCircuit(PRE_INIT.run(circuit))


def transpile(circuit, target, config):
    """QrmiRuntime transpiler hook: SABRE layout search, then transpile with that layout."""
    _validate(config)
    level = config["optimization_level"]
    layout = _find_optimal_layout(
        test_circuit=circuit,
        target=target,
        optimization_level=level,
        max_iterations=config["sabre_max_iterations"],
        swap_trials=config["sabre_swap_trials"],
        layout_trials=config["sabre_layout_trials"],
    )
    return _transpile_circuit(
        circuit=circuit, target=target, layout=layout, optimization_level=level
    )


def _post_optimization(target: Target) -> PassManager:
    return PassManager(
        [
            FoldRzzAngle(),
            Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
            RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
        ]
    )


def _transpile_circuit(
    circuit: QuantumCircuit,
    target: Target,
    layout: Layout,
    optimization_level: int,
) -> QuantumCircuit:
    custom_pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        seed_transpiler=TRANSPILE_SEED,
        target=target,
    )
    custom_pm.layout = PassManager(
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
        custom_pm.post_optimization = _post_optimization(target)
    return custom_pm.run(circuit)


def _find_optimal_layout(
    test_circuit: QuantumCircuit,
    target: Target,
    optimization_level: int,
    max_iterations: int,
    swap_trials: int,
    layout_trials: int,
) -> Layout:
    coupling_map = target.build_coupling_map()
    test_pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        seed_transpiler=TRANSPILE_SEED,
        target=target,
    )
    if "rzz" in target.operation_names:
        test_pm.post_optimization = _post_optimization(target)
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
    return isa_trial.layout.initial_virtual_layout(filter_ancillas=True)
