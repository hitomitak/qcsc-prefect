"""Pinned PR #42 functions and experimental QPY transport for the SQD PoC.

Requires a source checkout, including the immutable tests/fixtures/pr42 snapshot.
No GitHub access, PR checkout, or Prefect server is used by this loader.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import ffsim
import numpy as np
from ffsim.qiskit import PRE_INIT
from pydantic import BaseModel, Field
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import Target, generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGatesDecomposition,
    RemoveIdentityEquivalent,
    SabreLayout,
)
from qiskit.transpiler.passmanager import PassManager
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "algorithms/qcsc_workflow_utility/src"))

from qcsc_workflow_utility import ElectronicProperties, chem  # noqa: E402, F401

FIXTURE_DIR = ROOT / "packages/qcsc-prefect-qiskit/tests/fixtures/pr42"
LOGGER = logging.getLogger("qrmi-sqd-pr42")


def source_metadata() -> dict:
    return json.loads((FIXTURE_DIR / "source.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_pr42() -> ModuleType:
    """Load the original three definitions, replacing only @task and logging."""
    provenance = source_metadata()
    source_path = FIXTURE_DIR / "sqd_fulqrum_main.py.txt"
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != provenance["sha256"]:
        raise RuntimeError(f"PR #42 source snapshot hash mismatch: {source_path}")
    tree = ast.parse(source, filename=str(source_path))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef)
        and node.name in provenance["definitions"]
    ]
    if [node.name for node in definitions] != provenance["definitions"]:
        raise RuntimeError("PR #42 source snapshot definitions do not match source.json")
    module = ModuleType("_qrmi_sqd_pr42_offline")
    module.__dict__.update(
        BaseModel=BaseModel,
        Field=Field,
        QuantumCircuit=QuantumCircuit,
        QuantumRegister=QuantumRegister,
        ClassicalRegister=ClassicalRegister,
        ElectronicProperties=ElectronicProperties,
        ffsim=ffsim,
        np=np,
        PRE_INIT=PRE_INIT,
        Target=Target,
        generate_preset_pass_manager=generate_preset_pass_manager,
        PassManager=PassManager,
        ConditionalController=ConditionalController,
        ApplyLayout=ApplyLayout,
        BarrierBeforeFinalMeasurements=BarrierBeforeFinalMeasurements,
        EnlargeWithAncilla=EnlargeWithAncilla,
        FullAncillaAllocation=FullAncillaAllocation,
        Optimize1qGatesDecomposition=Optimize1qGatesDecomposition,
        RemoveIdentityEquivalent=RemoveIdentityEquivalent,
        SabreLayout=SabreLayout,
        FoldRzzAngle=FoldRzzAngle,
        task=lambda fn: fn,
        get_run_logger=lambda: LOGGER,
    )
    sys.modules[module.__name__] = module
    try:
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec"),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def prepare_qpy_transport(circuit):
    """Optimize while ffsim Gate attributes exist, then expand their definitions."""
    circuit = PRE_INIT.run(circuit)
    for _ in range(10):
        gates = sorted(
            {
                instruction.operation.name
                for instruction in circuit.data
                if (type(instruction.operation).__module__ or "").startswith("ffsim.")
            }
        )
        if not gates:
            return circuit
        circuit = circuit.decompose(gates_to_decompose=gates)
    raise RuntimeError("ffsim gates remain after ten decomposition rounds")


def transpile_for_target(circuit, target, config: dict):
    """Use the same pinned PR and parameters on both sides of the QPY handoff."""
    provenance = source_metadata()
    if config["source_commit"] != provenance["commit"]:
        raise ValueError("Sender/compute-node PR #42 source commit mismatch")
    if config["source_sha256"] != provenance["sha256"]:
        raise ValueError("Sender/compute-node PR #42 source hash mismatch")
    pr42 = load_pr42()
    params = pr42.CircuitParameters(**config["circuit_parameters"])
    isa = pr42.transpile_circuit(circuit, target, params, np.random.default_rng(config["seed"]))
    for instruction in isa.data:
        if instruction.operation.name == "barrier":
            continue
        qargs = tuple(isa.find_bit(bit).index for bit in instruction.qubits)
        if not target.instruction_supported(
            operation_name=instruction.operation.name,
            qargs=qargs,
            parameters=instruction.operation.params,
        ):
            raise RuntimeError(f"Unsupported ISA instruction: {instruction.operation.name} {qargs}")
    return isa
