"""Small QRMI boundary for an otherwise preserved PR #42 workflow copy."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime

from .pinned_recovery import load_recovery
from .pr42 import prepare_sampler_circuit

# Import aliases, not replacement implementations of the calculation loop.
_original_recovery = load_recovery()
SolveFn = _original_recovery.SolveFn
run_configuration_recovery = _original_recovery.run_configuration_recovery


async def load_qrmi_runtime(config_path: str) -> QrmiRuntime:
    """The original runner_name argument now identifies a saved runtime JSON file."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return QrmiRuntime(**config)


async def sample_with_qrmi(runtime, circuit, circuit_params, options):
    """Prepare QPY on the sender; acquire Target/transpile inside the allocation."""
    sampler = replace(
        runtime,
        prepare=partial(prepare_sampler_circuit, circuit_parameters=circuit_params.model_dump()),
        transpiler="examples.qrmi_sqd_validation.pr42:transpile_for_target",
    )
    result = await sampler.sampler(sampler_pubs=[(circuit,)], options=options)
    # Record the transport boundary; original sample_bitstrings still selects meas/test.
    record = {
        "job": result.metadata["qcsc_slurm"],
        "options": options,
        "circuit_parameters": circuit_params.model_dump(),
    }
    path = Path(runtime.root_dir).resolve().parent / "quantum_stage.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return result
