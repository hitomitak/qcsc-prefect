"""Compute-node entrypoint: QPY -> transpile -> QRMI task_runner -> raw result.

Run inside the SPANK-enabled Slurm allocation with the existing QRMI venv.
Only the task_runner subprocess submits the quantum workload.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from qiskit import qasm3, qpy
from qiskit.primitives import PrimitiveResult
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

try:
    from qiskit_ibm_runtime.decoders.result_decoder import ResultDecoder
except ModuleNotFoundError:
    from qiskit_ibm_runtime.utils.result_decoder import ResultDecoder

RESOURCE = "test_heron"


def require_batch_environment() -> None:
    # SPANK must acquire a Batch BEFORE this script starts. Exporting the mode
    # here cannot change an already acquired dedicated session into a Batch.
    mode = os.environ.get(f"{RESOURCE}_QRMI_IBM_QCS_SESSION_MODE", "")
    if mode.lower() != "batch":
        raise RuntimeError(
            "Configure test_heron environment in /etc/slurm/qrmi_config.json: "
            '"QRMI_IBM_QCS_SESSION_MODE": "batch" (on every compute node)'
        )
    if not os.environ.get(f"{RESOURCE}_QRMI_JOB_ACQUISITION_TOKEN"):
        raise RuntimeError("Missing SPANK acquisition token; submit with --qpu=test_heron")


def make_sampler_payload(isa_circuit, shots: int) -> dict:
    return {
        "program_id": "sampler",
        "parameters": {
            "pubs": [
                [qasm3.dumps(isa_circuit, disable_constants=True, allow_aliasing=True), None, shots]
            ],
            "version": 2,
            # ResultDecoder expects the encoded Qiskit containers, as in
            # QRMIBaseSamplerV2.run(), rather than the plain REST JSON format.
            "support_qiskit": True,
            "shots": shots,
            "options": {},
        },
    }


def execute_task_runner(isa_circuit, shots: int, executable: Path) -> None:
    Path("sampler_input.json").write_text(
        json.dumps(make_sampler_payload(isa_circuit, shots), indent=2), encoding="utf-8"
    )
    subprocess.run([str(executable), RESOURCE, "sampler_input.json", "result.json"], check=True)
    # Some task_runner versions exit zero even when the quantum task fails.
    # A decodable result with the requested shots is required for Slurm success.
    result_path = Path("result.json").resolve()
    result = ResultDecoder.decode(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, PrimitiveResult):
        raise RuntimeError(
            f"task_runner returned {type(result).__name__}; expected PrimitiveResult "
            f"(requires support_qiskit=true); see {result_path}"
        )
    if len(result) != 1:
        raise RuntimeError(
            f"task_runner returned {len(result)} PUBs; expected 1; see {result_path}"
        )
    actual_shots = result[0].data.meas.num_shots
    if actual_shots != shots:
        raise RuntimeError(
            f"task_runner returned {actual_shots} shots; expected {shots}; see {result_path}"
        )
    print(f">>> Counts ({shots} shots): {result[0].data.meas.get_counts()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100)
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    require_batch_environment()

    # Import only on the compute node; QRMI reads SPANK's environment here.
    from qrmi import QRMIService
    from qrmi.primitives.ibm import get_target

    resource = next((r for r in QRMIService().resources() if r.resource_id() == RESOURCE), None)
    if resource is None:
        raise RuntimeError("test_heron is not present in this Slurm allocation")
    with Path("circuit.qpy").open("rb") as stream:
        (circuit,) = qpy.load(stream)
    pm = generate_preset_pass_manager(optimization_level=1, target=get_target(resource))
    isa = pm.run(circuit)
    print(f">>> Host: {socket.gethostname()}, resource: {RESOURCE}, mode: batch", flush=True)
    print(f">>> Circuit ops (ISA): {dict(isa.count_ops())}", flush=True)
    execute_task_runner(isa, args.shots, Path(sys.executable).with_name("task_runner"))


if __name__ == "__main__":
    main()
