"""One H2 flow: fresh QRMI samples, pinned recovery, Slurm Fulqrum and FCI checks."""

from __future__ import annotations

import argparse
import asyncio
import json
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
from prefect import flow
from prefect.logging import get_run_logger
from pyscf import fci, lib
from qcsc_prefect.integrations.qiskit.qrmi.contract import positive_shots

from .integration import sample_h2_run
from .pinned_recovery import verify_sources
from .pr42 import source_metadata
from .recovery import (
    PIN,
    RecoverySettings,
    load_block,
    load_fulqrum_solver,
    load_saved_h2_run,
    read_json,
    recover_saved_h2,
    require,
    validate_result,
)

ENERGY_TOLERANCE = 1e-8


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def stage_directory(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path != root, "Stage path must stay inside the run")
    return path


def verify_h2_run(directory: str | Path) -> dict:
    """Read saved evidence and compare FCI, without Prefect API or job submission.

    This does not require a separately prepared sampling-input.json or sacct.
    The recovery's own pre-solve input hashes are the source of comparison.
    """
    from qcsc_prefect_fulqrum.io_utils import read_solver_output
    from qcsc_prefect_fulqrum.solver_job import solution_to_sci_result

    root = Path(directory).expanduser().resolve()
    request = read_json(root / "integration_request.json")
    require(request["source"]["commit"] == PIN, "Wrong integration source")
    require(request["qiskit_integration_version"] == "0.3.0", "Expected library 0.3.0")
    verify_sources()
    stages = read_json(root / "stages.json")
    sampling_dir = stage_directory(root, stages["sampling_run"])
    recovery_dir = stage_directory(root, stages["recovery_run"])
    saved = load_saved_h2_run(sampling_dir)
    require(saved.requested_shots == request["shots"], "Requested shots changed")
    require(saved.reset_mitigation is request["reset_mitigation"], "Reset setting changed")
    require(not (recovery_dir / "failure.json").exists(), "Recovery has a failure record")
    recovered = read_json(recovery_dir / "recovery_summary.json")
    inputs = read_json(recovery_dir / "input_validation.json")
    original_sampling = Path(request["work_dir"]) / stages["sampling_run"]
    for record in [inputs, recovered]:
        require(Path(record["sampling_run"]) == original_sampling, "Recovery used another input")
        require(record["backend"] == "fulqrum-block", "Expected actual Fulqrum block backend")
        require(record["requested_shots"] == saved.requested_shots, "Recovery request mismatch")
        require(record["retained_shots"] == saved.bits.num_shots, "Retained shots mismatch")
    require(inputs["input_sha256"] == saved.hashes, "Quantum evidence changed after sampling")
    require(recovered["status"] == "completed", "Recovery did not complete")
    require(recovered["source_commit"] == PIN, "Wrong recovery source")
    require(recovered["qiskit_integration_version"] == "0.3.0", "Wrong recovery library")
    require(recovered["solver_block"] == request["solver_block"], "Wrong solver block")
    expected_settings = {
        "subspace_dim": 4,
        "num_batches": 1,
        "max_iterations": 1,
        "seed": request["seed"],
    }
    require(inputs["settings"] == recovered["settings"] == expected_settings, "Settings changed")
    require(recovered["solver_calls"] == 1 and len(recovered["records"]) == 1, "Expected one solve")
    require(
        recovered["records"][0]["dimension"] == 4, "H2 acceptance requires the full 4D subspace"
    )

    with np.load(recovery_dir / "integrals.npz", allow_pickle=False) as data:
        require(tuple(data["nelec"]) == (1, 1), "Expected H2 electron counts")
        nuclear = float(data["nuclear_repulsion_energy"])
        with lib.with_omp_threads(1):
            electronic_fci = fci.direct_spin1.kernel(
                data["one_body_tensor"], data["two_body_tensor"], 2, (1, 1)
            )[0]
    total = float(recovered["total_energy"])
    require(np.isfinite(total), "Nonfinite total energy")
    require(
        np.isclose(recovered["nuclear_repulsion_energy"], nuclear, atol=1e-12, rtol=0)
        and np.isclose(total, recovered["electronic_energy"] + nuclear, atol=1e-12, rtol=0),
        "Nuclear repulsion must be added exactly once",
    )
    fci_total = float(electronic_fci + nuclear)
    require(abs(total - fci_total) <= ENERGY_TOLERANCE, "Fulqrum does not match H2 FCI")

    jobs = []
    require(len(recovered["solver_work_dirs"]) == 1, "Expected one classical solver job")
    for name in recovered["solver_work_dirs"]:
        job_dir = Path(name)
        job = read_json(job_dir / "slurm_job.json")
        require(job["state"] == "COMPLETED" and job["exit_status"] == 0, "Solver job failed")
        require(bool(job["job_id"]), "Missing solver job ID")
        solution = read_solver_output(job_dir, 2)
        validate_result(solution_to_sci_result(solution, norb=2, nelec=(1, 1)), 2, (1, 1))
        require(bool(solution.metadata.get("fulqrum_version")), "Missing native Fulqrum version")
        require(solution.metadata.get("subspace_dim") == 4, "Wrong native solver subspace")
        require(
            np.isclose(solution.energy, recovered["electronic_energy"], atol=1e-12, rtol=0),
            "Native output and recovery energy differ",
        )
        script = (job_dir / request["solver_script_filename"]).read_text()
        require("--qpu" not in script, "Classical solver script contains a QPU directive")
        jobs.append({**job, "fulqrum_metadata": solution.metadata})
    (quantum_record,) = (sampling_dir / "quantum").glob("*/slurm_job.json")
    return {
        "status": "completed",
        "validation": "passed",
        "source_commit": PIN,
        "qiskit_integration_version": "0.3.0",
        "work_dir": str(root),
        "sampling_run": str(sampling_dir),
        "recovery_run": str(recovery_dir),
        "requested_shots": saved.requested_shots,
        "retained_shots": saved.bits.num_shots,
        "reset_mitigation": saved.reset_mitigation,
        "counts": saved.bits.get_counts(),
        "electronic_energy": float(recovered["electronic_energy"]),
        "nuclear_repulsion_energy": nuclear,
        "total_energy": total,
        "fci_total_energy": fci_total,
        "absolute_energy_error": abs(total - fci_total),
        "energy_tolerance": ENERGY_TOLERANCE,
        "quantum_input_files_unchanged": len(saved.hashes),
        "quantum_job": read_json(quantum_record),
        "solver_jobs": jobs,
    }


@flow(name="qrmi-pr42-h2-fulqrum", retries=0, persist_result=False)
async def h2_qrmi_fulqrum_flow(
    *,
    solver_block: str,
    shots: int = 100,
    work_dir: str = "/shared/qrmi-pr42-end-to-end",
    reset_mitigation: bool = False,
    seed: int = 24,
) -> dict:
    """Submit fresh quantum sampling, then classical Fulqrum, in one parent flow.

    A rerun creates a new quantum job. No stage has an automatic retry/fallback.
    The fixed 4D/one-batch/one-round settings define a small H2 acceptance check.
    """
    from qcsc_prefect_blocks.common.blocks import HPCProfileBlock
    from qcsc_prefect_executor.from_blocks import build_scheduler_script_filename

    shots = positive_shots(shots)
    require(type(reset_mitigation) is bool, "reset_mitigation must be a boolean")
    settings = RecoverySettings(seed=seed)
    require(bool(solver_block), "A saved classical solver block is required")
    require(version("qcsc-prefect-qiskit") == "0.3.0", "Expected runtime environment 0.3.0")
    verify_sources()
    root = Path(work_dir).expanduser().resolve() / f"run_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    get_run_logger().info("H2 QRMI/Fulqrum integration directory: %s", root)
    request = {
        "work_dir": str(root),
        "source": source_metadata(),
        "qiskit_integration_version": version("qcsc-prefect-qiskit"),
        "shots": shots,
        "reset_mitigation": reset_mitigation,
        "solver_block": solver_block,
        "seed": seed,
    }
    write_json(root / "integration_request.json", request)
    phase = "preflight"
    stages = {}
    try:
        solver, _ = await load_fulqrum_solver(solver_block)
        request["solver_script_filename"] = build_scheduler_script_filename(
            solver.script_filename, "slurm"
        )
        write_json(root / "integration_request.json", request)
        quantum_hpc = await load_block(HPCProfileBlock, "poc-qrmi-hpc")
        require(
            quantum_hpc.hpc_target == "slurm" and quantum_hpc.slurm_qpu == "test_heron",
            "Sampling requires the poc-qrmi-hpc Slurm/test_heron profile",
        )
        phase = "sampling"
        sampled = await sample_h2_run(
            shots=shots, work_dir=str(root / "sampling"), reset_mitigation=reset_mitigation
        )
        stages["sampling_run"] = str(sampled.run_dir.relative_to(root))
        write_json(root / "stages.json", stages)
        phase = "recovery"
        recovered = await recover_saved_h2(
            sampled.run_dir,
            root / "recovery",
            settings=settings,
            backend="fulqrum-block",
            solver_block=solver_block,
        )
        stages["recovery_run"] = str(Path(recovered["work_dir"]).relative_to(root))
        write_json(root / "stages.json", stages)
        phase = "verification"
        summary = verify_h2_run(root)
        write_json(root / "integration_summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            root / "failure.json",
            {"phase": phase, "error_type": type(exc).__name__, "error": str(exc), "stages": stages},
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Submit fresh QPU sampling and classical Slurm solver jobs"
    )
    run.add_argument("--solver-block", required=True)
    run.add_argument("--shots", type=int, default=100)
    run.add_argument("--work-dir", default="/shared/qrmi-pr42-end-to-end")
    run.add_argument("--reset-mitigation", action="store_true")
    run.add_argument("--seed", type=int, default=24)
    verify = commands.add_parser(
        "verify", help="Check a saved integration run without submitting jobs"
    )
    verify.add_argument("directory", type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = (
        verify_h2_run(args["directory"])
        if command == "verify"
        else asyncio.run(h2_qrmi_fulqrum_flow(**args))
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
