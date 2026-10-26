"""Run the minimally adapted PR #42 flow with fast, accuracy-independent inputs."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterFlowRunId
from prefect.variables import Variable
from qcsc_prefect.integrations.qiskit.qrmi.contract import decode_result, resolve_shots

from .pr42 import FIXTURE_DIR
from .pr42_source_proof import verify_workflow_copy
from .recovery import load_fulqrum_solver, read_json, require, validate_result


async def resolved(value):
    return await value if inspect.isawaitable(value) else value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


async def read_run_artifacts(flow_run_id):
    async with get_client() as client:
        artifacts = []
        while True:
            page = await client.read_artifacts(
                artifact_filter=ArtifactFilter(
                    flow_run_id=ArtifactFilterFlowRunId(any_=[flow_run_id])
                ),
                limit=100,
                offset=len(artifacts),
            )
            artifacts.extend(page)
            if len(page) < 100:
                return artifacts


def scheduler_records(artifacts, keys):
    records = []
    for artifact in artifacts:
        if artifact.key not in keys:
            continue
        table = json.loads(artifact.data) if isinstance(artifact.data, str) else artifact.data
        require(isinstance(table, list) and len(table) == 2, "Unexpected scheduler artifact table")
        record = dict(zip(table[0], table[1], strict=True))
        require(
            record["state"] == "COMPLETED"
            and str(record["exit_code"]) == "0"
            and str(record["signal"]) == "0",
            f"Scheduler job failed: {record}",
        )
        records.append({"artifact_key": artifact.key, **record})
    return records


async def run_smoke(*, solver_block: str, work_dir: str, parameter_file: str | Path | None = None):
    """Invoke the original flow body, keeping checks and evidence outside it.

    This creates one uniquely named sampler-options Variable. No existing blocks
    or Variables are changed. A dedicated, otherwise idle solver root is required
    to identify its new native output directories without editing the solver code.
    """
    from . import pr42_workflow as original

    proof = verify_workflow_copy()
    require(version("qcsc-prefect-qiskit") == "0.3.0", "Expected QrmiRuntime 0.3.0")
    payload = read_json(
        Path(parameter_file)
        if parameter_file
        else Path(__file__).with_name("pr42_smoke_parameters.json")
    )
    parameters = original.Parameters.model_validate(payload["parameters"])
    require(
        parameters.quantum_source == "real-device", "This integration check requires QRMI samples"
    )
    options = payload["sampler_options"]
    require("shots" in options.get("params", {}), "Specify shots explicitly for the smoke run")
    shots = resolve_shots(None, options)
    solver, _ = await load_fulqrum_solver(solver_block)
    solver_root = Path(solver.root_dir).expanduser().resolve()
    before = set(solver_root.glob("job_*"))
    root = Path(work_dir).expanduser().resolve() / f"run_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    print(f"PR #42 runtime smoke directory: {root}", flush=True)
    write_json(root / "source_proof.json", proof)
    write_json(root / "inputs.json", payload)
    shutil.copyfile(FIXTURE_DIR / "sqd_fulqrum_main.py.txt", root / "original_main.py.txt")
    shutil.copyfile(Path(__file__).with_name("pr42_workflow.py"), root / "adapted_main.py.txt")
    shutil.copyfile(Path(__file__).with_name("pr42_qrmi.patch"), root / "pr42_qrmi.patch")
    runtime_file = root / "runtime.json"
    write_json(
        runtime_file,
        {
            "root_dir": str(root / "quantum"),
            "resource": "test_heron",
            "scheduler": "slurm",
            "command_block_name": "poc-qrmi-command",
            "execution_profile_block_name": "poc-qrmi-execution",
            "hpc_profile_block_name": "poc-qrmi-hpc",
        },
    )
    phase = "options"
    try:
        option_name = f"pr42_qrmi_smoke_{uuid4().hex}"
        await resolved(Variable.set(option_name, options, overwrite=False))
        write_json(
            root / "execution.json", {"solver_block": solver_block, "option_name": option_name}
        )
        phase = "original-flow"
        state = await original.sqd_2405_05068_fulqrum.with_options(
            retries=0,
            persist_result=False,
        )(
            parameters=parameters,
            runner_name=str(runtime_file),
            solver_name=solver_block,
            option_name=option_name,
            cache_compute_integrals=False,
            cache_sampling=False,
            return_state=True,
        )
        energy = float(await resolved(state.result()))
        require(
            state.is_completed() and np.isfinite(energy),
            "Original flow did not return a finite energy",
        )
        phase = "workflow-check"
        artifacts = await read_run_artifacts(state.state_details.flow_run_id)
        jobs = scheduler_records(artifacts, {"qrmi-slurm-job", solver.metrics_artifact_key})
        quantum = read_json(root / "quantum_stage.json")
        quantum_id = str(quantum["job"]["job_id"])
        require(
            sum(str(job["job_id"]) == quantum_id for job in jobs) == 1,
            "Missing quantum scheduler evidence",
        )
        classical = [job for job in jobs if str(job["job_id"]) != quantum_id]
        require(bool(classical), "Missing classical scheduler evidence")
        outcome_tables = [
            artifact for artifact in artifacts if artifact.key == original.ARTIFACT_KEY
        ]
        require(len(outcome_tables) == 1, "Missing original configuration-recovery artifact")
        records = outcome_tables[0].data
        records = json.loads(records) if isinstance(records, str) else records
        require(bool(records), "No original recovery records")
        require(
            np.isclose(energy, min(record["energy"] for record in records), rtol=0, atol=1e-12),
            "Flow result and its own recovery records disagree",
        )
        require(len(classical) == len(records), "Solver job/record count mismatch")
        new_directories = sorted(set(solver_root.glob("job_*")) - before)
        require(
            len(new_directories) == len(classical),
            "Solver root must be dedicated to this validation run",
        )
        from qcsc_prefect_fulqrum.io_utils import read_solver_input, read_solver_output
        from qcsc_prefect_fulqrum.solver_job import solution_to_sci_result

        native = []
        for directory in new_directories:
            problem = read_solver_input(directory)
            solution = read_solver_output(directory, problem["norb"])
            validate_result(
                solution_to_sci_result(solution, norb=problem["norb"], nelec=problem["nelec"]),
                problem["norb"],
                problem["nelec"],
            )
            require(
                bool(solution.metadata.get("fulqrum_version")), "Missing native Fulqrum metadata"
            )
            native.append(
                {
                    "work_dir": str(directory),
                    "electronic_energy": float(solution.energy),
                    "metadata": solution.metadata,
                }
            )
        job_dir = Path(quantum["job"]["work_dir"])
        request = read_json(job_dir / "sampler_request.json")
        result = decode_result(job_dir / "result.json", shots, request["registers"])
        meas = result[0].data.meas
        selected = (
            meas.get_bitstrings(result[0].data.test.bitcount() == 0)
            if parameters.circuit.use_reset_mitigation
            else meas.get_bitstrings()
        )
        require(bool(selected), "No retained quantum samples")
        (root / "bitstrings.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
        summary = {
            "status": "completed",
            "validation": "workflow-passed",
            "accuracy_checked": False,
            "source_commit": proof["source_commit"],
            "work_dir": str(root),
            "flow_run_id": str(state.state_details.flow_run_id),
            "flow_name": original.FLOW_NAME,
            "qiskit_integration_version": version("qcsc-prefect-qiskit"),
            "requested_shots": shots,
            "retained_shots": len(selected),
            "total_energy": energy,
            "source_proof": proof,
            "scheduler_jobs": jobs,
            "native_outputs": native,
            "recovery_records": records,
            "option_name": option_name,
        }
        write_json(root / "workflow_summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            root / "failure.json",
            {"phase": phase, "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "proof", help="Verify the exact patch and unchanged calculation definitions"
    )
    run = commands.add_parser(
        "run", help="Run the adapted original flow; submits quantum and classical jobs"
    )
    run.add_argument("--solver-block", required=True)
    run.add_argument("--work-dir", default="/shared/qrmi-pr42-original-smoke")
    run.add_argument("--parameter-file", type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = verify_workflow_copy() if command == "proof" else asyncio.run(run_smoke(**args))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
