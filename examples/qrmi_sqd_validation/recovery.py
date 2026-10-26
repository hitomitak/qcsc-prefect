"""Replay saved QRMI H2 samples through pinned recovery and an explicit solver."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
from prefect.logging import disable_run_logger
from pyscf import lib
from qcsc_prefect.integrations.qiskit.qrmi.contract import decode_result
from qiskit.primitives.containers import BitArray
from qiskit_addon_sqd.counts import bit_array_to_arrays
from qiskit_addon_sqd.fermion import SCIResult

from .pinned_recovery import load_recovery, verify_sources
from .pr42 import FIXTURE_DIR, chem, source_metadata

PIN = "2d435b19af4648cbbc5fbc6a27ff258c93817d1a"
PIN_HASH = "f5078060a1b8227a401e757bd811c569823a490cf9c2f42eaf9f4858d098cce7"
TRANSPILER = "examples.qrmi_sqd_validation.pr42:transpile_for_target"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def positive_integer(value, name):
    require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class RecoverySettings:
    subspace_dim: int = 4
    num_batches: int = 1
    max_iterations: int = 1
    seed: int = 24

    def __post_init__(self):
        for name in ["subspace_dim", "num_batches", "max_iterations"]:
            positive_integer(getattr(self, name), name)
        require(type(self.seed) is int and self.seed >= 0, "seed must be a nonnegative integer")


@dataclass(frozen=True)
class SavedSampling:
    run_dir: Path
    bits: BitArray
    requested_shots: int
    reset_mitigation: bool
    hashes: dict[str, str]


def load_saved_h2_run(directory: str | Path) -> SavedSampling:
    """Validate full saved evidence, including the exact retained shot order.

    Paths within the saved run are resolved relative to its current location,
    allowing a completed run to be copied from the cluster for offline replay.
    """
    root = Path(directory).expanduser().resolve()
    summary = read_json(root / "sampling_summary.json")
    require(summary["molecule"] == "H2" and summary["basis"] == "sto-3g", "Expected H2/STO-3G")
    require(summary["qiskit_integration_version"] == "0.3.0", "Expected sampling library 0.3.0")
    require(summary["runtime"] == "qrmi" and summary["scheduler"] == "slurm", "Expected QRMI/Slurm")
    require(summary["quantum_source"] == "real-device", "Expected saved real-device samples")
    require(summary["source"]["commit"] == PIN, "Wrong sampling source commit")
    require(summary["source"]["sha256"] == PIN_HASH, "Wrong sampling source hash")
    require(
        hashlib.sha256((FIXTURE_DIR / "sqd_fulqrum_main.py.txt").read_bytes()).hexdigest()
        == PIN_HASH,
        "Local sampling fixture hash mismatch",
    )
    require(summary["num_bits"] == 4, "Expected four-bit H2 samples")
    reset = summary["reset_mitigation"]
    require(type(reset) is bool, "reset_mitigation must be a boolean")
    requested = positive_integer(summary["requested_shots"], "requested_shots")
    retained = positive_integer(summary["returned_shots"], "returned_shots")
    require(retained <= requested, "Returned shots exceed requested shots")
    jobs = list((root / "quantum").glob("*/slurm_job.json"))
    require(len(jobs) == 1, "Expected exactly one saved quantum job")
    job_dir = jobs[0].parent
    job = read_json(jobs[0])
    request = read_json(job_dir / "sampler_request.json")
    config = read_json(job_dir / "transpile_config.json")
    report = read_json(job_dir / "transpile_report.json")
    require(job["state"] == "COMPLETED" and job["exit_status"] == 0, "Sampling job did not succeed")
    require(Path(job["work_dir"]).name == job_dir.name, "Saved job directory mismatch")
    require(
        Path(job["work_dir"]).parent == Path(summary["quantum_root_dir"]), "Job provenance mismatch"
    )
    require(job["mode"] == report["mode"] == "batch", "Expected Batch sampling")
    registers = {"meas": 4, **({"test": 4} if reset else {})}
    require(
        request["schema_version"] == 1 and request["shots"] == requested, "Wrong sampler request"
    )
    require(request["registers"] == report["registers"] == registers, "Wrong sampling registers")
    for record in [job, request, report]:
        require(record["transpiler"] == TRANSPILER, "Wrong sampling transpiler")
        require(record["resource"] == request["resource"], "Sampling resource mismatch")
    require(config["source_commit"] == report["source_commit"] == PIN, "Compiler source mismatch")
    require(
        config["source_sha256"] == report["source_sha256"] == PIN_HASH, "Compiler hash mismatch"
    )
    require(config["transport"] == "ffsim-pre-init-decomposed", "Wrong circuit preparation")
    require(config["circuit_parameters"]["use_reset_mitigation"] is reset, "Reset setting mismatch")
    result = decode_result(job_dir / "result.json", requested, registers)
    meas = result[0].data.meas
    selected = (
        meas.get_bitstrings(result[0].data.test.bitcount() == 0) if reset else meas.get_bitstrings()
    )
    require(len(selected) == retained, "Retained shot count does not match raw test selection")
    saved = (root / "bitstrings.txt").read_text(encoding="utf-8").splitlines()
    require(saved == selected, "Saved bitstrings differ from raw selection or shot order")
    bits = BitArray.from_samples(saved, num_bits=4)
    require(bits.get_counts() == summary["counts"], "Saved counts differ from raw selection")
    files = [root / "sampling_summary.json", root / "bitstrings.txt"]
    files += [
        job_dir / name
        for name in [
            "slurm_job.json",
            "sampler_request.json",
            "transpile_config.json",
            "transpile_report.json",
            "result.json",
            "circuit.qpy",
            "isa.qpy",
            "sampler_input.json",
        ]
    ]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return SavedSampling(root, bits, requested, reset, hashes)


def validate_result(result, norb, nelec):
    require(isinstance(result, SCIResult), "Solver must return SCIResult")
    require(np.isfinite(result.energy), "Solver returned a nonfinite energy")
    require(len(result.orbital_occupancies) == 2, "Expected alpha/beta occupancies")
    for occupation, electrons in zip(result.orbital_occupancies, nelec, strict=True):
        values = np.asarray(occupation)
        require(values.shape == (norb,) and np.isfinite(values).all(), "Invalid solver occupancies")
        require(np.all((values >= -1e-8) & (values <= 1 + 1e-8)), "Occupancies outside [0, 1]")
        require(
            np.isclose(values.sum(), electrons, atol=1e-7, rtol=0), "Wrong occupancy electron count"
        )


async def load_block(block_class, name):
    block = block_class.load(name)
    return await block if inspect.isawaitable(block) else block


async def load_fulqrum_solver(solver_block):
    """Check the classical solver route before any new quantum submission."""
    from qcsc_prefect_blocks.common.blocks import HPCProfileBlock
    from qcsc_prefect_fulqrum import FulqrumSolverJob

    require(bool(solver_block), "fulqrum-block requires solver_block")
    solver = await load_block(FulqrumSolverJob, solver_block)
    hpc = await load_block(HPCProfileBlock, solver.hpc_profile_block_name)
    require(
        hpc.hpc_target == "slurm" and not hpc.slurm_qpu,
        "Fulqrum recovery requires a classical Slurm profile without slurm_qpu",
    )
    return solver, hpc


async def make_solver(backend, properties, solver_block):
    if backend == "reference":
        from qiskit_addon_sqd.fermion import solve_fermion

        async def solve(ci_strings):
            energy, state, occupations, _ = solve_fermion(
                ci_strings,
                properties.one_body_tensor,
                properties.two_body_tensor,
                spin_sq=None,
                verbose=0,
            )
            return SCIResult(energy=energy, sci_state=state, orbital_occupancies=occupations)

        return solve
    if backend == "fulqrum-local":
        from qcsc_prefect_fulqrum import solve_fulqrum
        from qcsc_prefect_fulqrum.solver_job import solution_to_sci_result

        async def solve(ci_strings):
            solution = solve_fulqrum(
                ci_strings,
                properties.one_body_tensor,
                properties.two_body_tensor,
                properties.num_orbitals,
                properties.num_electrons,
            )
            return solution_to_sci_result(
                solution, norb=properties.num_orbitals, nelec=properties.num_electrons
            )

        return solve
    if backend == "fulqrum-block":
        from qcsc_prefect_executor.from_blocks import run_job_from_blocks
        from qcsc_prefect_fulqrum.io_utils import (
            make_job_work_dir,
            read_solver_output,
            write_solver_input,
        )
        from qcsc_prefect_fulqrum.solver_job import solution_to_sci_result

        solver, _ = await load_fulqrum_solver(solver_block)
        job_directories = []

        async def solve(ci_strings):
            job_dir = make_job_work_dir(Path(solver.root_dir).expanduser().resolve())
            job_directories.append(str(job_dir))
            write_solver_input(
                work_dir=job_dir,
                ci_strings=ci_strings,
                one_body_tensor=properties.one_body_tensor,
                two_body_tensor=properties.two_body_tensor,
                norb=properties.num_orbitals,
                nelec=properties.num_electrons,
                options=solver.solver_options(),
            )
            job = await run_job_from_blocks(
                command_block_name=solver.command_block_name,
                execution_profile_block_name=solver.execution_profile_block_name,
                hpc_profile_block_name=solver.hpc_profile_block_name,
                work_dir=job_dir,
                script_filename=solver.script_filename,
                user_args=[str(job_dir)],
                watch_poll_interval=5.0,
                timeout_seconds=solver.timeout_seconds,
                metrics_artifact_key=solver.metrics_artifact_key,
            )
            record = {
                "job_id": job.job_id,
                "state": job.state,
                "exit_status": job.exit_status,
                "work_dir": str(job_dir),
            }
            (job_dir / "slurm_job.json").write_text(json.dumps(record, indent=2))
            require(
                job.state == "COMPLETED" and job.exit_status == 0,
                f"Fulqrum job failed; see {job_dir}",
            )
            solution = read_solver_output(job_dir, properties.num_orbitals)
            return solution_to_sci_result(
                solution, norb=properties.num_orbitals, nelec=properties.num_electrons
            )

        solve.job_directories = job_directories
        return solve
    raise ValueError(f"Unsupported solver backend: {backend}")


async def recover_saved_h2(
    sampling_run: str | Path,
    output_root: str | Path,
    *,
    settings: RecoverySettings | None = None,
    backend: str,
    solver_block: str | None = None,
    solve=None,
) -> dict:
    """Replay completed sampling evidence without a quantum runtime or resampling.

    ``reference`` is a local PySCF comparison, ``fulqrum-local`` uses the native
    backend, and ``fulqrum-block`` explicitly submits classical solver jobs.
    Tests may inject a solver only with the truthful ``injected`` backend label.
    """
    settings = settings or RecoverySettings()
    require(version("qcsc-prefect-qiskit") == "0.3.0", "Expected runtime environment 0.3.0")
    require(
        backend in {"reference", "fulqrum-local", "fulqrum-block", "injected"}, "Unknown backend"
    )
    require(
        (solve is not None) == (backend == "injected"),
        "Injected solvers require backend='injected'",
    )
    require(backend != "fulqrum-block" or bool(solver_block), "fulqrum-block requires solver_block")
    saved = load_saved_h2_run(sampling_run)
    provenance = verify_sources()
    recovery = load_recovery()
    output_root = Path(output_root).expanduser().resolve()
    require(
        not output_root.is_relative_to(saved.run_dir),
        "Keep recovery output outside the sampling run",
    )
    work_dir = output_root / f"recovery_{uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (work_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

    write(
        "input_validation.json",
        {
            "sampling_run": str(saved.run_dir),
            "source": source_metadata(),
            "requested_shots": saved.requested_shots,
            "retained_shots": saved.bits.num_shots,
            "reset_mitigation": saved.reset_mitigation,
            "input_sha256": saved.hashes,
            "recovery_source": provenance,
            "settings": asdict(settings),
            "backend": backend,
        },
    )
    selected_solver = None
    try:
        with lib.with_omp_threads(1), disable_run_logger():
            properties = chem.compute_molecular_integrals_from_geometry.fn(
                atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"
            )
        np.savez_compressed(
            work_dir / "integrals.npz",
            one_body_tensor=properties.one_body_tensor,
            two_body_tensor=properties.two_body_tensor,
            nuclear_repulsion_energy=properties.nuclear_repulsion_energy,
            nelec=properties.num_electrons,
        )
        selected_solver = solve or await make_solver(backend, properties, solver_block)
        counter = 0

        async def recorded_solve(ci_strings):
            nonlocal counter
            index = counter
            counter += 1
            request = {
                "alpha": [int(x) for x in ci_strings[0]],
                "beta": [int(x) for x in ci_strings[1]],
                "status": "pending",
            }
            write(f"solve_{index:04d}.json", request)
            result = await selected_solver(ci_strings)
            validate_result(result, properties.num_orbitals, properties.num_electrons)
            write(
                f"solve_{index:04d}.json",
                {
                    **request,
                    "status": "completed",
                    "electronic_energy": float(result.energy),
                    "orbital_occupancies": [
                        np.asarray(x).tolist() for x in result.orbital_occupancies
                    ],
                },
            )
            return result

        raw_bitstrings, raw_probs = bit_array_to_arrays(saved.bits)
        np.savez_compressed(
            work_dir / "recovery_input.npz", bitstrings=raw_bitstrings, probabilities=raw_probs
        )
        outcome = await recovery.run_configuration_recovery(
            raw_bitstrings,
            raw_probs,
            solve=recorded_solve,
            initial_occupancies=properties.initial_occupancy,
            nelec=properties.num_electrons,
            open_shell=properties.open_shell,
            subspace_dim=settings.subspace_dim,
            num_batches=settings.num_batches,
            max_iterations=settings.max_iterations,
            rng=np.random.default_rng(settings.seed),
            nuclear_repulsion_energy=properties.nuclear_repulsion_energy,
        )
        require(np.isfinite(outcome.best_energy), "Recovery returned a nonfinite total energy")
        for name, digest in saved.hashes.items():
            require(
                hashlib.sha256((saved.run_dir / name).read_bytes()).hexdigest() == digest,
                f"Sampling evidence changed during recovery: {name}",
            )
        summary = {
            "status": "completed",
            "backend": backend,
            "solver_block": solver_block,
            "sampling_run": str(saved.run_dir),
            "work_dir": str(work_dir),
            "source_commit": PIN,
            "qiskit_integration_version": version("qcsc-prefect-qiskit"),
            "requested_shots": saved.requested_shots,
            "retained_shots": saved.bits.num_shots,
            "settings": asdict(settings),
            "solver_calls": counter,
            "electronic_energy": float(outcome.best_result.energy),
            "nuclear_repulsion_energy": float(properties.nuclear_repulsion_energy),
            "total_energy": float(outcome.best_energy),
            "round_energies": outcome.round_energies,
            "records": outcome.records,
            "solver_work_dirs": getattr(selected_solver, "job_directories", []),
        }
        write("recovery_summary.json", summary)
        return summary
    except Exception as exc:
        write(
            "failure.json",
            {
                "status": "failed",
                "backend": backend,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "solver_work_dirs": getattr(selected_solver, "job_directories", []),
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--backend", required=True, choices=["reference", "fulqrum-local", "fulqrum-block"]
    )
    parser.add_argument("--solver-block")
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args(argv)
    settings = RecoverySettings(args.subspace_dim, args.num_batches, args.max_iterations, args.seed)
    execute = recover_saved_h2
    if args.backend == "fulqrum-block":
        from prefect import flow

        execute = flow(execute, name="qrmi-pr42-saved-recovery", retries=0, persist_result=False)
    result = asyncio.run(
        execute(
            args.sampling_run,
            args.output_root,
            settings=settings,
            backend=args.backend,
            solver_block=args.solver_block,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
