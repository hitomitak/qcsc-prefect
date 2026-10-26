"""Saved replay tests: no QPU calls, pinned recovery, real reference numerics."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
from importlib.metadata import requires, version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ffsim")
pytest.importorskip("pydantic_numpy")
pytest.importorskip("qcsc_workflow_utility")
pytest.importorskip("qcsc_prefect_fulqrum")
from prefect.logging import disable_run_logger
from pyscf import fci, lib
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime
from qiskit import ClassicalRegister, QuantumCircuit, qpy
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit_addon_sqd.fermion import SCIResult, solve_fermion
from qiskit_ibm_runtime import RuntimeEncoder

replay = importlib.import_module("examples.qrmi_sqd_validation.recovery")
pinned = importlib.import_module("examples.qrmi_sqd_validation.pinned_recovery")


def dump(path, data):
    path.write_text(json.dumps(data, indent=2))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("Saved recovery must not use network or quantum sampling")

    for name in ["connect", "connect_ex"]:
        monkeypatch.setattr(socket.socket, name, reject)
    monkeypatch.setattr(socket, "create_connection", reject)
    monkeypatch.setattr(QrmiRuntime, "sampler", reject)


@pytest.fixture(scope="module")
def h2():
    with lib.with_omp_threads(1), disable_run_logger():
        return replay.chem.compute_molecular_integrals_from_geometry.fn(
            atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"
        )


@pytest.fixture
def saved_run(tmp_path):
    def create(reset=True):
        # Synthetic order matching reported counts, not the user's unavailable files.
        kept = ["1101", "0010", "1100", "0001", "0110", "1110", "1001", "1100"]
        meas = kept + ["0011"] * 92
        test = ["0000"] * 8 + ["0001"] * 92
        root = tmp_path / f"sampling-{reset}" / "run_fixture"
        job_dir = root / "quantum" / "job_fixture"
        job_dir.mkdir(parents=True)
        recorded_root = Path("/shared/original/run_fixture/quantum")
        registers = {"meas": 4, **({"test": 4} if reset else {})}
        data = {"meas": BitArray.from_samples(meas, num_bits=4)}
        if reset:
            data["test"] = BitArray.from_samples(test, num_bits=4)
        result = PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})])
        (job_dir / "result.json").write_text(json.dumps(result, cls=RuntimeEncoder))
        source = replay.source_metadata()
        selected = kept if reset else meas
        dump(
            root / "sampling_summary.json",
            {
                "molecule": "H2",
                "basis": "sto-3g",
                "qiskit_integration_version": "0.3.0",
                "runtime": "qrmi",
                "scheduler": "slurm",
                "quantum_source": "real-device",
                "source": source,
                "num_bits": 4,
                "requested_shots": 100,
                "returned_shots": len(selected),
                "reset_mitigation": reset,
                "counts": BitArray.from_samples(selected).get_counts(),
                "work_dir": "/shared/original/run_fixture",
                "quantum_root_dir": str(recorded_root),
            },
        )
        (root / "bitstrings.txt").write_text("\n".join(selected) + "\n")
        common = {"resource": "test_heron", "transpiler": replay.TRANSPILER}
        dump(
            job_dir / "slurm_job.json",
            {
                **common,
                "job_id": "16",
                "state": "COMPLETED",
                "exit_status": 0,
                "mode": "batch",
                "work_dir": str(recorded_root / "job_fixture"),
            },
        )
        dump(
            job_dir / "sampler_request.json",
            {
                **common,
                "schema_version": 1,
                "shots": 100,
                "registers": registers,
            },
        )
        config = {
            "source_commit": source["commit"],
            "source_sha256": source["sha256"],
            "transport": "ffsim-pre-init-decomposed",
            "circuit_parameters": {"use_reset_mitigation": reset},
        }
        dump(job_dir / "transpile_config.json", config)
        dump(
            job_dir / "transpile_report.json",
            {
                **common,
                **config,
                "mode": "batch",
                "registers": registers,
                "host": "offline",
            },
        )
        dump(
            job_dir / "sampler_input.json", {"program_id": "sampler", "parameters": {"shots": 100}}
        )
        circuit = QuantumCircuit(4)
        if reset:
            circuit.add_register(ClassicalRegister(4, "test"))
            circuit.measure(range(4), circuit.cregs[0])
        circuit.measure_all()
        for name in ["circuit.qpy", "isa.qpy"]:
            with (job_dir / name).open("wb") as stream:
                qpy.dump(circuit, stream)
        return root

    return create


def input_hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("reset", [False, True], ids=["meas", "eight-retained"])
def test_saved_h2_recovery_matches_fci_without_resampling(saved_run, tmp_path, h2, reset):
    root = saved_run(reset)
    original = input_hashes(root)
    summary = asyncio.run(replay.recover_saved_h2(root, tmp_path / "outputs", backend="reference"))
    reference = fci.direct_spin1.kernel(
        h2.one_body_tensor,
        h2.two_body_tensor,
        h2.num_orbitals,
        h2.num_electrons,
    )[0]
    assert summary["backend"] == "reference"
    assert summary["retained_shots"] == (8 if reset else 100)
    assert summary["requested_shots"] == 100
    assert summary["solver_calls"] == 1
    assert summary["records"][0]["dimension"] == 4
    assert summary["electronic_energy"] == pytest.approx(reference, abs=1e-9)
    assert summary["total_energy"] == pytest.approx(
        reference + h2.nuclear_repulsion_energy, abs=1e-9
    )
    assert summary["total_energy"] == pytest.approx(
        summary["electronic_energy"] + summary["nuclear_repulsion_energy"], abs=1e-12
    )
    assert input_hashes(root) == original
    work = Path(summary["work_dir"])
    assert (work / "input_validation.json").is_file()
    assert (work / "integrals.npz").is_file()
    assert (work / "solve_0000.json").is_file()


def test_second_recovery_round(saved_run, tmp_path):
    summary = asyncio.run(
        replay.recover_saved_h2(
            saved_run(),
            tmp_path / "outputs",
            settings=replay.RecoverySettings(max_iterations=2),
            backend="reference",
        )
    )
    assert summary["solver_calls"] == 2
    assert len(summary["round_energies"]) == 2
    assert {r["round"] for r in summary["records"]} == {0, 1}


@pytest.mark.parametrize(
    "change", ["shot-order", "summary-counts", "failed-job", "wrong-version", "raw-test"]
)
def test_bad_evidence_fails_before_solver_or_output(saved_run, tmp_path, change):
    root = saved_run()
    job = root / "quantum/job_fixture"
    if change == "shot-order":
        lines = (root / "bitstrings.txt").read_text().splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        (root / "bitstrings.txt").write_text("\n".join(lines) + "\n")
    elif change in {"summary-counts", "wrong-version"}:
        path = root / "sampling_summary.json"
        data = json.loads(path.read_text())
        if change == "summary-counts":
            data["counts"] = {"0000": 8}
        else:
            data["qiskit_integration_version"] = "0.2.5"
        dump(path, data)
    elif change == "failed-job":
        path = job / "slurm_job.json"
        dump(path, {**json.loads(path.read_text()), "exit_status": 1})
    else:
        result = replay.decode_result(job / "result.json", 100, {"meas": 4, "test": 4})
        altered = PrimitiveResult(
            [
                SamplerPubResult(
                    DataBin(
                        meas=result[0].data.meas,
                        test=BitArray.from_samples(["0001"] * 100, num_bits=4),
                    ),
                    metadata={},
                )
            ]
        )
        (job / "result.json").write_text(json.dumps(altered, cls=RuntimeEncoder))

    async def reject(_):
        pytest.fail("Invalid input must not call the solver")

    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(
            replay.recover_saved_h2(root, tmp_path / "outputs", backend="injected", solve=reject)
        )
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("failure", ["exception", "energy", "occupancies"])
def test_solver_failure_retains_diagnostics(saved_run, tmp_path, failure):
    async def fail(_):
        if failure == "exception":
            raise RuntimeError("Solver unavailable")
        return SCIResult(
            energy=float("nan") if failure == "energy" else -1.0,
            sci_state=None,
            orbital_occupancies=(np.zeros(1), np.zeros(1))
            if failure == "occupancies"
            else (np.array([1.0, 0.0]), np.array([1.0, 0.0])),
        )

    root = saved_run()
    original = input_hashes(root)
    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(
            replay.recover_saved_h2(root, tmp_path / "outputs", backend="injected", solve=fail)
        )
    (work,) = (tmp_path / "outputs").iterdir()
    assert json.loads((work / "failure.json").read_text())["status"] == "failed"
    assert (work / "solve_0000.json").is_file()
    assert not (work / "recovery_summary.json").exists()
    assert input_hashes(root) == original


@pytest.fixture
def block_boundary(tmp_path, monkeypatch):
    from qcsc_prefect_blocks.common.blocks import HPCProfileBlock
    from qcsc_prefect_fulqrum import FulqrumSolverJob

    solver = FulqrumSolverJob(root_dir=str(tmp_path / "solver-jobs"))
    monkeypatch.setattr(FulqrumSolverJob, "load", lambda _: solver)
    monkeypatch.setattr(HPCProfileBlock, "load", lambda _: HPCProfileBlock(hpc_target="slurm"))
    return solver


def test_fulqrum_block_v030_executor_file_protocol(
    saved_run, tmp_path, monkeypatch, block_boundary
):
    from qcsc_prefect_executor import from_blocks
    from qcsc_prefect_fulqrum import cli, io_utils
    from qcsc_prefect_fulqrum.solver import FulqrumSolution

    calls = []

    def reference_cli(ci_strings, one_body_tensor, two_body_tensor, norb, nelec, options):
        energy, state, occupations, _ = solve_fermion(
            ci_strings, one_body_tensor, two_body_tensor, verbose=0
        )
        return FulqrumSolution(
            energy=energy,
            orbital_occupancies=occupations,
            ci_strs_a=state.ci_strs_a,
            ci_strs_b=state.ci_strs_b,
            amplitudes=state.amplitudes,
            metadata={"subspace_dim": 4, "solver_path": "test-reference"},
        )

    monkeypatch.setattr(cli, "solve_fulqrum", reference_cli)

    async def fake_submit(**kwargs):
        calls.append(kwargs)
        assert cli.main([str(kwargs["work_dir"])]) == 0
        return SimpleNamespace(job_id="31", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(from_blocks, "run_job_from_blocks", fake_submit)
    summary = asyncio.run(
        replay.recover_saved_h2(
            saved_run(), tmp_path / "outputs", backend="fulqrum-block", solver_block="test-solver"
        )
    )
    assert len(calls) == 1
    assert calls[0]["hpc_profile_block_name"] == block_boundary.hpc_profile_block_name
    assert summary["solver_calls"] == 1
    job = Path(summary["solver_work_dirs"][0])
    assert io_utils.read_solver_output(job, 2).energy == pytest.approx(summary["electronic_energy"])
    assert json.loads((job / "slurm_job.json").read_text())["state"] == "COMPLETED"


def test_fulqrum_failed_job_is_rejected(saved_run, tmp_path, monkeypatch, block_boundary):
    from qcsc_prefect_executor import from_blocks

    async def fail(**kwargs):
        return SimpleNamespace(job_id="32", state="FAILED", exit_status=1)

    monkeypatch.setattr(from_blocks, "run_job_from_blocks", fail)
    with pytest.raises(ValueError, match="Fulqrum job failed"):
        asyncio.run(
            replay.recover_saved_h2(
                saved_run(),
                tmp_path / "outputs",
                backend="fulqrum-block",
                solver_block="test-solver",
            )
        )
    (work,) = (tmp_path / "outputs").iterdir()
    assert json.loads((work / "failure.json").read_text())["solver_work_dirs"]


def test_fulqrum_qpu_profile_is_rejected(saved_run, tmp_path, monkeypatch, block_boundary):
    from qcsc_prefect_blocks.common.blocks import HPCProfileBlock

    monkeypatch.setattr(
        HPCProfileBlock,
        "load",
        lambda _: HPCProfileBlock(hpc_target="slurm", slurm_qpu="test_heron"),
    )
    with pytest.raises(ValueError, match="without slurm_qpu"):
        asyncio.run(
            replay.recover_saved_h2(
                saved_run(),
                tmp_path / "outputs",
                backend="fulqrum-block",
                solver_block="test-solver",
            )
        )


def test_dependency_pins_and_immutable_sources():
    assert version("qcsc-prefect-fulqrum") == "0.3.0.dev0"
    dependencies = requires("qcsc-prefect-fulqrum")
    for name in ["blocks", "core", "executor"]:
        assert f"qcsc-prefect-{name}==0.3.0" in dependencies
    assert pinned.verify_sources()["commit"] == replay.PIN


def test_modified_pinned_recovery_is_rejected(saved_run, tmp_path, monkeypatch):
    original = pinned.FIXTURE_DIR
    fixtures = tmp_path / "snapshot"
    shutil.copytree(original, fixtures)
    manifest = json.loads((fixtures / "postprocessing.json").read_text())
    for record in manifest["files"]:
        record["local_path"] = str(pinned.ROOT / record["local_path"])
        if record["path"].endswith("sqd_fulqrum/recovery.py"):
            source = fixtures / "changed.py.txt"
            source.write_text("raise RuntimeError('Changed source')\n")
            record["local_path"] = str(source)
    dump(fixtures / "postprocessing.json", manifest)
    monkeypatch.setattr(pinned, "FIXTURE_DIR", fixtures)
    with pytest.raises(ValueError, match="source hash mismatch"):
        asyncio.run(replay.recover_saved_h2(saved_run(), tmp_path / "outputs", backend="reference"))
    assert not (tmp_path / "outputs").exists()


def test_missing_native_backend_has_no_reference_fallback(saved_run, tmp_path, monkeypatch):
    import qcsc_prefect_fulqrum

    def unavailable(*args, **kwargs):
        raise ModuleNotFoundError("Native Fulqrum unavailable")

    monkeypatch.setattr(qcsc_prefect_fulqrum, "solve_fulqrum", unavailable)
    with pytest.raises(ModuleNotFoundError, match="Native Fulqrum unavailable"):
        asyncio.run(
            replay.recover_saved_h2(saved_run(), tmp_path / "outputs", backend="fulqrum-local")
        )
    (work,) = (tmp_path / "outputs").iterdir()
    assert json.loads((work / "failure.json").read_text())["backend"] == "fulqrum-local"
    assert not (work / "recovery_summary.json").exists()


def test_reference_cli_from_outside_checkout(saved_run, tmp_path):
    # Run the installed module with network and QPU access prohibited in the child too.
    code = """
import runpy
import socket
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime
def reject(*args, **kwargs):
    raise AssertionError('Offline recovery attempted network or quantum sampling')
socket.socket.connect = reject
socket.socket.connect_ex = reject
socket.create_connection = reject
QrmiRuntime.sampler = reject
runpy.run_module('examples.qrmi_sqd_validation.recovery', run_name='__main__')
"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PREFECT_API_URL="http://127.0.0.1:1/api"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--sampling-run",
            str(saved_run()),
            "--output-root",
            str(tmp_path / "cli-output"),
            "--backend",
            "reference",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["backend"] == "reference"
    assert summary["retained_shots"] == 8
    assert summary["status"] == "completed"


def test_native_fulqrum_if_installed(saved_run, tmp_path, h2):
    pytest.importorskip("fulqrum", reason="Native Fulqrum >=0.5.0 needs an OpenMP toolchain")
    summary = asyncio.run(
        replay.recover_saved_h2(saved_run(), tmp_path / "native", backend="fulqrum-local")
    )
    reference = (
        fci.direct_spin1.kernel(h2.one_body_tensor, h2.two_body_tensor, 2, (1, 1))[0]
        + h2.nuclear_repulsion_energy
    )
    assert summary["backend"] == "fulqrum-local"
    assert summary["total_energy"] == pytest.approx(reference, abs=1e-9)
