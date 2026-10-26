"""Fresh H2 orchestration with real algorithms and fake quantum/native boundaries."""

from __future__ import annotations

import asyncio
import importlib
import ipaddress
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("ffsim")
pytest.importorskip("qcsc_workflow_utility")
pytest.importorskip("qcsc_prefect_fulqrum")

from prefect.context import TaskRunContext
from prefect.testing.utilities import prefect_test_harness
from qcsc_prefect.integrations.qiskit.qrmi import runtime as quantum_runtime
from qcsc_prefect.integrations.qiskit.qrmi import worker
from qcsc_prefect_blocks.common.blocks import HPCProfileBlock
from qcsc_prefect_executor import from_blocks
from qcsc_prefect_fulqrum import FulqrumSolverJob, cli
from qcsc_prefect_fulqrum.solver import FulqrumSolution
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_ibm_runtime import RuntimeEncoder

pipeline = importlib.import_module("examples.qrmi_sqd_validation.end_to_end")
sampling = importlib.import_module("examples.qrmi_sqd_validation.integration")
LOGGER = SimpleNamespace(info=lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def allowed(sock, address):
        if sock.family != socket.AF_UNIX:
            host = address[0]
            assert host == "localhost" or ipaddress.ip_address(host).is_loopback

    def guarded_connect(sock, address):
        allowed(sock, address)
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        allowed(sock, address)
        return connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


@pytest.fixture
def boundaries(tmp_path, monkeypatch):
    solver = FulqrumSolverJob(root_dir=str(tmp_path / "classical"), script_filename="h2.slurm")
    quantum_hpc = HPCProfileBlock(hpc_target="slurm", slurm_qpu="test_heron")
    classical_hpc = HPCProfileBlock(hpc_target="slurm", queue_cpu="debug")
    monkeypatch.setattr(FulqrumSolverJob, "load", lambda name: solver)
    monkeypatch.setattr(
        HPCProfileBlock,
        "load",
        lambda name: quantum_hpc if name == "poc-qrmi-hpc" else classical_hpc,
    )
    target = GenericBackendV2(num_qubits=6, basis_gates=["rz", "sx", "x", "cz"], seed=42).target
    service = ModuleType("qrmi")
    service.QRMIService = lambda: SimpleNamespace(
        resources=lambda: [SimpleNamespace(resource_id=lambda: "test_heron")]
    )
    ibm = ModuleType("qrmi.primitives.ibm")
    ibm.get_target = lambda resource: target
    monkeypatch.setitem(sys.modules, "qrmi", service)
    monkeypatch.setitem(sys.modules, "qrmi.primitives", ModuleType("qrmi.primitives"))
    monkeypatch.setitem(sys.modules, "qrmi.primitives.ibm", ibm)
    monkeypatch.setenv("test_heron_QRMI_IBM_QCS_SESSION_MODE", "batch")
    monkeypatch.setenv("test_heron_QRMI_JOB_ACQUISITION_TOKEN", "test-token")
    state = SimpleNamespace(
        calls=[],
        solver=solver,
        quantum_hpc=quantum_hpc,
        classical_hpc=classical_hpc,
        fail=None,
        reject_all=False,
        task_contexts=[],
    )

    def fake_readouts(isa, shots, executable, resource):
        # Synthetic bit strings, not a QPU simulation. Keep both alpha/beta orbitals.
        kept = ["1101", "0010", "1100", "0001", "0110", "1110", "1001", "1100"]
        meas = (kept + ["0011"] * shots)[:shots]
        data = {"meas": BitArray.from_samples(meas, num_bits=4)}
        if any(reg.name == "test" for reg in isa.cregs):
            test = (["0000"] * 8 + ["0001"] * shots)[:shots]
            if state.reject_all:
                test = ["0001"] * shots
            data["test"] = BitArray.from_samples(test, num_bits=4)
        Path("sampler_input.json").write_text(json.dumps(worker.make_sampler_payload(isa, shots)))
        result = PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})])
        Path("result.json").write_text(json.dumps(result, cls=RuntimeEncoder))

    monkeypatch.setattr(worker, "execute_task_runner", fake_readouts)

    def reference_core(ci_strings, one_body_tensor, two_body_tensor, norb, nelec, options):
        energy, sci, occupations, _ = solve_fermion(
            ci_strings, one_body_tensor, two_body_tensor, verbose=0
        )
        return FulqrumSolution(
            energy=energy,
            orbital_occupancies=occupations,
            ci_strs_a=sci.ci_strs_a,
            ci_strs_b=sci.ci_strs_b,
            amplitudes=sci.amplitudes,
            metadata={
                "fulqrum_version": "test-double",
                "subspace_dim": 4,
                "solver_path": "test-reference",
            },
        )

    monkeypatch.setattr(cli, "solve_fulqrum", reference_core)

    async def quantum_submit(**kwargs):
        state.calls.append("quantum")
        state.task_contexts.append(TaskRunContext.get())
        if state.fail == "quantum":
            return SimpleNamespace(job_id="41", state="FAILED", exit_status=1)
        with monkeypatch.context() as local:
            local.chdir(kwargs["work_dir"])
            worker.main(kwargs["user_args"])
        return SimpleNamespace(job_id="41", state="COMPLETED", exit_status=0)

    async def classical_submit(**kwargs):
        state.calls.append("classical")
        if state.fail == "classical":
            return SimpleNamespace(job_id="42", state="FAILED", exit_status=1)
        (kwargs["work_dir"] / kwargs["script_filename"]).write_text(
            "#!/bin/bash\n# CPU test stub\n"
        )
        assert cli.main([str(kwargs["work_dir"])]) == 0
        return SimpleNamespace(job_id="42", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(quantum_runtime, "run_job_from_blocks", quantum_submit)
    monkeypatch.setattr(from_blocks, "run_job_from_blocks", classical_submit)
    return state


@pytest.fixture
def offline_body(boundaries, monkeypatch):
    monkeypatch.setattr(quantum_runtime, "_sample_on_slurm", quantum_runtime._sample_on_slurm.fn)
    monkeypatch.setattr(quantum_runtime, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(sampling, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(sampling.chem, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(pipeline, "get_run_logger", lambda: LOGGER)
    return boundaries


def run_body(tmp_path, reset=True, **kwargs):
    return asyncio.run(
        pipeline.h2_qrmi_fulqrum_flow.fn(
            solver_block="test-solver",
            shots=100,
            work_dir=str(tmp_path / "runs"),
            reset_mitigation=reset,
            **kwargs,
        )
    )


@pytest.mark.parametrize("reset", [False, True], ids=["meas-100", "reset-retains-8"])
def test_fresh_samples_to_recovery_solver_and_fci(tmp_path, offline_body, reset):
    summary = run_body(tmp_path, reset)
    assert offline_body.calls == ["quantum", "classical"]
    assert summary["validation"] == "passed"
    assert summary["retained_shots"] == (8 if reset else 100)
    assert summary["absolute_energy_error"] < 1e-9
    assert summary["quantum_input_files_unchanged"] == 10
    root = Path(summary["work_dir"])
    assert json.loads((root / "integration_summary.json").read_text()) == summary
    assert pipeline.verify_h2_run(root) == summary
    assert offline_body.calls == ["quantum", "classical"]


@pytest.mark.parametrize(
    "failure", ["quantum", "classical", "all-rejected", "qpu-profile", "quantum-profile"]
)
def test_failures_stop_without_resampling(tmp_path, offline_body, failure):
    offline_body.fail = failure
    offline_body.reject_all = failure == "all-rejected"
    if failure == "qpu-profile":
        offline_body.classical_hpc.slurm_qpu = "test_heron"
    if failure == "quantum-profile":
        offline_body.quantum_hpc.slurm_qpu = None
    with pytest.raises((ValueError, RuntimeError)):
        run_body(tmp_path)
    (root,) = (tmp_path / "runs").iterdir()
    diagnostic = json.loads((root / "failure.json").read_text())
    expected = {
        "quantum": ["quantum"],
        "classical": ["quantum", "classical"],
        "all-rejected": ["quantum"],
        "qpu-profile": [],
        "quantum-profile": [],
    }
    assert offline_body.calls == expected[failure]
    assert (
        diagnostic["phase"]
        == {
            "quantum": "sampling",
            "classical": "recovery",
            "all-rejected": "sampling",
            "qpu-profile": "preflight",
            "quantum-profile": "preflight",
        }[failure]
    )
    assert not (root / "integration_summary.json").exists()


@pytest.mark.parametrize("change", ["input", "nuclear", "native-version"])
def test_saved_verification_detects_modified_evidence(tmp_path, offline_body, change):
    summary = run_body(tmp_path)
    if change == "input":
        path = Path(summary["sampling_run"]) / "bitstrings.txt"
        path.write_text(path.read_text()[::-1])
    elif change == "nuclear":
        path = Path(summary["recovery_run"]) / "recovery_summary.json"
        data = json.loads(path.read_text())
        data["total_energy"] += data["nuclear_repulsion_energy"]
        path.write_text(json.dumps(data))
    else:
        path = Path(summary["solver_jobs"][0]["work_dir"]) / "result.json"
        data = json.loads(path.read_text())
        data["metadata"].pop("fulqrum_version")
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, RuntimeError)):
        pipeline.verify_h2_run(summary["work_dir"])
    assert offline_body.calls == ["quantum", "classical"]


def test_verify_cli_outside_checkout_has_no_network_or_jobs(tmp_path, offline_body):
    summary = run_body(tmp_path)
    code = """
import runpy, socket
def reject(*args, **kwargs):
    raise AssertionError('Saved verification attempted network access')
socket.socket.connect = reject
socket.socket.connect_ex = reject
socket.create_connection = reject
runpy.run_module('examples.qrmi_sqd_validation.end_to_end', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, "verify", summary["work_dir"]],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["validation"] == "passed"


def test_real_parent_flow_and_sampling_task(tmp_path, monkeypatch, boundaries):
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_API_KEY", "")
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    with prefect_test_harness():
        summary = asyncio.run(
            pipeline.h2_qrmi_fulqrum_flow(
                solver_block="test-solver",
                shots=100,
                reset_mitigation=True,
                work_dir=str(tmp_path / "runs"),
            )
        )
    assert summary["validation"] == "passed" and summary["retained_shots"] == 8
    assert boundaries.calls == ["quantum", "classical"]
    assert len(boundaries.task_contexts) == 1
    context = boundaries.task_contexts[0]
    assert context is not None and context.task.name == "quantum-sampling-qrmi-slurm"
