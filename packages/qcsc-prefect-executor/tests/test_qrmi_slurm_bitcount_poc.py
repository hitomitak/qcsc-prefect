"""Exercise the PoC's CLI boundary and file handoff without a QPU or Slurm."""

from __future__ import annotations

import asyncio
import json
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")

from qcsc_prefect.integrations.qiskit.qrmi import runtime as runtime_mod  # noqa: E402
from qcsc_prefect.integrations.qiskit.qrmi import worker as node_mod  # noqa: E402
from qiskit import QuantumCircuit, qpy  # noqa: E402
from qiskit.primitives import PrimitiveResult, SamplerPubResult  # noqa: E402
from qiskit.primitives.containers import BitArray, DataBin  # noqa: E402
from qiskit_ibm_runtime import RuntimeEncoder  # noqa: E402

from examples.prefect_bitcount_demo import (  # noqa: E402
    create_qrmi_slurm_blocks as setup_mod,
)
from examples.prefect_bitcount_demo import flow_optimized as bitcount_mod  # noqa: E402
from examples.prefect_bitcount_demo import flow_qrmi_slurm as flow_mod  # noqa: E402
from examples.prefect_bitcount_demo import qrmi_slurm_runtime as example_runtime  # noqa: E402

SAMPLES = ["0000000000", "1111111111", "0000000000", "1010101010"]


def make_result(samples=SAMPLES):
    return PrimitiveResult(
        [SamplerPubResult(DataBin(meas=BitArray.from_samples(samples)), metadata={})],
        metadata={},
    )


def logger_stub():
    return SimpleNamespace(info=lambda *args, **kwargs: None)


@pytest.mark.parametrize("mode", [None, "dedicated"])
def test_batch_guard_rejects_missing_or_dedicated_mode(monkeypatch, mode):
    monkeypatch.delenv("test_heron_QRMI_IBM_QCS_SESSION_MODE", raising=False)
    if mode is not None:
        monkeypatch.setenv("test_heron_QRMI_IBM_QCS_SESSION_MODE", mode)
    with pytest.raises(RuntimeError, match="SESSION_MODE"):
        node_mod.require_batch_environment("test_heron")


def test_batch_guard_requires_spank_acquisition(monkeypatch):
    monkeypatch.setenv("test_heron_QRMI_IBM_QCS_SESSION_MODE", "batch")
    monkeypatch.delenv("test_heron_QRMI_JOB_ACQUISITION_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="acquisition token"):
        node_mod.require_batch_environment("test_heron")
    monkeypatch.setenv("test_heron_QRMI_JOB_ACQUISITION_TOKEN", "test-token")
    node_mod.require_batch_environment("test_heron")


def test_node_invokes_actual_cli_contract_and_decodes_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = tmp_path / "task_runner"
    encoded = json.dumps(make_result(), cls=RuntimeEncoder)
    runner.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "resource, source, destination = sys.argv[1:]\n"
        "assert resource == 'test_heron'\n"
        "payload = json.loads(Path(source).read_text())\n"
        "assert payload['program_id'] == 'sampler'\n"
        "assert payload['parameters']['shots'] == 4\n"
        "assert payload['parameters']['pubs'][0][1:] == [None, 4]\n"
        "assert 'OPENQASM 3.0' in payload['parameters']['pubs'][0][0]\n"
        # Match the service's format switch: a plain dict is not a PrimitiveResult.
        f"result = {encoded!r} if payload['parameters']['support_qiskit'] else '{{}}'\n"
        "Path(destination).write_text(result)\n",
        encoding="utf-8",
    )
    runner.chmod(0o700)
    circuit = QuantumCircuit(10)
    circuit.measure_all()

    node_mod.execute_task_runner(circuit, 4, runner, "test_heron")

    payload = json.loads((tmp_path / "sampler_input.json").read_text())
    assert payload["parameters"]["pubs"][0][1:] == [None, 4]
    assert payload["parameters"]["support_qiskit"] is True
    assert (tmp_path / "result.json").is_file()


@pytest.mark.parametrize(
    "result, message",
    [
        ({}, "returned dict; expected PrimitiveResult"),
        (PrimitiveResult([], metadata={}), "returned 0 PUBs; expected 1"),
        (make_result(SAMPLES[:2]), "returned 2 shots; expected 4"),
    ],
)
def test_node_reports_unexpected_result_and_preserves_raw_file(
    tmp_path, monkeypatch, result, message
):
    monkeypatch.chdir(tmp_path)
    encoded = json.dumps(result, cls=RuntimeEncoder)

    def fake_run(*args, **kwargs):
        Path("result.json").write_text(encoded)

    monkeypatch.setattr(node_mod.subprocess, "run", fake_run)
    circuit = QuantumCircuit(10)
    circuit.measure_all()
    with pytest.raises(RuntimeError, match=message) as exc:
        node_mod.execute_task_runner(circuit, 4, tmp_path / "task_runner", "test_heron")
    assert str(tmp_path / "result.json") in str(exc.value)
    assert (tmp_path / "result.json").read_text() == encoded


def test_node_rejects_successful_cli_without_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(node_mod.subprocess, "run", lambda *args, **kwargs: None)
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    with pytest.raises(FileNotFoundError):
        node_mod.execute_task_runner(circuit, 4, tmp_path / "task_runner", "test_heron")


def test_node_transpiles_qpy_before_calling_task_runner(tmp_path, monkeypatch):
    from types import ModuleType

    from qiskit.providers.fake_provider import GenericBackendV2

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("test_heron_QRMI_IBM_QCS_SESSION_MODE", "batch")
    monkeypatch.setenv("test_heron_QRMI_JOB_ACQUISITION_TOKEN", "test-token")
    circuit = QuantumCircuit(10)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    Path("sampler_request.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "resource": "test_heron",
                "shots": 4,
                "registers": {"meas": 10},
                "transpiler": runtime_mod.DEFAULT_TRANSPILER,
            }
        )
    )
    Path("transpile_config.json").write_text("{}")
    with Path("circuit.qpy").open("wb") as stream:
        qpy.dump(circuit, stream)

    resource = SimpleNamespace(resource_id=lambda: "test_heron")
    qrmi = ModuleType("qrmi")
    qrmi.QRMIService = lambda: SimpleNamespace(resources=lambda: [resource])
    ibm = ModuleType("qrmi.primitives.ibm")
    target = GenericBackendV2(num_qubits=10, basis_gates=["rz", "sx", "x", "cx"]).target
    ibm.get_target = lambda res: target
    monkeypatch.setitem(sys.modules, "qrmi", qrmi)
    monkeypatch.setitem(sys.modules, "qrmi.primitives", ModuleType("qrmi.primitives"))
    monkeypatch.setitem(sys.modules, "qrmi.primitives.ibm", ibm)
    monkeypatch.setattr(sys, "argv", ["qrmi_task_runner.py"])
    calls = []
    monkeypatch.setattr(node_mod, "execute_task_runner", lambda *args: calls.append(args))

    node_mod.main()

    isa, shots, executable, resource_id = calls[0]
    assert resource_id == "test_heron"
    assert "h" not in isa.count_ops()
    assert isa.num_clbits == 10
    assert shots == 4
    assert executable == Path(sys.executable).with_name("task_runner")


def test_sampler_sends_qpy_to_existing_executor_and_returns_shots(tmp_path, monkeypatch):
    async def fake_run(**kwargs):
        work_dir = kwargs["work_dir"]
        with (work_dir / "circuit.qpy").open("rb") as stream:
            (circuit,) = qpy.load(stream)
        assert circuit.num_qubits == 10
        assert circuit.num_clbits == 10
        assert kwargs["hpc_profile_block_name"] == "poc-qrmi-hpc"
        assert kwargs["user_args"] == ["--request", "sampler_request.json"]
        (work_dir / "result.json").write_text(json.dumps(make_result(), cls=RuntimeEncoder))
        return SimpleNamespace(job_id="6", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", fake_run)
    monkeypatch.setattr(runtime_mod, "get_run_logger", logger_stub)
    monkeypatch.setattr(runtime_mod, "_sample_on_slurm", runtime_mod._sample_on_slurm.fn)
    circuit = QuantumCircuit(10)
    circuit.measure_all()

    result = asyncio.run(example_runtime.QrmiSlurmRuntime(tmp_path).sampler([(circuit,)], shots=4))

    assert result[0].data.meas.get_bitstrings() == SAMPLES
    assert result.metadata["qcsc_slurm"]["job_id"] == "6"
    assert Path(result.metadata["qcsc_slurm"]["work_dir"], "slurm_job.json").is_file()


@pytest.mark.parametrize(
    "state,exit_status,write_result",
    [
        ("FAILED", 1, True),
        ("CANCELLED", 0, True),
        ("COMPLETED", 0, False),
    ],
)
def test_sampler_rejects_failed_or_missing_result(
    tmp_path, monkeypatch, state, exit_status, write_result
):
    async def fake_run(**kwargs):
        if write_result:
            (kwargs["work_dir"] / "result.json").write_text(
                json.dumps(make_result(), cls=RuntimeEncoder)
            )
        return SimpleNamespace(job_id="6", state=state, exit_status=exit_status)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", fake_run)
    monkeypatch.setattr(runtime_mod, "_sample_on_slurm", runtime_mod._sample_on_slurm.fn)
    circuit = QuantumCircuit(10)
    circuit.measure_all()
    with pytest.raises(RuntimeError):
        asyncio.run(example_runtime.QrmiSlurmRuntime(tmp_path).sampler([(circuit,)], shots=4))


def test_bitcount_rejects_slurm_failure_even_with_zero_exit_status(tmp_path, monkeypatch):
    async def fake_run(**kwargs):
        return bitcount_mod.SlurmRunResult(
            job_id="9", exit_status=0, state="FAILED", job_status={"State": "FAILED"}
        )

    monkeypatch.setattr(bitcount_mod, "run_job_from_blocks", fake_run)
    with pytest.raises(RuntimeError, match="job 9 failed: state=FAILED, exit_status=0") as exc:
        asyncio.run(
            bitcount_mod.run_hpc_bitcount_from_input.fn(
                command_block_name="poc-bitcount-command",
                execution_profile_block_name="poc-bitcount-execution",
                hpc_profile_block_name="poc-bitcount-hpc",
                job_work_dir=str(tmp_path),
                script_filename="bitcount.slurm",
            )
        )
    assert "slurmd log" in str(exc.value)


def test_bitcount_reports_missing_histogram_with_job_id(tmp_path, monkeypatch):
    async def fake_run(**kwargs):
        return bitcount_mod.SlurmRunResult(
            job_id="10", exit_status=0, state="COMPLETED", job_status={"State": "COMPLETED"}
        )

    monkeypatch.setattr(bitcount_mod, "run_job_from_blocks", fake_run)
    with pytest.raises(RuntimeError, match="job 10 produced no histogram") as exc:
        asyncio.run(
            bitcount_mod.run_hpc_bitcount_from_input.fn(
                command_block_name="poc-bitcount-command",
                execution_profile_block_name="poc-bitcount-execution",
                hpc_profile_block_name="poc-bitcount-hpc",
                job_work_dir=str(tmp_path),
                script_filename="bitcount.slurm",
            )
        )
    assert str(tmp_path / "hist_u64.bin") in str(exc.value)


def test_flow_passes_real_shots_to_mpi_and_checks_histogram(tmp_path, monkeypatch):
    async def fake_sampler(self, pubs, *, shots):
        assert pubs[0][0].num_qubits == 10
        assert shots == 4
        result = make_result()
        result.metadata["qcsc_slurm"] = {"job_id": "6"}
        return result

    async def fake_bitcount(**kwargs):
        assert kwargs["hpc_profile_block_name"] == "poc-bitcount-hpc"
        mpi_dir = Path(kwargs["job_work_dir"])
        values = array("I")
        values.frombytes((mpi_dir / "input.bin").read_bytes())
        assert list(values) == [int(bits, 2) for bits in SAMPLES]
        return {"job_id": "7", "counts": dict(flow_mod.Counter(SAMPLES))}

    artifacts = []

    async def fake_artifact(**kwargs):
        artifacts.append(kwargs)

    monkeypatch.setattr(example_runtime.QrmiSlurmRuntime, "sampler", fake_sampler)
    monkeypatch.setattr(flow_mod, "run_hpc_bitcount_from_input", fake_bitcount)
    monkeypatch.setattr(flow_mod, "create_table_artifact", fake_artifact)
    monkeypatch.setattr(flow_mod, "get_run_logger", logger_stub)

    summary = asyncio.run(flow_mod.qrmi_slurm_bitcount_flow.fn(shots=4, work_dir=str(tmp_path)))

    assert summary["quantum_slurm_job_id"] == "6"
    assert summary["mpi_slurm_job_id"] == "7"
    assert summary["counts_match"] is True
    assert sum(summary["counts"].values()) == 4
    assert json.loads(Path(summary["work_dir"], "summary.json").read_text()) == summary
    assert artifacts[0]["key"] == "qrmi-slurm-bitcount-counts"


def test_mismatched_mpi_histogram_is_a_failure():
    with pytest.raises(RuntimeError, match="does not match"):
        flow_mod.check_counts(SAMPLES, {"0000000000": 4})


def test_setup_separates_quantum_and_classical_resources(tmp_path, monkeypatch):
    example_dir = tmp_path / "example"
    binary = example_dir / "bin/get_counts_hist"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(setup_mod, "__file__", str(example_dir / "create_qrmi_slurm_blocks.py"))
    saved = {}

    def save(self, name, *, overwrite):
        assert overwrite is True
        saved[name] = self

    for cls in (setup_mod.CommandBlock, setup_mod.ExecutionProfileBlock, setup_mod.HPCProfileBlock):
        monkeypatch.setattr(cls, "save", save)
    setup_mod.create_blocks()

    assert len(saved) == 6
    assert saved["poc-qrmi-hpc"].slurm_qpu == "test_heron"
    assert saved["poc-bitcount-hpc"].slurm_qpu is None
    assert saved["poc-bitcount-execution"].num_nodes == 2
    assert saved["poc-bitcount-execution"].launcher == "mpirun"
    assert saved["poc-qrmi-execution"].launcher == "single"
