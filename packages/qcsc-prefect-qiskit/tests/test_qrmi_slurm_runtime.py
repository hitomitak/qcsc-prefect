"""Library contracts: no Slurm, QRMI service, Prefect server or QPU calls."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import socket
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from qcsc_prefect.integrations.qiskit.qrmi import PreparedCircuit, QrmiSlurmRuntime, worker
from qcsc_prefect.integrations.qiskit.qrmi import runtime as runtime_mod
from qiskit import ClassicalRegister, QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import RuntimeEncoder


def measured_circuit():
    circuit = QuantumCircuit(2)
    circuit.add_register(ClassicalRegister(2, "readout"), ClassicalRegister(1, "test"))
    circuit.measure(0, circuit.cregs[1][0])
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], circuit.cregs[0])
    return circuit


def make_result(shots=4, registers=None):
    if registers is None:
        registers = {"readout": 2, "test": 1}
    return PrimitiveResult(
        [
            SamplerPubResult(
                DataBin(
                    **{
                        name: BitArray.from_samples(["0" * width] * shots, num_bits=width)
                        for name, width in registers.items()
                    }
                ),
                metadata={},
            )
        ],
        metadata={},
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject_network(*args, **kwargs):
        pytest.fail("QRMI library tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(runtime_mod, "_sample_on_slurm", runtime_mod._sample_on_slurm.fn)
    monkeypatch.setattr(
        runtime_mod, "get_run_logger", lambda: SimpleNamespace(info=lambda *a: None)
    )


@pytest.fixture
def runtime(tmp_path):
    return QrmiSlurmRuntime(
        root_dir=tmp_path / "jobs",
        command_block_name="sampler-command",
        execution_profile_block_name="quantum-execution",
        hpc_profile_block_name="quantum-hpc",
        resource="custom_resource",
        timeout_seconds=120,
    )


@pytest.fixture
def submitted(monkeypatch):
    calls = []

    async def submit(**kwargs):
        calls.append(kwargs)
        (kwargs["work_dir"] / "result.json").write_text(
            json.dumps(make_result(), cls=RuntimeEncoder)
        )
        return SimpleNamespace(job_id="73", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    return calls


def test_pinned_pr42_sampler_call_is_accepted(runtime, submitted):
    """Execute the actual await expression from the immutable PR source."""
    fixture = Path(__file__).parent / "fixtures/pr42"
    provenance = json.loads((fixture / "source.json").read_text())
    source = (fixture / "sqd_fulqrum_main.py.txt").read_bytes()
    assert provenance["commit"] == "2d435b19af4648cbbc5fbc6a27ff258c93817d1a"
    assert hashlib.sha256(source).hexdigest() == provenance["sha256"]
    call = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Await)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "sampler"
    )
    harness = ast.parse("async def invoke(runtime, isa_circuit, options):\n    return None\n")
    harness.body[0].body[0].value = call
    namespace = {}
    exec(compile(ast.fix_missing_locations(harness), "pinned-pr42-sampler-call", "exec"), namespace)
    result = asyncio.run(namespace["invoke"](runtime, measured_circuit(), {"params": {"shots": 4}}))
    assert result[0].data.readout.num_shots == 4
    assert result[0].data.test.num_bits == 1
    request = json.loads((submitted[0]["work_dir"] / "sampler_request.json").read_text())
    assert request["shots"] == 4
    assert request["resource"] == "custom_resource"
    assert submitted[0]["timeout_seconds"] == 120
    assert submitted[0]["command_block_name"] == "sampler-command"
    assert submitted[0]["execution_profile_block_name"] == "quantum-execution"
    assert submitted[0]["hpc_profile_block_name"] == "quantum-hpc"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shots": 0},
        {"shots": -1},
        {"shots": True},
        {"shots": 1.5},
        {"shots": "4"},
        {"options": []},
        {"options": {"params": []}},
        {"options": {"params": {"shots": False}}},
        {"options": {"params": {"shots": 0}}},
        {"options": {"params": {"shots": 4}, "resilience_level": 1}},
        {"options": {"params": {"shots": 4, "other": True}}},
        {"shots": 5, "options": {"params": {"shots": 4}}},
    ],
)
def test_bad_options_fail_before_files_or_submission(runtime, submitted, kwargs):
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(runtime.sampler([(measured_circuit(),)], **kwargs))
    assert submitted == []
    assert not runtime.root_dir.exists()


@pytest.mark.parametrize("kind", ["empty", "multiple", "params", "unbound", "unmeasured"])
def test_unsupported_pubs_fail_before_submission(runtime, submitted, kind):
    circuit = measured_circuit()
    pubs = [(circuit,)]
    if kind == "empty":
        pubs = []
    elif kind == "multiple":
        pubs *= 2
    elif kind == "params":
        pubs = [(circuit, [1])]
    elif kind == "unbound":
        circuit.rx(Parameter("theta"), 0)
    elif kind == "unmeasured":
        pubs = [(QuantumCircuit(1, 1),)]
    with pytest.raises(ValueError):
        asyncio.run(runtime.sampler(pubs, shots=4))
    assert submitted == []


def test_prepare_config_and_qpy_handoff_does_not_mutate_input(runtime, submitted):
    original = measured_circuit()

    def prepare(circuit):
        circuit.x(1)
        return PreparedCircuit(circuit, {"seed": 24})

    runtime.prepare = prepare
    result = asyncio.run(runtime.sampler([(original,)], shots=4))
    work_dir = Path(result.metadata["qcsc_slurm"]["work_dir"])
    with (work_dir / "circuit.qpy").open("rb") as stream:
        (restored,) = qpy.load(stream)
    assert "x" not in original.count_ops()
    assert restored.count_ops()["x"] == 1
    assert json.loads((work_dir / "transpile_config.json").read_text()) == {"seed": 24}
    assert result.metadata["qcsc_slurm"]["job_id"] == "73"


def test_prepare_cannot_change_registers(runtime, submitted):
    runtime.prepare = lambda _: PreparedCircuit(QuantumCircuit(1))
    with pytest.raises(ValueError):
        asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    assert submitted == []


@pytest.mark.parametrize(
    "registers,shots,message",
    [
        ({"readout": 2}, 4, "registers"),
        ({"readout": 2, "test": 2}, 4, "1 bits for test"),
        ({"readout": 2, "test": 1}, 3, "3 shots; expected 4"),
    ],
)
def test_host_rejects_invalid_results_and_retains_evidence(
    runtime, monkeypatch, registers, shots, message
):
    async def submit(**kwargs):
        (kwargs["work_dir"] / "result.json").write_text(
            json.dumps(make_result(shots, registers), cls=RuntimeEncoder)
        )
        return SimpleNamespace(job_id="73", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    (work_dir,) = runtime.root_dir.iterdir()
    assert (work_dir / "result.json").is_file()
    assert json.loads((work_dir / "slurm_job.json").read_text())["job_id"] == "73"


def test_worker_validates_every_register_even_when_cli_exits_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def zero_exit_with_missing_test(*args, **kwargs):
        Path("result.json").write_text(
            json.dumps(make_result(registers={"readout": 2}), cls=RuntimeEncoder)
        )

    monkeypatch.setattr(worker.subprocess, "run", zero_exit_with_missing_test)
    with pytest.raises(RuntimeError, match="registers"):
        worker.execute_task_runner(measured_circuit(), 4, Path("task_runner"), "custom_resource")


def test_generic_handoff_with_custom_resource_and_registers(runtime, tmp_path, monkeypatch):
    resource = SimpleNamespace(resource_id=lambda: "custom_resource")
    qrmi = ModuleType("qrmi")
    qrmi.QRMIService = lambda: SimpleNamespace(resources=lambda: [resource])
    ibm = ModuleType("qrmi.primitives.ibm")
    target = GenericBackendV2(num_qubits=3, basis_gates=["rz", "sx", "x", "cx"]).target
    ibm.get_target = lambda _: target
    monkeypatch.setitem(sys.modules, "qrmi", qrmi)
    monkeypatch.setitem(sys.modules, "qrmi.primitives", ModuleType("qrmi.primitives"))
    monkeypatch.setitem(sys.modules, "qrmi.primitives.ibm", ibm)
    monkeypatch.setenv("custom_resource_QRMI_IBM_QCS_SESSION_MODE", "batch")
    monkeypatch.setenv("custom_resource_QRMI_JOB_ACQUISITION_TOKEN", "test-token")
    runner = tmp_path / "fake_task_runner"
    runner.write_text(
        f"#!{sys.executable}\n"
        """import json, socket, sys
def reject(*args, **kwargs):
    raise RuntimeError('No network in the fake CLI')
socket.socket.connect = reject
socket.socket.connect_ex = reject
socket.create_connection = reject
from pathlib import Path
from qiskit import qasm3
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit_ibm_runtime import RuntimeEncoder
resource, source, destination = sys.argv[1:]
assert resource == 'custom_resource'
parameters = json.loads(Path(source).read_text())['parameters']
assert parameters['support_qiskit'] is True
qasm, values, shots = parameters['pubs'][0]
assert values is None and shots == 4
circuit = qasm3.loads(qasm)
assert {r.name: len(r) for r in circuit.cregs} == {'readout': 2, 'test': 1}
data = {r.name: BitArray.from_samples(['0' * len(r)] * shots, num_bits=len(r))
        for r in circuit.cregs}
result = PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})
Path(destination).write_text(json.dumps(result, cls=RuntimeEncoder))
"""
    )
    runner.chmod(0o700)

    async def submit(**kwargs):
        monkeypatch.chdir(kwargs["work_dir"])
        worker.main([*kwargs["user_args"], "--task-runner", str(runner)])
        return SimpleNamespace(job_id="73", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    result = asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    work_dir = Path(result.metadata["qcsc_slurm"]["work_dir"])
    report = json.loads((work_dir / "transpile_report.json").read_text())
    assert report["resource"] == "custom_resource"
    assert report["registers"] == {"readout": 2, "test": 1}
    assert report["target_num_qubits"] == 3
    assert result[0].data.readout.get_counts() == {"00": 4}


def test_worker_rejects_unknown_schema_before_qrmi_import(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("sampler_request.json").write_text('{"schema_version": 2}')
    with pytest.raises(ValueError, match="schema_version"):
        worker.main([])


def test_repeated_calls_create_distinct_jobs(runtime, submitted):
    asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    assert len(submitted) == 2
    assert submitted[0]["work_dir"] != submitted[1]["work_dir"]


def test_generic_library_works_without_algorithm_or_qrmi_imports():
    import subprocess
    import sys

    program = """
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'examples', 'ffsim', 'pyscf', 'qrmi',
                                      'qcsc_prefect_executor'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Reject())
from qcsc_prefect.integrations.qiskit.qrmi import QrmiSlurmRuntime
from qcsc_prefect.integrations.qiskit.qrmi.worker import main
"""
    # Test source packages in a fresh interpreter, without inherited imports.
    import os

    src = Path(__file__).parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src)}
    subprocess.run([sys.executable, "-c", program], env=env, check=True, capture_output=True)
